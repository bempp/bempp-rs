//! Multipole to Local field translations for uniform and adaptive Kernel Indepenent FMMs
use bempp_field::types::FftFieldTranslationKiFmm;
use bempp_traits::tree::FmmTree;
use bempp_tree::types::single_node::SingleNodeTreeNew;
use itertools::Itertools;
use num::{Complex, Float};
use rayon::prelude::*;
use rlst_blis::interface::gemm::Gemm;
use rlst_dense::array::Array;
use rlst_dense::base_array::BaseArray;
use rlst_dense::data_container::VectorContainer;
use std::collections::HashSet;

use bempp_field::fft::Fft;

use bempp_field::helpers::ncoeffs_kifmm;
use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData},
    fmm::InteractionLists,
    kernel::Kernel,
    tree::Tree,
    types::{EvalType, Scalar},
};
use bempp_tree::types::morton::MortonKey;

use crate::{
    builder::FmmEvaluationMode,
    constants::NSIBLINGS,
    helpers::{find_chunk_size, homogenous_kernel_scale, m2l_scale},
    types::SendPtrMut,
};
use crate::{fmm::KiFmm, traits::FmmScalar};
use rlst_dense::{
    array::empty_array,
    rlst_dynamic_array2,
    traits::{MultIntoResize, RawAccess, RawAccessMut},
};

use rlst_dense::traits::{MatrixSvd, RandomAccessMut};

use crate::field_translation::hadamard::matmul8x8;

impl<T, U, V> KiFmm<V, FftFieldTranslationKiFmm<U, T>, T, U>
where
    T: Kernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: FmmScalar,
    Complex<U>: Scalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree<Tree = SingleNodeTreeNew<U>>,
{
    fn displacements(&self, level: u64) -> Vec<Vec<usize>> {
        let nneighbors = 26;
        let nsiblings = 8;

        let sources = self.tree.get_source_tree().get_keys(level).unwrap();

        let parents: HashSet<MortonKey> = sources.iter().map(|source| source.parent()).collect();
        let mut parents = parents.into_iter().collect_vec();
        parents.sort();
        let nparents = parents.len();

        let mut result = vec![Vec::new(); nneighbors];

        let parent_neighbors = parents
            .iter()
            .map(|parent| parent.all_neighbors())
            .collect_vec();

        for i in 0..nneighbors {
            for all_neighbors in parent_neighbors.iter().take(nparents) {
                // First check if neighbor exists is in the bounds of the tree
                if let Some(neighbor) = all_neighbors[i] {
                    let first_child = neighbor.first_child();
                    // Then need to check if first child exists in the tree
                    if let Some(first_child_index) =
                        self.level_index_pointer_multipoles[level as usize].get(&first_child)
                    {
                        result[i].push(*first_child_index)
                    } else {
                        result[i].push(nparents * nsiblings)
                    }
                } else {
                    result[i].push(nparents * nsiblings);
                }
            }
        }

        result
    }
}

impl<T, U, V> SourceToTarget for KiFmm<V, FftFieldTranslationKiFmm<U, T>, T, U>
where
    T: Kernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: FmmScalar,
    Complex<U>: Scalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree<Tree = SingleNodeTreeNew<U>> + Send + Sync,
{
    fn m2l(&self, level: u64) {
        match self.eval_mode {
            FmmEvaluationMode::Vector => {
                let Some(targets) = self.tree.get_target_tree().get_keys(level) else {
                    return;
                };

                let ntargets = targets.len();

                let sources = self.tree.get_source_tree().get_keys(level).unwrap();

                // Size of convolution grid
                let n = 2 * self.expansion_order - 1;
                let npad = n + 1;

                // Find the parents of the sources
                let sources_parents: HashSet<MortonKey> =
                    sources.iter().map(|source| source.parent()).collect();

                let mut sources_parents = sources_parents.into_iter().collect_vec();
                sources_parents.sort();
                let nsources_parents = sources_parents.len();

                // Pad amount
                let nzeros = 8;

                // Number of frequencies
                let size = npad * npad * npad;
                let size_real = npad * npad * (npad / 2 + 1);

                // Calculate save displacements
                let all_displacements = self.displacements(level);

                // Lookup multipole data from source tree
                let nsources = sources.len();
                let min = &sources[0];
                let max = &sources[nsources - 1];
                let min_idx = self.tree.get_source_tree().get_index(min).unwrap();
                let max_idx = self.tree.get_source_tree().get_index(max).unwrap();
                let multipoles =
                    &self.multipoles[min_idx * self.ncoeffs..(max_idx + 1) * self.ncoeffs];

                // Preprocesss to set up data structures for kernel
                let mut signals_hat_f_buffer = vec![U::zero(); size_real * (nsources + nzeros) * 2];
                let signals_hat_f: &mut [Complex<U>];
                unsafe {
                    let ptr = signals_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
                    signals_hat_f =
                        std::slice::from_raw_parts_mut(ptr, size_real * (nsources + nzeros));
                }

                // A thread safe mutable pointer for saving to this vector
                let raw = signals_hat_f.as_mut_ptr();
                let signals_hat_f_ptr = SendPtrMut { raw };

                // Pre processing chunk size, in terms of number of parents
                let max_chunk_size;
                if level == 2 {
                    max_chunk_size = 8
                } else if level == 3 {
                    max_chunk_size = 64
                } else {
                    max_chunk_size = 128
                }
                let chunk_size = find_chunk_size(nsources_parents, max_chunk_size);

                // Find FFT of all multipoles in source tree at this level
                multipoles
                    .par_chunks_exact(self.ncoeffs * NSIBLINGS * chunk_size)
                    .enumerate()
                    .for_each(|(i, multipole_chunk)| {
                        // Place Signal on convolution grid
                        let mut signal_chunk = vec![U::zero(); size * NSIBLINGS * chunk_size];

                        for i in 0..NSIBLINGS * chunk_size {
                            let multipole =
                                &multipole_chunk[i * self.ncoeffs..(i + 1) * self.ncoeffs];
                            let signal = &mut signal_chunk[i * size..(i + 1) * size];
                            for (surf_idx, &conv_idx) in self
                                .source_to_target_data
                                .surf_to_conv_map
                                .iter()
                                .enumerate()
                            {
                                signal[conv_idx] = multipole[surf_idx]
                            }
                        }

                        // Temporary buffer to hold results of FFT
                        let signal_hat_chunk_buffer =
                            vec![U::zero(); size_real * NSIBLINGS * chunk_size * 2];
                        let signal_hat_chunk_c;
                        unsafe {
                            let ptr = signal_hat_chunk_buffer.as_ptr() as *mut Complex<U>;
                            signal_hat_chunk_c = std::slice::from_raw_parts_mut(
                                ptr,
                                size_real * NSIBLINGS * chunk_size,
                            );
                        }

                        U::rfft3_fftw_slice(
                            &mut signal_chunk,
                            signal_hat_chunk_c,
                            &[npad, npad, npad],
                        );

                        // Re-order the temporary buffer into frequency order before flushing to main memory
                        let signal_hat_chunk_f_buffer =
                            vec![U::zero(); size_real * NSIBLINGS * chunk_size * 2];
                        let signal_hat_chunk_f_c;
                        unsafe {
                            let ptr = signal_hat_chunk_f_buffer.as_ptr() as *mut Complex<U>;
                            signal_hat_chunk_f_c = std::slice::from_raw_parts_mut(
                                ptr,
                                size_real * NSIBLINGS * chunk_size,
                            );
                        }

                        for i in 0..size_real {
                            for j in 0..NSIBLINGS * chunk_size {
                                signal_hat_chunk_f_c[NSIBLINGS * chunk_size * i + j] =
                                    signal_hat_chunk_c[size_real * j + i]
                            }
                        }

                        // Storing the results of the FFT in frequency order
                        unsafe {
                            let sibling_offset = i * NSIBLINGS * chunk_size;

                            // Pointer to storage buffer for frequency ordered FFT of signals
                            let ptr = signals_hat_f_ptr;

                            for i in 0..size_real {
                                let frequency_offset = i * (nsources + nzeros);

                                // Head of buffer for each frequency
                                let head = ptr.raw.add(frequency_offset).add(sibling_offset);

                                let signal_hat_f_chunk =
                                    std::slice::from_raw_parts_mut(head, NSIBLINGS * chunk_size);

                                // Store results for this frequency for this sibling set chunk
                                let results_i = &signal_hat_chunk_f_c
                                    [i * NSIBLINGS * chunk_size..(i + 1) * NSIBLINGS * chunk_size];

                                signal_hat_f_chunk
                                    .iter_mut()
                                    .zip(results_i)
                                    .for_each(|(c, r)| *c += *r);
                            }
                        }
                    });

                // Allocate check potentials (implicitly in frequency order)
                let mut check_potentials_hat_f_buffer = vec![U::zero(); 2 * size_real * ntargets];
                let check_potentials_hat_f: &mut [Complex<U>];
                unsafe {
                    let ptr = check_potentials_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
                    check_potentials_hat_f =
                        std::slice::from_raw_parts_mut(ptr, size_real * ntargets);
                }

                // M2L kernel
                let scale = Complex {
                    re: m2l_scale::<U>(level) * homogenous_kernel_scale::<U>(level),
                    im: U::zero(),
                };
                let kernel_data_ft = &self.source_to_target_data.operator_data.kernel_data_f;

                (0..size_real)
                    .into_par_iter()
                    .zip(signals_hat_f.par_chunks_exact(ntargets + nzeros))
                    .zip(check_potentials_hat_f.par_chunks_exact_mut(ntargets))
                    .for_each(|((freq, signal_hat_f), check_potential_hat_f)| {
                        (0..nparents).step_by(chunk_size).for_each(|chunk_start| {
                            let chunk_end = std::cmp::min(chunk_start + chunk_size, nparents);

                            let save_locations = &mut check_potential_hat_f
                                [chunk_start * NSIBLINGS..chunk_end * NSIBLINGS];

                            for i in 0..26 {
                                let frequency_offset = freq * 26;
                                let k_f = &kernel_data_ft[i + frequency_offset];
                                // Lookup signals
                                let displacements = &all_displacements[i][chunk_start..chunk_end];

                                for j in 0..(chunk_end - chunk_start) {
                                    let displacement = displacements[j];
                                    let s_f = &signal_hat_f[displacement..displacement + NSIBLINGS];

                                    matmul8x8(
                                        k_f,
                                        s_f,
                                        &mut save_locations[j * NSIBLINGS..(j + 1) * NSIBLINGS],
                                        scale,
                                    )
                                }
                            }
                        });
                    });
            }
            FmmEvaluationMode::Matrix(_nmatvec) => {
                panic!("unimplemnted FFT M2L for Matrix input")
            }
        }
    }

    fn p2l(&self, level: u64) {}
}
