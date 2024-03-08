//! Multipole to Local field translations for uniform and adaptive Kernel Indepenent FMMs
use bempp_field::{
    constants::{NHALO, NSIBLINGS},
    types::FftFieldTranslationKiFmm,
};
use bempp_traits::tree::FmmTree;
use bempp_tree::types::single_node::SingleNodeTreeNew;
use itertools::Itertools;
use num::Complex;
use rayon::prelude::*;
use rlst_dense::array::Array;
use rlst_dense::base_array::BaseArray;
use rlst_dense::data_container::VectorContainer;
use std::collections::HashSet;

use bempp_traits::{field::SourceToTarget, kernel::Kernel, tree::Tree, types::Scalar};
use bempp_tree::types::morton::MortonKey;

use crate::{
    builder::FmmEvalType,
    helpers::{find_chunk_size, homogenous_kernel_scale, m2l_scale},
    types::SendPtrMut,
};
use crate::{fmm::KiFmm, traits::FmmScalar};
use rlst_dense::{
    array::empty_array,
    rlst_dynamic_array2,
    traits::{MultIntoResize, RawAccess},
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
        let targets = self.tree.get_target_tree().get_keys(level).unwrap();

        let targets_parents: HashSet<MortonKey> =
            targets.iter().map(|target| target.parent()).collect();
        let mut targets_parents = targets_parents.into_iter().collect_vec();
        targets_parents.sort();
        let ntargets_parents = targets_parents.len();

        let sources = self.tree.get_source_tree().get_keys(level).unwrap();

        let sources_parents: HashSet<MortonKey> =
            sources.iter().map(|source| source.parent()).collect();
        let mut sources_parents = sources_parents.into_iter().collect_vec();
        sources_parents.sort();
        let nsources_parents = sources_parents.len();

        let mut result = vec![Vec::new(); NHALO];

        let targets_parents_neighbors = targets_parents
            .iter()
            .map(|parent| parent.all_neighbors())
            .collect_vec();

        let zero_displacement = nsources_parents * NSIBLINGS;

        for i in 0..NHALO {
            for all_neighbors in targets_parents_neighbors.iter().take(ntargets_parents) {
                // Check if neighbor exists in a valid tree
                if let Some(neighbor) = all_neighbors[i] {
                    // If it does, check if first child exists in the source tree
                    let first_child = neighbor.first_child();
                    if let Some(neighbor_displacement) =
                        self.level_index_pointer_multipoles[level as usize].get(&first_child)
                    {
                        result[i].push(*neighbor_displacement)
                    } else {
                        result[i].push(zero_displacement)
                    }
                } else {
                    result[i].push(zero_displacement)
                }
            }
        }
        result
    }
}

impl<T, U, V> SourceToTarget for KiFmm<V, FftFieldTranslationKiFmm<U, T>, T, U>
where
    T: Kernel<T = U> + Default + Send + Sync,
    U: FmmScalar,
    Complex<U>: Scalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree<Tree = SingleNodeTreeNew<U>> + Send + Sync,
{
    fn m2l(&self, level: u64) {
        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let Some(targets) = self.tree.get_target_tree().get_keys(level) else {
                    return;
                };

                let Some(sources) = self.tree.get_source_tree().get_keys(level) else {
                    return;
                };

                // Number of target and source boxes at this level
                let ntargets = targets.len();
                let nsources = sources.len();

                // Size of convolution grid
                let nconv = 2 * self.expansion_order - 1;
                let nconv_pad = nconv + 1;

                // Find parents of targets
                let targets_parents: HashSet<MortonKey> =
                    targets.iter().map(|target| target.parent()).collect();

                let targets_parents = targets_parents.into_iter().collect_vec();
                // targets_parents.sort();
                let ntargets_parents = targets_parents.len();

                let sources_parents: HashSet<MortonKey> =
                    sources.iter().map(|source| source.parent()).collect();
                let nsources_parents = sources_parents.len();

                // Size of FFT sequence
                let fft_size = nconv_pad * nconv_pad * nconv_pad;

                // Size of real FFT sequence
                let fft_size_real = nconv_pad * nconv_pad * (nconv_pad / 2 + 1);

                // Calculate displacements of multipole data with respect to source tree
                let all_displacements = self.displacements(level);

                // Lookup multipole data from source tree
                let min = &sources[0];
                let max = &sources[nsources - 1];
                let min_idx = self.tree.get_source_tree().get_index(min).unwrap();
                let max_idx = self.tree.get_source_tree().get_index(max).unwrap();
                let multipoles =
                    &self.multipoles[min_idx * self.ncoeffs..(max_idx + 1) * self.ncoeffs];

                // Buffer to store FFT of multipole data in frequency order
                let nzeros = 8; // pad amount
                let mut signals_hat_f_buffer =
                    vec![U::zero(); fft_size_real * (nsources + nzeros) * 2];
                let signals_hat_f: &mut [Complex<U>];
                unsafe {
                    let ptr = signals_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
                    signals_hat_f =
                        std::slice::from_raw_parts_mut(ptr, fft_size_real * (nsources + nzeros));
                }

                // A thread safe mutable pointer for saving to this vector
                let signals_hat_f_ptr = SendPtrMut {
                    raw: signals_hat_f.as_mut_ptr(),
                };

                // Pre processing chunk size, in terms of number of source parents
                let max_chunk_size;
                if level == 2 {
                    max_chunk_size = 8
                } else if level == 3 {
                    max_chunk_size = 64
                } else {
                    max_chunk_size = 128
                }
                let chunk_size_pre_proc = find_chunk_size(nsources_parents, max_chunk_size);
                let chunk_size_kernel = find_chunk_size(ntargets_parents, max_chunk_size);

                // Allocate check potentials (implicitly in frequency order)
                let mut check_potentials_hat_f_buffer =
                    vec![U::zero(); 2 * fft_size_real * ntargets];
                let check_potentials_hat_f: &mut [Complex<U>];
                unsafe {
                    let ptr = check_potentials_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
                    check_potentials_hat_f =
                        std::slice::from_raw_parts_mut(ptr, fft_size_real * ntargets);
                }

                // Amount to scale the application of the kernel by
                let scale = Complex::from(m2l_scale::<U>(level) * homogenous_kernel_scale(level));

                // Lookup all of the precomputed Green's function evaluations' FFT sequences
                let kernel_data_ft = &self.source_to_target_data.operator_data.kernel_data_f;

                // Allocate buffer to store the check potentials in frequency order
                let mut check_potential_hat = vec![U::zero(); fft_size_real * ntargets * 2];

                // Allocate buffer to store the check potentials in box order
                let mut check_potential = vec![U::zero(); fft_size * ntargets];
                let check_potential_hat_c;
                unsafe {
                    let ptr = check_potential_hat.as_mut_ptr() as *mut Complex<U>;
                    check_potential_hat_c =
                        std::slice::from_raw_parts_mut(ptr, fft_size_real * ntargets)
                }

                // 1. Compute FFT of all multipoles in source boxes at this level
                {
                    multipoles
                        .par_chunks_exact(self.ncoeffs * NSIBLINGS * chunk_size_pre_proc)
                        .enumerate()
                        .for_each(|(i, multipole_chunk)| {
                            // Place Signal on convolution grid
                            let mut signal_chunk =
                                vec![U::zero(); fft_size * NSIBLINGS * chunk_size_pre_proc];

                            for i in 0..NSIBLINGS * chunk_size_pre_proc {
                                let multipole =
                                    &multipole_chunk[i * self.ncoeffs..(i + 1) * self.ncoeffs];
                                let signal = &mut signal_chunk[i * fft_size..(i + 1) * fft_size];
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
                                vec![
                                    U::zero();
                                    fft_size_real * NSIBLINGS * chunk_size_pre_proc * 2
                                ];
                            let signal_hat_chunk_c;
                            unsafe {
                                let ptr = signal_hat_chunk_buffer.as_ptr() as *mut Complex<U>;
                                signal_hat_chunk_c = std::slice::from_raw_parts_mut(
                                    ptr,
                                    fft_size_real * NSIBLINGS * chunk_size_pre_proc,
                                );
                            }

                            U::rfft3_fftw_slice(
                                &mut signal_chunk,
                                signal_hat_chunk_c,
                                &[nconv_pad, nconv_pad, nconv_pad],
                            );

                            // Re-order the temporary buffer into frequency order before flushing to main memory
                            let signal_hat_chunk_f_buffer =
                                vec![
                                    U::zero();
                                    fft_size_real * NSIBLINGS * chunk_size_pre_proc * 2
                                ];
                            let signal_hat_chunk_f_c;
                            unsafe {
                                let ptr = signal_hat_chunk_f_buffer.as_ptr() as *mut Complex<U>;
                                signal_hat_chunk_f_c = std::slice::from_raw_parts_mut(
                                    ptr,
                                    fft_size_real * NSIBLINGS * chunk_size_pre_proc,
                                );
                            }

                            for i in 0..fft_size_real {
                                for j in 0..NSIBLINGS * chunk_size_pre_proc {
                                    signal_hat_chunk_f_c[NSIBLINGS * chunk_size_pre_proc * i + j] =
                                        signal_hat_chunk_c[fft_size_real * j + i]
                                }
                            }

                            // Storing the results of the FFT in frequency order
                            unsafe {
                                let sibling_offset = i * NSIBLINGS * chunk_size_pre_proc;

                                // Pointer to storage buffer for frequency ordered FFT of signals
                                let ptr = signals_hat_f_ptr;

                                for i in 0..fft_size_real {
                                    let frequency_offset = i * (nsources + nzeros);

                                    // Head of buffer for each frequency
                                    let head = ptr.raw.add(frequency_offset).add(sibling_offset);

                                    let signal_hat_f_chunk = std::slice::from_raw_parts_mut(
                                        head,
                                        NSIBLINGS * chunk_size_pre_proc,
                                    );

                                    // Store results for this frequency for this sibling set chunk
                                    let results_i =
                                        &signal_hat_chunk_f_c[i * NSIBLINGS * chunk_size_pre_proc
                                            ..(i + 1) * NSIBLINGS * chunk_size_pre_proc];

                                    signal_hat_f_chunk
                                        .iter_mut()
                                        .zip(results_i)
                                        .for_each(|(c, r)| *c += *r);
                                }
                            }
                        });
                }

                // 2. Compute the Hadamard product
                {
                    (0..fft_size_real)
                        .into_par_iter()
                        .zip(signals_hat_f.par_chunks_exact(nsources + nzeros))
                        .zip(check_potentials_hat_f.par_chunks_exact_mut(ntargets))
                        .for_each(|((freq, signal_hat_f), check_potential_hat_f)| {
                            (0..ntargets_parents).step_by(chunk_size_kernel).for_each(
                                |chunk_start| {
                                    let chunk_end = std::cmp::min(
                                        chunk_start + chunk_size_kernel,
                                        ntargets_parents,
                                    );

                                    let save_locations = &mut check_potential_hat_f
                                        [chunk_start * NSIBLINGS..chunk_end * NSIBLINGS];

                                    for i in 0..NHALO {
                                        let frequency_offset = freq * NHALO;
                                        let k_f = &kernel_data_ft[i + frequency_offset];
                                        // Lookup signals
                                        let displacements =
                                            &all_displacements[i][chunk_start..chunk_end];

                                        for j in 0..(chunk_end - chunk_start) {
                                            let displacement = displacements[j];
                                            let s_f = &signal_hat_f
                                                [displacement..displacement + NSIBLINGS];

                                            matmul8x8(
                                                k_f,
                                                s_f,
                                                &mut save_locations
                                                    [j * NSIBLINGS..(j + 1) * NSIBLINGS],
                                                scale,
                                            )
                                        }
                                    }
                                },
                            );
                        });
                }

                // 3. Post process to find local expansions at target boxes
                {
                    check_potential_hat_c
                        .par_chunks_exact_mut(fft_size_real)
                        .enumerate()
                        .for_each(|(i, check_potential_hat_chunk)| {
                            // Lookup all frequencies for this target box
                            for j in 0..fft_size_real {
                                check_potential_hat_chunk[j] =
                                    check_potentials_hat_f[j * ntargets + i]
                            }
                        });

                    // Compute inverse FFT
                    U::irfft3_fftw_par_slice(
                        check_potential_hat_c,
                        &mut check_potential,
                        &[nconv_pad, nconv_pad, nconv_pad],
                    );

                    check_potential
                        .par_chunks_exact(NSIBLINGS * fft_size)
                        .zip(self.level_locals[level as usize].par_chunks_exact(NSIBLINGS))
                        .for_each(|(check_potential_chunk, local_ptrs)| {
                            // Map to surface grid
                            let mut potential_chunk =
                                rlst_dynamic_array2!(U, [self.ncoeffs, NSIBLINGS]);

                            for i in 0..NSIBLINGS {
                                for (surf_idx, &conv_idx) in self
                                    .source_to_target_data
                                    .conv_to_surf_map
                                    .iter()
                                    .enumerate()
                                {
                                    *potential_chunk.get_mut([surf_idx, i]).unwrap() =
                                        check_potential_chunk[i * fft_size + conv_idx];
                                }
                            }

                            // Can now find local expansion coefficients
                            let local_chunk = empty_array::<U, 2>().simple_mult_into_resize(
                                self.dc2e_inv_1.view(),
                                empty_array::<U, 2>().simple_mult_into_resize(
                                    self.dc2e_inv_2.view(),
                                    potential_chunk,
                                ),
                            );

                            local_chunk
                                .data()
                                .chunks_exact(self.ncoeffs)
                                .zip(local_ptrs)
                                .for_each(|(result, local)| {
                                    let local = unsafe {
                                        std::slice::from_raw_parts_mut(local[0].raw, self.ncoeffs)
                                    };
                                    local.iter_mut().zip(result).for_each(|(l, r)| *l += *r);
                                });
                        });
                }
            }

            FmmEvalType::Matrix(_nmatvec) => {
                panic!("unimplemnted FFT M2L for Matrix input")
            }
        }
    }

    fn p2l(&self, _level: u64) {}
}
