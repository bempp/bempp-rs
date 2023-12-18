//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.
use itertools::Itertools;
use num::{Complex, Float};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use bempp_field::{
    fft::Fft,
    types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::Fmm,
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
};
use bempp_tree::types::single_node::SingleNodeTree;

use crate::types::{FmmDataLinear, KiFmmLinear, SendPtrMut};
use rlst::{
    algorithms::{linalg::DenseMatrixLinAlgBuilder, traits::svd::Svd},
    common::traits::*,
    dense::{rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer},
};

use super::hadamard::matmul8x8x2;

// pub fn size_real(order: usize) -> usize {
//     let m = 2 * order - 1; // Size of each dimension of 3D kernel/signal
//     let pad_size = 1;
//     let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
//     p * p * (p / 2 + 1) // Number of Fourier coefficients when working with real data
// }

// pub fn nparents(level: usize) -> usize {
//     8i32.pow((level - 1) as u32) as usize
// }

fn displacements<U>(tree: &SingleNodeTree<U>, level: u64) -> Vec<Vec<usize>>
where
    U: Float + Default + Scalar<Real = U>,
{
    let parents = tree.get_keys(level - 1).unwrap();
    let nparents = parents.len();
    let nneighbors = 26; // Number of neighors for a given box

    let mut target_map = HashMap::new();

    for (i, parent) in parents.iter().enumerate() {
        target_map.insert(parent, i);
    }

    let mut result = vec![Vec::new(); nneighbors];

    let parent_neighbours = parents
        .iter()
        .map(|parent| parent.all_neighbors())
        .collect_vec();

    for i in 0..nneighbors {
        for all_neighbours in parent_neighbours.iter().take(nparents) {
            if let Some(neighbour) = all_neighbours[i] {
                result[i].push(*target_map.get(&neighbour).unwrap())
            } else {
                result[i].push(nparents);
            }
        }
    }

    result
}

/// Implement the multipole to local translation operator for an FFT accelerated KiFMM on a single node.
impl<T, U> FieldTranslation<U>
    for FmmDataLinear<KiFmmLinear<SingleNodeTree<U>, T, FftFieldTranslationKiFmm<U, T>, U>, U>
where
    T: Kernel<T = U>
        + ScaleInvariantKernel<T = U>
        + std::marker::Send
        + std::marker::Sync
        + Default,
    U: Scalar<Real = U> + Float + Default + std::marker::Send + std::marker::Sync + Fft,
    Complex<U>: Scalar,
    U: MultiplyAdd<
        U,
        VectorContainer<U>,
        VectorContainer<U>,
        VectorContainer<U>,
        Dynamic,
        Dynamic,
        Dynamic,
    >,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else {
            return;
        };
        let n = 2 * self.fmm.order - 1;
        let npad = n + 1;

        let nparents = self.fmm.tree().get_keys(level - 1).unwrap().len();
        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
        let nsiblings = 8;
        let nzeros = 8;
        let size = npad * npad * npad;
        let size_real = npad * npad * (npad / 2 + 1);
        let all_displacements = displacements(self.fmm.tree(), level);

        let ntargets = targets.len();
        let min = &targets[0];
        let max = &targets[ntargets - 1];
        let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
        let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

        let multipoles = &self.multipoles[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

        ////////////////////////////////////////////////////////////////////////////////////
        // Pre-process to setup data structures for M2L kernel
        ////////////////////////////////////////////////////////////////////////////////////

        // Allocation of FFT of multipoles on convolution grid, in frequency order
        let mut signals_hat_f_buffer = vec![U::zero(); size_real * (ntargets + nzeros) * 2];
        let signals_hat_f: &mut [Complex<U>];
        unsafe {
            let ptr = signals_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
            signals_hat_f = std::slice::from_raw_parts_mut(ptr, size_real * (ntargets + nzeros));
        }

        // A thread safe mutable pointer for saving to this vector
        let raw = signals_hat_f.as_mut_ptr();
        let signals_hat_f_ptr = SendPtrMut { raw };

        // Pre processing chunk size, in terms of number of parents
        let chunksize;
        if level == 2 {
            chunksize = 8; // Maximum size at level 2
        } else if level == 3 {
            chunksize = 64 // Maximum size at level 3
        } else {
            chunksize = 128 // Little cache benefit found beyond this
        }

        // Pre-processing to find FFT
        multipoles
            .par_chunks_exact(ncoeffs * nsiblings * chunksize)
            .enumerate()
            .for_each(|(i, multipole_chunk)| {
                // Place Signal on convolution grid
                let mut signal_chunk = vec![U::zero(); size * nsiblings * chunksize];

                for i in 0..nsiblings * chunksize {
                    let multipole = &multipole_chunk[i * ncoeffs..(i + 1) * ncoeffs];
                    let signal = &mut signal_chunk[i * size..(i + 1) * size];
                    for (surf_idx, &conv_idx) in self.fmm.m2l.surf_to_conv_map.iter().enumerate() {
                        signal[conv_idx] = multipole[surf_idx]
                    }
                }

                // Temporary buffer to hold results of FFT
                let signal_hat_chunk_buffer =
                    vec![U::zero(); size_real * nsiblings * chunksize * 2];
                let signal_hat_chunk_c;
                unsafe {
                    let ptr = signal_hat_chunk_buffer.as_ptr() as *mut Complex<U>;
                    signal_hat_chunk_c =
                        std::slice::from_raw_parts_mut(ptr, size_real * nsiblings * chunksize);
                }

                U::rfft3_fftw_slice(&mut signal_chunk, signal_hat_chunk_c, &[npad, npad, npad]);

                // Re-order the temporary buffer into frequency order before flushing to main memory
                let signal_hat_chunk_f_buffer =
                    vec![U::zero(); size_real * nsiblings * chunksize * 2];
                let signal_hat_chunk_f_c;
                unsafe {
                    let ptr = signal_hat_chunk_f_buffer.as_ptr() as *mut Complex<U>;
                    signal_hat_chunk_f_c =
                        std::slice::from_raw_parts_mut(ptr, size_real * nsiblings * chunksize);
                }

                for i in 0..size_real {
                    for j in 0..nsiblings * chunksize {
                        signal_hat_chunk_f_c[nsiblings * chunksize * i + j] =
                            signal_hat_chunk_c[size_real * j + i]
                    }
                }

                // Storing the results of the FFT in frequency order
                unsafe {
                    let sibling_offset = i * nsiblings * chunksize;

                    // Pointer to storage buffer for frequency ordered FFT of signals
                    let ptr = signals_hat_f_ptr;

                    for i in 0..size_real {
                        let frequency_offset = i * (ntargets + nzeros);

                        // Head of buffer for each frequency
                        let mut head = ptr.raw.add(frequency_offset).add(sibling_offset);

                        // Store results for this frequency for this sibling set chunk
                        let results_i = &signal_hat_chunk_f_c
                            [i * nsiblings * chunksize..(i + 1) * nsiblings * chunksize];

                        for &res in results_i {
                            *head += res;
                            head = head.add(1);
                        }
                    }
                }
            });

        // Allocate check potentials (implicitly in frequency order)
        let mut check_potentials_hat_f_buffer = vec![U::zero(); 2 * size_real * ntargets];
        let check_potentials_hat_f: &mut [Complex<U>];
        unsafe {
            let ptr = check_potentials_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
            check_potentials_hat_f = std::slice::from_raw_parts_mut(ptr, size_real * ntargets);
        }

        ////////////////////////////////////////////////////////////////////////////////////
        // M2L Kernel
        ////////////////////////////////////////////////////////////////////////////////////
        let scale = Complex::from(self.m2l_scale(level) * self.fmm.kernel.scale(level));
        let kernel_data_f = &self.fmm.m2l.operator_data.kernel_data_f;
        let max_chunksize = 512;

        (0..size_real)
            .into_par_iter()
            .zip(signals_hat_f.par_chunks_exact(ntargets + nzeros))
            .zip(check_potentials_hat_f.par_chunks_exact_mut(ntargets))
            .for_each(|((freq, signal_f), check_potential_hat_f)| {
                (0..nparents)
                    .step_by(max_chunksize)
                    .for_each(|chunk_start| {
                        let chunk_end = std::cmp::min(chunk_start + max_chunksize, nparents);

                        let save_locations =
                            &mut check_potential_hat_f[chunk_start * 8..(chunk_end) * 8];

                        for (i, kernel_f) in kernel_data_f.iter().enumerate().take(26) {
                            let frequency_offset = 64 * freq;
                            let k_f = &kernel_f[frequency_offset..(frequency_offset + 64)].to_vec();

                            // Lookup signals
                            let displacements = &all_displacements[i][chunk_start..chunk_end];

                            for j in 0..(chunk_end - chunk_start) {
                                let displacement = displacements[j];
                                let s_f = &signal_f[displacement * 8..(displacement + 1) * 8];

                                unsafe {
                                    matmul8x8x2(
                                        k_f,
                                        s_f,
                                        &mut save_locations[j * 8..(j + 1) * 8],
                                        scale,
                                    )
                                }
                            }
                        }
                    });
            });

        ////////////////////////////////////////////////////////////////////////////////////
        // Post processing to find local expansions from check potentials
        ////////////////////////////////////////////////////////////////////////////////////

        // Get check potentials back into target order from frequency order
        let mut check_potential_hat = vec![U::zero(); size_real * ntargets * 2];
        let mut check_potential = vec![U::zero(); size * ntargets];
        let check_potential_hat_c;
        unsafe {
            let ptr = check_potential_hat.as_mut_ptr() as *mut Complex<U>;
            check_potential_hat_c = std::slice::from_raw_parts_mut(ptr, size_real)
        }

        check_potential_hat_c
            .par_chunks_exact_mut(size_real)
            .enumerate()
            .for_each(|(i, check_potential_hat_chunk)| {
                // Lookup all frequencies for this target box
                for j in 0..size_real {
                    check_potential_hat_chunk[j] = check_potentials_hat_f[j * ntargets + i]
                }
            });

        // Compute inverse FFT
        U::irfft3_fftw_par_slice(
            check_potential_hat_c,
            &mut check_potential,
            &[npad, npad, npad],
        );

        // TODO: Experiment with chunk size for post processing
        check_potential
            .par_chunks_exact(nsiblings * size)
            .zip(self.level_locals[level as usize].par_chunks_exact(nsiblings))
            .for_each(|(check_potential_chunk, local_ptrs)| {
                // Map to surface grid
                let mut potential_buffer = vec![U::zero(); ncoeffs * nsiblings];
                for i in 0..nsiblings {
                    let tmp = &mut potential_buffer[i * ncoeffs..(i + 1) * ncoeffs];
                    let check_potential = &check_potential_chunk[i * size..(i + 1) * size];
                    for (surf_idx, &conv_idx) in self.fmm.m2l.conv_to_surf_map.iter().enumerate() {
                        tmp[surf_idx] = check_potential[conv_idx];
                    }
                }

                // Can now find local expansion coefficients
                let potential_chunk = unsafe {
                    rlst_pointer_mat!['a, U, potential_buffer.as_ptr(), (ncoeffs, nsiblings), (1, ncoeffs)]
                };

                let local_chunk = self
                    .fmm
                    .dc2e_inv_1
                    .dot(&self.fmm.dc2e_inv_2.dot(&potential_chunk))
                    .eval();

                local_chunk
                    .data()
                    .chunks_exact(ncoeffs)
                    .zip(local_ptrs)
                    .for_each(|(result, local)| unsafe {
                        let mut ptr = local.raw;
                        for &r in result.iter().take(ncoeffs) {
                            *ptr += r;
                            ptr = ptr.add(1)
                        }
                    });
            });
    }

    fn m2l_scale(&self, level: u64) -> U {
        if level < 2 {
            panic!("M2L only perfomed on level 2 and below")
        }

        if level == 2 {
            U::from(1. / 2.).unwrap()
        } else {
            let two = U::from(2.0).unwrap();
            Scalar::powf(two, U::from(level - 3).unwrap())
        }
    }
}

/// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
impl<T, U> FieldTranslation<U>
    for FmmDataLinear<KiFmmLinear<SingleNodeTree<U>, T, SvdFieldTranslationKiFmm<U, T>, U>, U>
where
    T: Kernel<T = U>
        + ScaleInvariantKernel<T = U>
        + std::marker::Send
        + std::marker::Sync
        + Default,
    DenseMatrixLinAlgBuilder<U>: Svd,
    U: Scalar<Real = U>,
    U: Float
        + Default
        + MultiplyAdd<
            U,
            VectorContainer<U>,
            VectorContainer<U>,
            VectorContainer<U>,
            Dynamic,
            Dynamic,
            Dynamic,
        >,
    U: std::marker::Send + std::marker::Sync + Default,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(sources) = self.fmm.tree().get_keys(level) else {
            return;
        };

        let nsources = sources.len();

        let mut source_map = HashMap::new();

        for (i, t) in sources.iter().enumerate() {
            source_map.insert(t, i);
        }

        let mut target_indices = vec![vec![-1i64; nsources]; 316];

        // Need to identify all save locations in a pre-processing step.
        for (j, source) in sources.iter().enumerate() {
            let v_list = source
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| !source.is_adjacent(pnc))
                .collect_vec();

            let transfer_vectors = v_list
                .iter()
                .map(|target| target.find_transfer_vector(source))
                .collect_vec();

            let mut transfer_vectors_map = HashMap::new();
            for (i, v) in transfer_vectors.iter().enumerate() {
                transfer_vectors_map.insert(v, i);
            }

            let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().collect();

            for (i, tv) in self.fmm.m2l.transfer_vectors.iter().enumerate() {
                if transfer_vectors_set.contains(&tv.hash) {
                    let target = &v_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                    let target_index = source_map.get(target).unwrap();
                    target_indices[i][j] = *target_index as i64;
                }
            }
        }

        // Interpret multipoles as a matrix
        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
        let multipoles = unsafe {
            rlst_pointer_mat!['a, U, self.level_multipoles[level as usize][0].raw, (ncoeffs, nsources), (1, ncoeffs)]
        };

        let (nrows, _) = self.fmm.m2l.operator_data.c.shape();
        let dim = (nrows, self.fmm.m2l.k);

        let mut compressed_multipoles = self.fmm.m2l.operator_data.st_block.dot(&multipoles);

        compressed_multipoles
            .data_mut()
            .iter_mut()
            .for_each(|d| *d *= self.fmm.kernel.scale(level) * self.m2l_scale(level));

        (0..316).into_par_iter().for_each(|c_idx| {
            let top_left = (0, c_idx * self.fmm.m2l.k);
            let c_sub = self.fmm.m2l.operator_data.c.block(top_left, dim);

            let locals = self.fmm.dc2e_inv_1.dot(
                &self.fmm.dc2e_inv_2.dot(
                    &self
                        .fmm
                        .m2l
                        .operator_data
                        .u
                        .dot(&c_sub.dot(&compressed_multipoles)),
                ),
            );

            let displacements = &target_indices[c_idx];

            for (result_idx, &save_idx) in displacements.iter().enumerate() {
                if save_idx > -1 {
                    let save_idx = save_idx as usize;
                    let mut local_ptr = self.level_locals[(level) as usize][save_idx].raw;
                    let res = &locals.data()[result_idx * ncoeffs..(result_idx + 1) * ncoeffs];

                    unsafe {
                        for &r in res.iter() {
                            *local_ptr += r;
                            local_ptr = local_ptr.add(1);
                        }
                    }
                }
            }
        })
    }

    fn m2l_scale(&self, level: u64) -> U {
        if level < 2 {
            panic!("M2L only perfomed on level 2 and below")
        }

        if level == 2 {
            U::from(1. / 2.).unwrap()
        } else {
            let two = U::from(2.0).unwrap();
            Scalar::powf(two, U::from(level - 3).unwrap())
        }
    }
}
