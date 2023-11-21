//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, RwLock},
};
use num::Zero;
use bempp_tools::Array3D;
use cauchy::c64;
use fftw::{plan::{C2RPlan64, R2CPlan64, C2RPlan, R2CPlan}, types::Flag, array::AlignedVec};
use itertools::Itertools;
use num::{Complex, Float};
use rayon::prelude::*;

use bempp_field::{
    array::pad3,
    fft::Fft,
    types::{FftFieldTranslationKiFmm, FftMatrix, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    arrays::Array3DAccess,
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, InteractionLists, SourceTranslation, TargetTranslation},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::types::{FmmData, FmmDataLinear, KiFmm, KiFmmLinear, SendPtr, SendPtrMut, SendPtrMutIter};
use rlst::{
    algorithms::{linalg::DenseMatrixLinAlgBuilder, traits::svd::Svd},
    common::traits::*,
    dense::{
        base_matrix::BaseMatrix, rlst_col_vec, rlst_dynamic_mat, rlst_pointer_mat, traits::*, Dot,
        Matrix, MultiplyAdd, Shape, VectorContainer,
    },
};
const P2M_MAX_CHUNK_SIZE: usize = 256;

/// Euclidean algorithm to find greatest common divisor less than max
fn find_chunk_size(n: usize, max_chunk_size: usize) -> usize {
    let max_divisor = max_chunk_size;
    for divisor in (1..=max_divisor).rev() {
        if n % divisor == 0 {
            return divisor;
        }
    }
    1 // If no divisor is found greater than 1, return 1 as the GCD
}

impl<T, U, V> SourceTranslation for FmmDataLinear<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V> + Float + Default + std::marker::Sync + std::marker::Send,
    V: MultiplyAdd<
        V,
        VectorContainer<V>,
        VectorContainer<V>,
        VectorContainer<V>,
        Dynamic,
        Dynamic,
        Dynamic,
    >,
{
    /// Point to multipole evaluations, multithreaded over each leaf box.
    fn p2m<'a>(&self) {
        // Iterate over sibling sets
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let nleaves = leaves.len();
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let surface_size = ncoeffs * self.fmm.kernel.space_dimension();

            let mut check_potentials = rlst_col_vec![V, nleaves * ncoeffs];
            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
            let dim = self.fmm.kernel.space_dimension();

            check_potentials
                .data_mut()
                .par_chunks_exact_mut(ncoeffs)
                .enumerate()
                .zip(self.leaf_upward_surfaces.par_chunks_exact(surface_size))
                .zip(&self.charge_index_pointer)
                .for_each(
                    |(((i, check_potential), upward_check_surface), charge_index_pointer)| {
                        let charges = &self.charges[charge_index_pointer.0..charge_index_pointer.1];
                        let coordinates = &coordinates
                            [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];

                        let nsources = coordinates.len() / dim;

                        let coordinates = unsafe {
                            rlst_pointer_mat!['a, V, coordinates.as_ptr(), (nsources, dim), (dim, 1)]
                        }.eval();

                        if nsources > 0 {
                            self.fmm.kernel.evaluate_st(
                                EvalType::Value,
                                coordinates.data(),
                                upward_check_surface,
                                charges,
                                check_potential,
                            );
                        }
                    },
                );

            // Now compute the multipole expansions, with each of chunk_size boxes at a time.
            let chunk_size = find_chunk_size(nleaves, P2M_MAX_CHUNK_SIZE);

            check_potentials
                .data()
                .par_chunks_exact(ncoeffs*chunk_size)
                .zip(self.leaf_multipoles.par_chunks_exact(chunk_size))
                .zip(self.scales.par_chunks_exact(ncoeffs*chunk_size))
                .for_each(|((check_potential, multipole_ptrs), scale)| {

                    let check_potential = unsafe { rlst_pointer_mat!['a, V, check_potential.as_ptr(), (ncoeffs, chunk_size), (1, ncoeffs)] };
                    let scale = unsafe {rlst_pointer_mat!['a, V, scale.as_ptr(), (ncoeffs, chunk_size), (1, ncoeffs)]}.eval();

                    let tmp = (self.fmm.uc2e_inv_1.dot(&self.fmm.uc2e_inv_2.dot(&check_potential.cmp_wise_product(&scale)))).eval();

                    unsafe {
                        for i in 0..chunk_size {
                            let mut ptr = multipole_ptrs[i].raw;
                            for j in 0..ncoeffs {
                                *ptr += tmp.data()[i*ncoeffs+j];
                                ptr = ptr.add(1);
                            }
                        }
                    }
                })
        }
    }

    /// Multipole to multipole translations, multithreaded over all boxes at a given level.
    fn m2m<'a>(&self, level: u64) {
        if let Some(sources) = self.fmm.tree().get_keys(level) {
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let nsources = sources.len();
            let min = &sources[0];
            let max = &sources[nsources - 1];
            let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
            let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

            let multipoles = &self.multipoles[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

            let nsiblings = 8;
            let mut max_chunk_size = 8_i32.pow((level - 1).try_into().unwrap()) as usize;

            if max_chunk_size > P2M_MAX_CHUNK_SIZE {
                max_chunk_size = P2M_MAX_CHUNK_SIZE;
            }
            let chunk_size = find_chunk_size(nsources, max_chunk_size);

            multipoles
                .par_chunks_exact(nsiblings * ncoeffs*chunk_size)
                .zip(self.level_multipoles[(level - 1) as usize].par_chunks_exact(chunk_size))
                .for_each(|(multipole_chunk, parent)| {

                    unsafe {
                        let tmp = rlst_pointer_mat!['a, V, multipole_chunk.as_ptr(), (ncoeffs*nsiblings, chunk_size), (1, ncoeffs*nsiblings)];
                        let tmp = self.fmm.m2m.dot(&tmp).eval();

                        for i in 0..chunk_size {
                            let mut ptr = parent[i].raw;
                            for j in 0..ncoeffs {
                                *ptr += tmp.data()[(i*ncoeffs)+j];
                                ptr = ptr.add(1)
                            }
                        }
                    }
                })
        }
    }
}

// Try this two different ways, ignoring w,x lists and also including them
impl<T, U, V> TargetTranslation for FmmDataLinear<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V> + Float + Default + std::marker::Sync + std::marker::Send,
    V: MultiplyAdd<
        V,
        VectorContainer<V>,
        VectorContainer<V>,
        VectorContainer<V>,
        Dynamic,
        Dynamic,
        Dynamic,
    >,
{
    fn l2l<'a>(&self, level: u64) {
        if let Some(sources) = self.fmm.tree().get_keys(level) {
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let nsources = sources.len();
            let min = &sources[0];
            let max = &sources[nsources - 1];
            let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
            let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

            let locals = &self.locals[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

            let nsiblings = 8;
            let mut max_chunk_size = 8_i32.pow((level - 1).try_into().unwrap()) as usize;

            if max_chunk_size > P2M_MAX_CHUNK_SIZE {
                max_chunk_size = P2M_MAX_CHUNK_SIZE;
            }
            let chunk_size = find_chunk_size(nsources, max_chunk_size);
            locals
                .par_chunks_exact(nsiblings * ncoeffs*chunk_size)
                .zip(self.level_multipoles[(level + 1) as usize].par_chunks_exact(chunk_size))
                .for_each(|(multipole_chunk, parent)| {

                    unsafe {
                        let tmp = rlst_pointer_mat!['a, V, multipole_chunk.as_ptr(), (ncoeffs*nsiblings, chunk_size), (1, ncoeffs*nsiblings)];
                        let tmp = self.fmm.l2l.dot(&tmp).eval();

                        for i in 0..chunk_size {
                            let mut ptr = parent[i].raw;
                            for j in 0..ncoeffs {
                                *ptr += tmp.data()[(i*ncoeffs)+j];
                                ptr = ptr.add(1)
                            }
                        }
                    }
                })
        }
    }

    fn m2p<'a>(&self) {}

    fn l2p<'a>(&self) {}

    fn p2l<'a>(&self) {}

    fn p2p<'a>(&self) {}
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
    U: Scalar<Real = U>
        + Float
        + Default
        + std::marker::Send
        + std::marker::Sync
        + Fft<FftMatrix<U>, FftMatrix<Complex<U>>>,
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
        // Form signals to use for convolution first
        let n = 2 * self.fmm.order - 1;
        let ntargets = targets.len();
        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        // Pad the signal
        let &(m, n, o) = &(n, n, n);

        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let size = p * q * r;
        let size_real = p * q * (r / 2 + 1);
        let pad_size = (p - m, q - n, r - o);
        let pad_index = (p - m, q - n, r - o);
        // let mut padded_signals = rlst_col_vec![U, size * ntargets];
        let mut padded_signals = vec![U::zero(); size*ntargets];

        let padded_signals_chunks = padded_signals.par_chunks_exact_mut(size);

        let ntargets = targets.len();
        let min = &targets[0];
        let max = &targets[ntargets - 1];
        let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
        let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

        let multipoles = &self.multipoles[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

        let multipoles_chunks = multipoles.par_chunks_exact(ncoeffs);

        padded_signals_chunks
            .zip(multipoles_chunks)
            .for_each(|(padded_signal, multipole)| {
                let signal = self.fmm.m2l.compute_signal(self.fmm.order, multipole);

                let mut tmp = pad3(&signal, pad_size, pad_index);

                padded_signal.copy_from_slice(tmp.get_data());
            });


        // Allocating and handling this vec of structs is really shit
        // let mut padded_signals_hat = rlst_col_vec![Complex<U>, size_real * ntargets];
        let mut padded_signals_hat = vec![Complex::<U>::zero(); size_real*ntargets];
        let mut padded_signals_hat = unsafe {rlst_pointer_mat!['a, Complex<U>, padded_signals_hat.as_mut_ptr(), (size_real*ntargets, 1), (1,1)]};

        // U::rfft3_fftw_par_vec(&mut padded_signals, &mut padded_signals_hat, &[p, q, r]);

        // // let mut real_parts = Vec::with_capacity(padded_signals_hat.data().len());
        // // let mut imag_parts = Vec::with_capacity(padded_signals_hat.data().len());

        // // for complex_val in padded_signals_hat.data().iter() {
        // //     real_parts.push(complex_val.re);
        // //     imag_parts.push(complex_val.im);
        // // }

        // let kernel_data_halo = &self.fmm.m2l.operator_data.kernel_data_rearranged;
        // let ntargets = targets.len();
        // let nparents = ntargets / 8;
        // let mut global_check_potentials_hat = rlst_col_vec![Complex<U>, size_real * ntargets];
        // let mut global_check_potentials = rlst_col_vec![U, size * ntargets];

        // Get check potentials in frequency order
        // let mut global_check_potentials_hat_freq = vec![Vec::new(); size_real];

        // unsafe {
        //     let ptr = global_check_potentials_hat.get_pointer_mut();
        //     for (i, elem) in global_check_potentials_hat_freq
        //         .iter_mut()
        //         .enumerate()
        //         .take(size_real)
        //     {
        //         for j in 0..ntargets {
        //             let raw = ptr.offset((j * size_real + i).try_into().unwrap());
        //             let send_ptr = SendPtrMut { raw };
        //             elem.push(send_ptr);
        //         }
        //     }
        // }

        // // Get signals into frequency order
        // let mut padded_signals_hat_freq = vec![Vec::new(); size_real];
        // let zero = rlst_col_vec![Complex<U>, 8];
        // unsafe {
        //     let ptr = padded_signals_hat.get_pointer();

        //     for (i, elem) in padded_signals_hat_freq
        //         .iter_mut()
        //         .enumerate()
        //         .take(size_real)
        //     {
        //         for j in 0..ntargets {
        //             let raw = ptr.offset((j * size_real + i).try_into().unwrap());
        //             let send_ptr = SendPtr { raw };
        //             elem.push(send_ptr);
        //         }
        //         // put in a bunch of zeros at the end
        //         let ptr = zero.get_pointer();
        //         for _ in 0..8 {
        //             let send_ptr = SendPtr { raw: ptr };
        //             elem.push(send_ptr)
        //         }
        //     }
        // }

        // // Create a map between targets and index positions in vec of len 'ntargets'
        // let mut target_map = HashMap::new();

        // for (i, t) in targets.iter().enumerate() {
        //     target_map.insert(t, i);
        // }
        // // Find all the displacements used for saving results
        // let mut all_displacements = Vec::new();
        // targets.chunks_exact(8).for_each(|sibling_chunk| {
        //     // not in Morton order (refer to sort method when called on 'neighbours')
        //     let parent_neighbours: Vec<Option<MortonKey>> =
        //         sibling_chunk[0].parent().all_neighbors();

        //     let displacements = parent_neighbours
        //         .iter()
        //         .map(|pn| {
        //             let mut tmp = Vec::new();
        //             if let Some(pn) = pn {
        //                 if self.fmm.tree.keys_set.contains(pn) {
        //                     let mut children = pn.children();
        //                     children.sort();
        //                     for child in children {
        //                         // tmp.push(*target_map.get(&child).unwrap() as i64)
        //                         tmp.push(*target_map.get(&child).unwrap())
        //                     }
        //                 } else {
        //                     for i in 0..8 {
        //                         // tmp.push(-1 as i64)
        //                         tmp.push(ntargets + i)
        //                     }
        //                 }
        //             } else {
        //                 for i in 0..8 {
        //                     tmp.push(ntargets + i)
        //                 }
        //             }

        //             assert!(tmp.len() == 8);
        //             tmp
        //         })
        //         .collect_vec();
        //     all_displacements.push(displacements);
        // });

        // // let scale = self.m2l_scale(level);
        // let scale = Complex::from(self.m2l_scale(level));

        // let chunk_size = 64;

        // let mut all_save_locations = Vec::new();
        // // nchunks long
        // let mut all_displacements_chunked = Vec::new();

        // (0..nparents).step_by(chunk_size).for_each(|chunk_start| {
        //     let chunk_end = std::cmp::min(chunk_size+chunk_start, nparents);
            
        //     // lookup save locations
        //     let save_locations = (chunk_start..chunk_end).map(|sibling_idx| {
        //         sibling_idx*8
        //     }).collect_vec();
        //     all_save_locations.push(save_locations);

        //     // 26 long
        //     let mut tmp = Vec::new(); 
        //     for i in 0..26 {
        //         // chunk_size long
        //         let tmp2 = (chunk_start..chunk_end).map(|sibling_idx| {
        //             all_displacements[sibling_idx][i][0]
        //         }).collect_vec();
        //         tmp.push(tmp2);
        //     }
        //     all_displacements_chunked.push(tmp);
        // });

        // (0..size_real).into_par_iter().for_each(|freq| {
        //     // Extract frequency component of signal (ntargets long)
        //     let padded_signal_freq = &padded_signals_hat_freq[freq];

        //     // Extract frequency components of save locations (ntargets long)
        //     let check_potential_freq = &global_check_potentials_hat_freq[freq];

        //     (0..nparents).step_by(chunk_size).enumerate().for_each(|(c, chunk_start)| {
        //         let chunk_end = std::cmp::min(chunk_size+chunk_start, nparents);

        //         let first = all_save_locations[c].first().unwrap();
        //         let last = all_save_locations[c].last().unwrap();
        //         let save_locations = &check_potential_freq[*first..*last+8];

        //         for (i, kernel_data) in kernel_data_halo.iter().enumerate().take(26) {
        //             let frequency_offset = 64 * freq;
        //             let kernel_data_i = &kernel_data[frequency_offset..(frequency_offset + 64)];

        //             // lookup signals
        //             let disps = &all_displacements_chunked[c][i];
        //             let signals = disps.iter().map(|d| &padded_signal_freq[*d..d+8]).collect_vec();
        //             let nsignals = signals.len();

        //             // Loop over all signals and apply Hadamard product for a specific kernel
        //             for k in 0..nsignals {
        //                 // println!("save_locations {:?} {:?}", save_locations.len(), nsignals);
        //                 let save_locations_raw = &save_locations[k*8..(k+1)*8];

        //                 for j in 0..8 {
        //                     let kernel_data_ij = &kernel_data_i[j * 8..(j + 1) * 8];
        //                     let sig = signals[k][j].raw;
        //                     unsafe {
        //                         save_locations_raw
        //                             .iter()
        //                             .zip(kernel_data_ij.iter())
        //                             .for_each(|(&sav, &ker)| *sav.raw += scale * ker * *sig)
        //                     }
        //                 } // inner loop 
        //             }

        //         }

        //     });
        // });


        // // Find all the displacements used for saving results
        // let mut all_displacements = Vec::new();
        // targets.chunks_exact(8).for_each(|sibling_chunk| {
        //     // not in Morton order (refer to sort method when called on 'neighbours')
        //     let parent_neighbours: Vec<Option<MortonKey>> =
        //         sibling_chunk[0].parent().all_neighbors();

        //     let displacements = parent_neighbours
        //         .iter()
        //         .map(|pn| {
        //             let mut tmp = Vec::new();
        //             if let Some(pn) = pn {
        //                 if self.fmm.tree.keys_set.contains(pn) {
        //                     let mut children = pn.children();
        //                     children.sort();
        //                     for child in children {
        //                         // tmp.push(*target_map.get(&child).unwrap() as i64)
        //                         tmp.push(*target_map.get(&child).unwrap())
        //                     }
        //                 } else {
        //                     for i in 0..8 {
        //                         tmp.push(ntargets + i)
        //                     }
        //                 }
        //             } else {
        //                 for i in 0..8 {
        //                     tmp.push(ntargets + i)
        //                 }
        //             }

        //             assert!(tmp.len() == 8);
        //             tmp
        //         })
        //         .collect_vec();
        //     all_displacements.push(displacements);
        // });

        // let scale = Complex::from(self.m2l_scale(level));

        // (0..size_real).into_par_iter().for_each(|freq| {
        //     // Extract frequency component of signal (ntargets long)
        //     let padded_signal_freq = &padded_signals_hat_freq[freq];

        //     // Extract frequency components of save locations (ntargets long)
        //     let check_potential_freq = &global_check_potentials_hat_freq[freq];

        //     (0..nparents).for_each(|sibling_idx| {
        //         // lookup associated save locations for our current sibling set
        //         let save_locations =
        //             &check_potential_freq[(sibling_idx * 8)..(sibling_idx + 1) * 8];
        //         let save_locations_raw = save_locations.iter().map(|s| s.raw).collect_vec();

        //         // for each halo position compute convolutions to a given sibling set
        //         for (i, kernel_data) in kernel_data_halo.iter().enumerate().take(26) {
        //             let frequency_offset = 64 * freq;
        //             let kernel_data_i = &kernel_data[frequency_offset..(frequency_offset + 64)];

        //             // Find displacements for signal being translated
        //             let displacements = &all_displacements[sibling_idx][i];

        //             // Lookup signal to be translated if a translation is to be performed
        //             let signal = &padded_signal_freq[(displacements[0])..=(displacements[7])];
        //             for j in 0..8 {
        //                 let kernel_data_ij = &kernel_data_i[j * 8..(j + 1) * 8];
        //                 let sig = signal[j].raw;
        //                 unsafe {
        //                     save_locations_raw
        //                         .iter()
        //                         .zip(kernel_data_ij.iter())
        //                         .for_each(|(&sav, &ker)| *sav += scale * ker * *sig)
        //                 }
        //             } // inner loop
        //         }
        //     }); // over each sibling set
        // });

        // U::irfft_fftw_par_vec(
        //     &mut global_check_potentials_hat,
        //     &mut global_check_potentials,
        //     &[p, q, r],
        // );

        // // Compute local expansion coefficients and save to data tree
        // let (_, multi_indices) = MortonKey::surface_grid::<U>(self.fmm.order);

        // let check_potentials = global_check_potentials
        //     .data()
        //     .chunks_exact(size)
        //     .flat_map(|chunk| {
        //         let m = 2 * self.fmm.order - 1;
        //         let p = m + 1;
        //         let mut potentials = Array3D::new((p, p, p));
        //         potentials.get_data_mut().copy_from_slice(chunk);

        //         let mut tmp = Vec::new();
        //         let ntargets = multi_indices.len() / 3;
        //         let xs = &multi_indices[0..ntargets];
        //         let ys = &multi_indices[ntargets..2 * ntargets];
        //         let zs = &multi_indices[2 * ntargets..];

        //         for i in 0..ntargets {
        //             let val = potentials.get(zs[i], ys[i], xs[i]).unwrap();
        //             tmp.push(*val);
        //         }
        //         tmp
        //     })
        //     .collect_vec();


        // // This should be blocked and use blas3
        // let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
        // let check_potentials = unsafe {
        //     rlst_pointer_mat!['a, U, check_potentials.as_ptr(), (ncoeffs, ntargets), (1, ncoeffs)]
        // };

        // let mut tmp = self
        //     .fmm
        //     .dc2e_inv_1
        //     .dot(&self.fmm.dc2e_inv_2.dot(&check_potentials))
        //     .eval();

    
        // tmp.data_mut()
        //     .iter_mut()
        //     .for_each(|d| *d *= self.fmm.kernel.scale(level));
        // let locals = tmp;


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
        let Some(targets) = self.fmm.tree().get_keys(level) else {
            return;
        };
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
