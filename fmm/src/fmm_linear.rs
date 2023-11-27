//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.
use bempp_tools::Array3D;
use cauchy::c64;
use fftw::{
    array::AlignedVec,
    plan::{C2RPlan, C2RPlan64, R2CPlan, R2CPlan64},
    types::Flag,
};
use itertools::Itertools;
use num::Zero;
use num::{Complex, Float};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, RwLock},
};

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

use crate::types::{FmmData, FmmDataLinear, KiFmm, KiFmmLinear, SendPtr, SendPtrMut};
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

    fn l2p<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let nleaves = leaves.len();
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let surface_size = ncoeffs * self.fmm.kernel.space_dimension();

        }
    }

    fn p2l<'a>(&self) {}

    fn p2p<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let nleaves = leaves.len();
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let surface_size = ncoeffs * self.fmm.kernel.space_dimension();

            let mut check_potentials = rlst_col_vec![V, nleaves * ncoeffs];
            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
            let dim = self.fmm.kernel.space_dimension();

            let mut target_map = HashMap::new();

            for (i, k) in leaves.iter().enumerate() {
                target_map.insert(k, i);
            }

            
            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
            let potentials = &self.potentials;

            leaves.par_iter().enumerate().zip(&self.charge_index_pointer).for_each(
                |(((i, leaf), charge_index_pointer))| {
                    let targets =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = targets.len() / dim;
                    
                    if ntargets > 0 {
                        let mut local_result = vec![V::zero(); ntargets];
                        let mut result = potentials[i].raw;
                        
                        if let Some(u_list) = self.fmm.get_u_list(leaf) {
                            let u_list_indices = u_list.iter().filter_map(|k| target_map.get(k));


                            let charges = u_list_indices
                                .clone()
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    let charges = &self.charges[index_pointer.0..index_pointer.1];
                                    charges
                                })
                                .collect_vec();

                            let coordinates = u_list_indices
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    let coords =
                                        &coordinates[index_pointer.0 * dim..index_pointer.1 * dim];
                                    coords
                                })
                                .collect_vec();

                            for (&charges, coords) in charges.iter().zip(coordinates) {
                                let nsources = coords.len() / dim;

                                if nsources > 0 {
                                    self.fmm.kernel.evaluate_st(
                                        EvalType::Value,
                                        coords,
                                        targets,
                                        charges,
                                        &mut local_result,
                                    )
                                }
                            }
                            // Save to global locations
                            for res in local_result.iter() {

                                unsafe {
                                    *result += *res;
                                    result = result.add(1);
                                }

                            } 
                        }

                    }
                },
            )
        }
    }
}

pub fn ncoeffs(order: usize) -> usize {
    6 * (order - 1).pow(2) + 2
}

pub fn size_real(order: usize) -> usize {
    let m = 2 * order - 1; // Size of each dimension of 3D kernel/signal
    let pad_size = 1;
    let p = m + pad_size; // Size of each dimension of padded 3D kernel/signal
    p * p * (p / 2 + 1) // Number of Fourier coefficients when working with real data
}

pub fn nleaves(level: usize) -> usize {
    8i32.pow(level as u32) as usize
}

pub fn nparents(level: usize) -> usize {
    8i32.pow((level - 1) as u32) as usize
}

pub fn signal_freq_order_cplx_optimized<U>(
    order: usize,
    level: usize,
    signal: &[Complex<U>],
) -> Vec<Vec<Complex<U>>>
where
    U: Scalar,
{
    let size_real = size_real(order);
    let nleaves = nleaves(level);

    (0..size_real)
        .map(|i| {
            let mut tmp = (0..nleaves)
                .map(|j| signal[j * size_real + i])
                .collect::<Vec<_>>();

            // Pad with zeros
            tmp.extend(std::iter::repeat(Complex::new(U::zero(), U::zero())).take(8));

            tmp
        })
        .collect()
}

pub unsafe fn check_potentials_freq_cplx<U>(
    order: usize,
    level: usize,
    check_potentials: &mut Vec<Complex<U>>,
) -> Vec<Vec<SendPtrMut<Complex<U>>>>
where
    U: Scalar,
{
    let size_real = size_real(order); // Number of Fourier coefficients when working with real data
    let nleaves = nleaves(level);
    let mut check_potentials_freq = vec![Vec::new(); size_real];

    let ptr = check_potentials.as_mut_ptr();

    for (i, elem) in check_potentials_freq.iter_mut().enumerate().take(size_real) {
        for j in 0..nleaves {
            let raw = ptr.offset((j * size_real + i).try_into().unwrap());
            let send_ptr = SendPtrMut { raw };
            elem.push(send_ptr);
        }
    }

    check_potentials_freq
}

pub fn displacements<U>(
    tree: &SingleNodeTree<U>,
    level: u64,
    target_map_leaves: &HashMap<&MortonKey, usize>,
) -> Vec<Vec<Vec<usize>>>
where
    U: Float + Default + Scalar<Real = U>,
{
    let leaves = tree.get_keys(level).unwrap();
    let nleaves = leaves.len();

    let mut all_displacements = Vec::new();

    leaves.chunks_exact(8).for_each(|sibling_chunk| {
        let parent_neighbours = sibling_chunk[0].parent().all_neighbors();
        let displacements = parent_neighbours
            .iter()
            .map(|pn| {
                let mut tmp = Vec::new();

                if let Some(pn) = pn {
                    if tree.keys_set.contains(pn) {
                        let mut children = pn.children();
                        children.sort();
                        for child in children.iter() {
                            tmp.push(*target_map_leaves.get(child).unwrap())
                        }
                    } else {
                        for i in 0..8 {
                            tmp.push(nleaves + i);
                        }
                    }
                } else {
                    for i in 0..8 {
                        tmp.push(nleaves + i)
                    }
                }
                tmp
            })
            .collect_vec();
        all_displacements.push(displacements)
    });

    all_displacements
}

pub fn chunked_displacements(
    level: usize,
    chunksize: usize,
    displacements: &[Vec<Vec<usize>>],
) -> (Vec<Vec<Vec<usize>>>, Vec<Vec<usize>>) {
    let mut all_save_locations = Vec::new();
    let mut all_displacements_chunked = Vec::new(); // indexed by chunk index
    let nparents = nparents(level);
    let nsiblings = 8;

    (0..nparents).step_by(chunksize).for_each(|chunk_start| {
        let chunk_end = std::cmp::min(chunksize + chunk_start, nparents);

        // lookup save locations (head)
        let save_locations = (chunk_start..chunk_end)
            .map(|sibling_idx| sibling_idx * nsiblings)
            .collect_vec();
        all_save_locations.push(save_locations);

        // 26 long
        let mut tmp = Vec::new();
        for i in 0..26 {
            // chunk_size long
            let tmp2 = (chunk_start..chunk_end)
                .map(|sibling_idx| {
                    // head
                    displacements[sibling_idx][i][0]
                })
                .collect_vec();

            tmp.push(tmp2);
        }
        all_displacements_chunked.push(tmp);
    });

    (all_displacements_chunked, all_save_locations)
}

#[inline(always)]
pub unsafe fn matmul8x8x2_cplx_simple_local<U>(
    kernel_data_freq: &[Complex<U>],
    signal: &[Complex<U>],
    save_locations: &mut [Complex<U>],
    scale: Complex<U>,
) where
    U: Scalar,
{
    for j in 0..8 {
        let kernel_data_ij = &kernel_data_freq[j * 8..(j + 1) * 8];
        let sig = signal[j];

        // save_locations
        //     .iter_mut()
        //     .zip(kernel_data_ij.iter())
        //     .for_each(|(sav, &ker)| *sav += scale * ker * sig)
    } // inner loop
}

#[inline(always)]
pub fn m2l_cplx_chunked<U>(
    order: usize,
    level: usize,
    signal_freq_order: &[Vec<Complex<U>>],
    check_potential_freq_order: &[Vec<SendPtrMut<Complex<U>>>],
    kernel_data_halo: &[Vec<Complex<U>>],
    all_displacements_chunked: &[Vec<Vec<usize>>],
    all_save_locations_chunked: &[Vec<usize>],
    chunksize: usize,
    scale: Complex<U>,
) where
    U: Scalar + Sync,
{
    let nparents = nparents(level);
    let size_real = size_real(order);
    // let scale = m2l_scale(level as u64);

    (0..size_real).into_par_iter().for_each(|freq| {
        // Extract frequency component of signal (ntargets long)
        let padded_signal_freq = &signal_freq_order[freq];

        // Extract frequency components of save locations (ntargets long)
        let check_potential_freq = &check_potential_freq_order[freq];

        let zero = U::from(0.).unwrap();

        (0..nparents)
            .step_by(chunksize)
            .enumerate()
            .for_each(|(c, _chunk_start)| {
                let first = all_save_locations_chunked[c].first().unwrap();
                let last = all_save_locations_chunked[c].last().unwrap();
                let save_locations = &check_potential_freq[*first..*last + 8];
                let save_locations_raw = save_locations.iter().map(|s| s.raw).collect_vec();
                let mut local_save_locations = vec![Complex::new(zero, zero); 8 * chunksize];

                for (i, kernel_data) in kernel_data_halo.iter().enumerate().take(26) {
                    let frequency_offset = 64 * freq;
                    let kernel_data_freq =
                        &kernel_data[frequency_offset..(frequency_offset + 64)].to_vec();

                    // lookup signals
                    let disps = &all_displacements_chunked[c][i];

                    // Expect this to be 8*chunksize
                    let signals = disps
                        .iter()
                        .map(|d| &padded_signal_freq[*d..d + 8])
                        .collect_vec();

                    for j in 0..chunksize {
                        unsafe {
                            matmul8x8x2_cplx_simple_local(
                                kernel_data_freq,
                                signals[j],
                                &mut local_save_locations[j * 8..(j + 1) * 8],
                                scale,
                            )
                        };
                    }
                }

                unsafe {
                    save_locations_raw
                        .iter()
                        .zip(local_save_locations.iter())
                        .for_each(|(&glob, &loc)| {
                            *glob += loc;
                        });
                }
            });
    });
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
        let mut padded_signals = vec![U::default(); size * ntargets];

        // let padded_signals_chunks = padded_signals.data_mut().par_chunks_exact_mut(size);
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

                let tmp = pad3(&signal, pad_size, pad_index);

                padded_signal.copy_from_slice(tmp.get_data());
            });

        // Allocating and handling this vec of structs is really shit
        // let mut padded_signals_hat = rlst_col_vec![Complex<U>, size_real * ntargets];
        let mut padded_signals_hat = vec![Complex::<U>::default(); size_real * ntargets];

        U::rfft3_fftw_par_vec(&mut padded_signals, &mut padded_signals_hat, &[p, q, r]);

        let ntargets = targets.len();
        let mut global_check_potentials_hat = vec![Complex::<U>::default(); size_real * ntargets];
        let mut global_check_potentials = vec![U::default(); size * ntargets];

        // Get check potentials in frequency order
        let global_check_potentials_hat_freq = unsafe {
            check_potentials_freq_cplx(
                self.fmm.order,
                level as usize,
                &mut global_check_potentials_hat,
            )
        };
     
        let padded_signals_hat_freq = signal_freq_order_cplx_optimized(
            self.fmm.order,
            level as usize,
            &padded_signals_hat[..],
        );

        // Create a map between targets and index positions in vec of len 'ntargets'
        let mut target_map = HashMap::new();

        for (i, t) in targets.iter().enumerate() {
            target_map.insert(t, i);
        }

        let chunksize;
        if level == 2 {
            chunksize = 8;
        } else if level == 3 {
            chunksize = 64
        } else {
            chunksize = 128
        }

        let all_displacements = displacements(&self.fmm.tree, level, &target_map);

        let (chunked_displacements, chunked_save_locations) =
            chunked_displacements(level as usize, chunksize, &all_displacements);

        // // let scale = self.m2l_scale(level);
        let scale = Complex::from(self.m2l_scale(level));


        let kernel_data_halo = &self.fmm.m2l.operator_data.kernel_data_rearranged;

        m2l_cplx_chunked(
            self.fmm.order,
            level as usize,
            &padded_signals_hat_freq,
            &global_check_potentials_hat_freq,
            &kernel_data_halo,
            &chunked_displacements,
            &chunked_save_locations,
            chunksize,
            scale,
        );

        U::irfft_fftw_par_vec(
            &mut global_check_potentials_hat,
            &mut global_check_potentials,
            &[p, q, r],
        );

        // Compute local expansion coefficients and save to data tree
        let (_, multi_indices) = MortonKey::surface_grid::<U>(self.fmm.order);

        let check_potentials = global_check_potentials
            .chunks_exact(size)
            .flat_map(|chunk| {
                let m = 2 * self.fmm.order - 1;
                let p = m + 1;
                let mut potentials = Array3D::new((p, p, p));
                potentials.get_data_mut().copy_from_slice(chunk);

                let mut tmp = Vec::new();
                let ntargets = multi_indices.len() / 3;
                let xs = &multi_indices[0..ntargets];
                let ys = &multi_indices[ntargets..2 * ntargets];
                let zs = &multi_indices[2 * ntargets..];

                for i in 0..ntargets {
                    let val = potentials.get(zs[i], ys[i], xs[i]).unwrap();
                    tmp.push(*val);
                }
                tmp
            })
            .collect_vec();

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
