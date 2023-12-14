//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.
use bempp_tools::Array3D;

use fftw::{
    plan::{R2CPlan, R2CPlan64},
    types::Flag,
};
use itertools::Itertools;
use num::{Complex, Float, Zero};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    time::Instant, ops::Add,
};

use bempp_field::{
    array::pad3,
    fft::Fft,
    types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    arrays::Array3DAccess,
    field::{FieldTranslation, FieldTranslationData},
    fmm::Fmm,
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::types::{FmmDataLinear, KiFmmLinear, SendPtrMut, SendPtr};
use rlst::{
    algorithms::{linalg::DenseMatrixLinAlgBuilder, traits::svd::Svd},
    common::traits::*,
    dense::{rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer},
};

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

fn displacements_new<U>(tree: &SingleNodeTree<U>, level: u64) -> Vec<Vec<Option<usize>>>
where
    U: Float + Default + Scalar<Real = U>,
{
    let parents = tree.get_keys(level - 1).unwrap();
    let nparents = parents.len();

    let mut target_map = HashMap::new();

    for (i, parent) in parents.iter().enumerate() {
        target_map.insert(parent, i);
    }

    let mut result = vec![Vec::new(); 26];

    let parent_neighbours = parents
        .iter()
        .map(|parent| parent.all_neighbors())
        .collect_vec();

    for i in 0..26 {
        for j in 0..nparents {
            let all_neighbours = &parent_neighbours[j];

            if let Some(neighbour) = all_neighbours[i] {
                result[i].push(Some(*target_map.get(&neighbour).unwrap()))
            } else {
                result[i].push(None);
            }
        }
    }

    result
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

        save_locations
            .iter_mut()
            .zip(kernel_data_ij.iter())
            .for_each(|(sav, &ker)| *sav += scale * ker * sig)
    } // inner loop
}
#[allow(clippy::too_many_arguments)]
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
        // let s = Instant::now();
        // Form signals to use for convolution first
        let n = 2 * self.fmm.order - 1;
        let ntargets = targets.len();
        let nparents = nparents(level as usize);
        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        // Pad the signal
        let &(m, n, o) = &(n, n, n);

        let nsiblings = 8;
        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let size = p * q * r;
        let size_real = p * q * (r / 2 + 1);
        let all_displacements = displacements_new(&self.fmm.tree(), level);

        let ntargets = targets.len();
        let min = &targets[0];
        let max = &targets[ntargets - 1];
        let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
        let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

        let multipoles = &self.multipoles[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

        ////////////////////////////////////////////////////////////////////////////////////
        // Pre processing without using parallel FFT implementation
        // Allocation of Complex vector
        let mut signals_hat_f_buffer = vec![U::zero(); size_real * ntargets * 2];
        let signals_hat_f: &mut [Complex<U>];
        unsafe {
            let ptr = signals_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
            signals_hat_f = std::slice::from_raw_parts_mut(ptr, size_real * ntargets);
        }

        let raw = signals_hat_f.as_mut_ptr();
        let signals_hat_f_ptr = SendPtrMut{raw};

        // Find offsets for each frequency location and store using send pointers

        multipoles
            .par_chunks_exact(ncoeffs * nsiblings)
            .enumerate()
            .for_each(|(i, multipole_chunk)| {
                // Place Signal on convolution grid
                let mut signal_chunk = vec![U::zero(); size * nsiblings];

                for i in 0..nsiblings {
                    let multipole = &multipole_chunk[i * ncoeffs..(i + 1) * ncoeffs];
                    let signal = &mut signal_chunk[i * size..(i + 1) * size];
                    for (surf_idx, &conv_idx) in self.fmm.m2l.surf_to_conv_map.iter().enumerate() {
                        signal[conv_idx] = multipole[surf_idx]
                    }
                }

                // Temporary buffer to hold results of FFT
                let signal_hat_chunk_buffer = vec![U::zero(); size_real * nsiblings * 2];
                let signal_hat_chunk_c;
                unsafe {
                    let ptr = signal_hat_chunk_buffer.as_ptr() as *mut Complex<U>;
                    signal_hat_chunk_c = std::slice::from_raw_parts_mut(ptr, size_real * nsiblings);
                }

                U::rfft3_fftw_slice(&mut signal_chunk, signal_hat_chunk_c, &[p, q, r]);

                // Re-order the temporary buffer into frequency order before flushing to main memory
                let signal_hat_chunk_f_buffer = vec![U::zero(); size_real * nsiblings * 2];
                let signal_hat_chunk_f_c;
                unsafe {
                    let ptr = signal_hat_chunk_f_buffer.as_ptr() as *mut Complex<U>;
                    signal_hat_chunk_f_c = std::slice::from_raw_parts_mut(ptr, size_real * nsiblings);
                }

                for i in 0..size_real {
                    for j in 0..nsiblings {
                        signal_hat_chunk_f_c[nsiblings * i + j] = signal_hat_chunk_c[size_real * j + i]
                    }
                }

                // Storing the results of the FFT in frequency order
                unsafe {
                    let sibling_offset = i * nsiblings;
                    
                    // Pointer to storage buffer for frequency ordered FFT of signals
                    let ptr = signals_hat_f_ptr;
                    
                    for i in 0..size_real {
                        let frequency_offset = i * ntargets;

                        // Head of buffer for each frequency
                        let mut head = ptr.raw.add(frequency_offset).add(sibling_offset);

                        // store results for this frequency for this sibling set
                        let results_i = &signal_hat_chunk_f_c[i*8..(i+1)*8];

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
        // println!("l={:?} Pre processing time {:?}", level, s.elapsed());
        
        // // Test that the signals in frequency order are correct
        // if level == 3 {
        //     println!("signal hat f {:?}", &signals_hat_f[0..ntargets])
        // }
        ////////////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////////////
        let zeros = vec![Complex::<U>::zero(); nsiblings];
        let scale = Complex::from(self.m2l_scale(level));
        // let s = Instant::now();
        let kernel_data_halo = &self.fmm.m2l.operator_data.kernel_data_rearranged;
        (0..size_real)
            .into_par_iter()
            .zip(signals_hat_f.par_chunks_exact(ntargets))
            .zip(check_potentials_hat_f.par_chunks_exact_mut(ntargets))
            .for_each(|((freq, signal_freq), check_potentials_freq)| {

                (0..nparents).for_each(|parent_index| {
                    let save_locations = &mut check_potentials_freq[(parent_index*8)..(parent_index + 1)*8];

                    for (i, kernel_data) in kernel_data_halo.iter().enumerate() {
                        let frequency_offset = 64 * freq;
                        let kernel_data_freq = &kernel_data[frequency_offset..(frequency_offset + 64)];
                        let displacement = &all_displacements[i][parent_index];

                        let signal: &[Complex<U>];
                        if let Some(displacement) = displacement {
                            signal = &signal_freq[displacement*8..(displacement+1)*8];
                        } else {
                            signal = &zeros[..];
                        }
                        unsafe {
                            matmul8x8x2_cplx_simple_local(&kernel_data_freq, signal, save_locations, scale);
                        }
                    }
                    // if freq == 0 && level == 2 {
                    //     println!("check potentials freq 0 {:?}", local_save_locations)
                    // }
                });


            });

        // if level == 2 {
        //     println!("check_potentials_hat_f {:?}", &check_potentials_hat_f[ntargets..2*ntargets]);
        // }
        // println!("l={:?} kernel time {:?}", level, s.elapsed());
        ////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////
        // let s = Instant::now();

        // First step is to get check potentials back into target order from frequency order
        let mut check_potential_hat = vec![U::zero(); size_real * ntargets * 2];
        let mut check_potential = vec![U::zero(); size * ntargets];
        let check_potential_hat_c;
        unsafe { 
            let ptr = check_potential_hat.as_mut_ptr() as *mut Complex<U>;
            check_potential_hat_c = std::slice::from_raw_parts_mut(ptr, size_real)
        }

        check_potential_hat_c.par_chunks_exact_mut(size_real).enumerate().for_each(|(i, check_potential_hat_chunk)| {

            // Lookup all frequencies for this target box
            for j in 0..size_real {
                check_potential_hat_chunk[j] = check_potentials_hat_f[j*ntargets + i]
            };
        });
    
        // Compute FFT
        U::irfft3_fftw_par_slice(check_potential_hat_c, &mut check_potential, &[p, q, r]);

        check_potential.par_chunks_exact(nsiblings*size)
            .zip(self.level_locals[level as usize].par_chunks_exact(nsiblings))
            .for_each(|(check_potential_chunk, local_ptrs)| {

                // Map to surface grid
                let mut tmp = vec![U::zero(); ncoeffs*nsiblings];
                for i in 0..nsiblings {
                    let buffer = &mut tmp[i*ncoeffs..(i+1)*ncoeffs];
                    let check_potential = &check_potential_chunk[i*size..(i+1)*size];
                    for (surf_idx, &conv_idx) in self.fmm.m2l.conv_to_surf_map.iter().enumerate() {
                        buffer[surf_idx] = check_potential[conv_idx];
                    }
                }

                // Can now find local expansion coefficients
                let check_potential_chunk = unsafe {
                    rlst_pointer_mat!['a, U, tmp.as_ptr(), (ncoeffs, nsiblings), (1, ncoeffs)]
                };

                let mut local_chunk = self
                    .fmm
                    .dc2e_inv_1
                    .dot(&self.fmm.dc2e_inv_2.dot(&check_potential_chunk))
                    .eval();
                
                local_chunk.data_mut()
                    .iter_mut()
                    .for_each(|d| *d *= self.fmm.kernel.scale(level));

                local_chunk.data()
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
        
        // println!("l={:?} post processing time {:?}", level, s.elapsed());
        ////////////////////////////////////////////////////////////////////////////////////
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
