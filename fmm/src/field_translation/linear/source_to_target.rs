//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.
use bempp_tools::Array3D;

use itertools::Itertools;
use num::{Complex, Float, Zero};
use rayon::prelude::*;
use std::{collections::{HashMap, HashSet}, time::Instant};

use bempp_field::{
    array::pad3,
    fft::Fft,
    types::{FftFieldTranslationKiFmm, FftMatrix, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    arrays::Array3DAccess,
    field::{FieldTranslation, FieldTranslationData},
    fmm::Fmm,
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::types::{FmmDataLinear, KiFmmLinear, SendPtrMut};
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
        let s = Instant::now();
        // Form signals to use for convolution first
        let n = 2 * self.fmm.order - 1;
        let ntargets = targets.len();
        let nparents = nparents(level as usize);
        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        // Pad the signal
        let &(m, n, o) = &(n, n, n);

        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let size = p * q * r;
        let size_real = p * q * (r / 2 + 1);
        let mut signals = vec![U::default(); size * ntargets];

        let signals_chunks = signals.par_chunks_exact_mut(size);

        let ntargets = targets.len();
        let min = &targets[0];
        let max = &targets[ntargets - 1];
        let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
        let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

        let multipoles = &self.multipoles[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];
        let multipoles_chunks = multipoles.par_chunks_exact(ncoeffs);

        signals_chunks
            .zip(multipoles_chunks)
            .for_each(|(signal, multipole)| {
                for (i, &j) in self.fmm.m2l.surf_to_conv_map.iter().enumerate() {
                    signal[j] = multipole[i]
                }
            });

        let mut signals_hat = vec![Complex::<U>::default(); size_real * ntargets];

        U::rfft3_fftw_par_vec(&mut signals, &mut signals_hat, &[p, q, r]);

        // Get signal FFTs into frequency order
        signals_hat.par_chunks_exact_mut(8*size_real).for_each(|sibling_chunk| {

            for i in 0..size_real {
                for j in 0..8 {
                    sibling_chunk[8 * i + j] = sibling_chunk[size_real * j + i]
                }
            }
        });
        
        // Allocate check potentials (in frequency order at this point implicitly)
        let mut check_potentials_hat = vec![Complex::<U>::default(); size_real * ntargets];
        
        println!("pre processing time {:?}", s.elapsed());

        let s = Instant::now();
        let kernel_data_halo = &self.fmm.m2l.operator_data.kernel_data_rearranged;
        (0..size_real)
            .into_par_iter()
            .zip(signals_hat.par_chunks_exact(ntargets))
            .zip(check_potentials_hat.par_chunks_exact_mut(ntargets))
            .for_each(|((freq, signal_freq), check_potentials_freq)| {

                (0..nparents).for_each(|sibling_index| {
                    let save_locations = &mut check_potentials_freq[(sibling_index*8)..(sibling_index + 1)*8];

                    for (i, kernel_data) in kernel_data_halo.iter().enumerate() {
                        let frequency_offset = 64 * freq;
                        let kernel_data_freq = &kernel_data[frequency_offset..(frequency_offset + 64)];
                        

                    }
                })

            });

        println!("kernel time {:?}", s.elapsed());

        // let ntargets = targets.len();
        // let mut global_check_potentials_hat = vec![Complex::<U>::default(); size_real * ntargets];
        // let mut global_check_potentials = vec![U::default(); size * ntargets];

        // // Get check potentials in frequency order
        // let global_check_potentials_hat_freq = unsafe {
        //     check_potentials_freq_cplx(
        //         self.fmm.order,
        //         level as usize,
        //         &mut global_check_potentials_hat,
        //     )
        // };

        // let padded_signals_hat_freq = signal_freq_order_cplx_optimized(
        //     self.fmm.order,
        //     level as usize,
        //     &padded_signals_hat[..],
        // );

        // // Create a map between targets and index positions in vec of len 'ntargets'
        // let mut target_map = HashMap::new();

        // for (i, t) in targets.iter().enumerate() {
        //     target_map.insert(t, i);
        // }

        // let chunksize;
        // if level == 2 {
        //     chunksize = 8;
        // } else if level == 3 {
        //     chunksize = 64
        // } else {
        //     chunksize = 128
        // }

        // let all_displacements = displacements(&self.fmm.tree, level, &target_map);

        // let (chunked_displacements, chunked_save_locations) =
        //     chunked_displacements(level as usize, chunksize, &all_displacements);

        // let scale = Complex::from(self.m2l_scale(level));

        // let kernel_data_halo = &self.fmm.m2l.operator_data.kernel_data_rearranged;

        // m2l_cplx_chunked(
        //     self.fmm.order,
        //     level as usize,
        //     &padded_signals_hat_freq,
        //     &global_check_potentials_hat_freq,
        //     kernel_data_halo,
        //     &chunked_displacements,
        //     &chunked_save_locations,
        //     chunksize,
        //     scale,
        // );

        // U::irfft_fftw_par_vec(
        //     &mut global_check_potentials_hat,
        //     &mut global_check_potentials,
        //     &[p, q, r],
        // );

        // // Compute local expansion coefficients and save to data tree
        // let (_, multi_indices) = MortonKey::surface_grid::<U>(self.fmm.order);

        // let check_potentials = global_check_potentials
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

        // let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        // // Add result
        // tmp.data()
        //     .par_chunks_exact(ncoeffs)
        //     .zip(self.level_locals[level as usize].into_par_iter())
        //     .for_each(|(result, local)| unsafe {
        //         let mut ptr = local.raw;
        //         for &r in result.iter().take(ncoeffs) {
        //             *ptr += r;
        //             ptr = ptr.add(1)
        //         }
        //     });
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