//! Implementation of field translations for each FMM.
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut, Mul},
    sync::{Arc, Mutex, MutexGuard, RwLock},
    time::Instant, cell::RefCell,
};

use bempp_tools::Array3D;
use fftw::types::*;
use itertools::Itertools;
use num::Zero;
use num::{Complex, FromPrimitive};
use rayon::prelude::*;

use bempp_field::{
    array::pad3,
    fft::{irfft3_fftw, irfft3_fftw_par_vec, rfft3_fftw, rfft3_fftw_par_vec},
    hadamard::hadamard_product_sibling,
    types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    arrays::Array3DAccess,
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, InteractionLists, SourceTranslation, TargetTranslation},
    kernel::{Kernel, KernelScale, self},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};
use rlst::{
    common::tools::PrettyPrint,
    common::traits::*,
    dense::{
        global, rlst_col_vec, rlst_mat, rlst_pointer_mat, rlst_rand_col_vec, traits::*, Dot, Shape,
    },
};

use crate::{
    constants::CACHE_SIZE,
    types::{FmmData, KiFmm},
};


impl<T, U> SourceTranslation for FmmData<KiFmm<SingleNodeTree, T, U>>
where
    T: Kernel<T = f64> + KernelScale<T = f64> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
{
    /// Point to multipole evaluations, multithreaded over each leaf box.
    fn p2m<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_leaves() {
            leaves.par_iter().for_each(move |&leaf| {
                let leaf_multipole_arc = Arc::clone(self.multipoles.get(&leaf).unwrap());
                let fmm_arc = Arc::clone(&self.fmm);

                if let Some(leaf_points) = self.points.get(&leaf) {
                    let leaf_charges_arc = Arc::clone(self.charges.get(&leaf).unwrap());

                    // Lookup data
                    let leaf_coordinates = leaf_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    let global_idxs = leaf_points
                        .iter()
                        .map(|p| p.global_idx)
                        .collect_vec();

                    let nsources = leaf_coordinates.len() / self.fmm.kernel.space_dimension();

                    // Get into row major order
                    let leaf_coordinates = unsafe {
                        rlst_pointer_mat!['a, f64, leaf_coordinates.as_ptr(), (nsources, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                    }.eval();

                    let upward_check_surface = leaf.compute_surface(
                        &fmm_arc.tree().domain,
                        fmm_arc.order,
                        fmm_arc.alpha_outer,
                    );
                    let ntargets = upward_check_surface.len() / fmm_arc.kernel.space_dimension();

                    let leaf_charges = leaf_charges_arc.deref();

                    // Calculate check potential
                    let mut check_potential = rlst_col_vec![f64, ntargets];

                    fmm_arc.kernel.evaluate_st(
                        EvalType::Value,
                        leaf_coordinates.data(),
                        &upward_check_surface[..],
                        &leaf_charges[..],
                        check_potential.data_mut(),
                    );

                    let leaf_multipole_owned = (
                        fmm_arc.kernel.scale(leaf.level())
                        * fmm_arc.uc2e_inv.dot(&check_potential)
                    ).eval();

                    let mut leaf_multipole_lock = leaf_multipole_arc.lock().unwrap();

                    *leaf_multipole_lock.deref_mut() = (leaf_multipole_lock.deref() + leaf_multipole_owned).eval();
                }
            });
        }
    }

    /// Multipole to multipole translations, multithreaded over all boxes at a given level.
    fn m2m<'a>(&self, level: u64) {
        // Parallelise over nodes at a given level
        if let Some(sources) = self.fmm.tree().get_keys(level) {
            sources.par_iter().for_each(move |&source| {
                let operator_index = source.siblings().iter().position(|&x| x == source).unwrap();
                let source_multipole_arc = Arc::clone(self.multipoles.get(&source).unwrap());
                let target_multipole_arc =
                    Arc::clone(self.multipoles.get(&source.parent()).unwrap());
                let fmm_arc = Arc::clone(&self.fmm);

                let source_multipole_lock = source_multipole_arc.lock().unwrap();

                let target_multipole_owned =
                    fmm_arc.m2m[operator_index].dot(&source_multipole_lock);

                let mut target_multipole_lock = target_multipole_arc.lock().unwrap();

                *target_multipole_lock.deref_mut() =
                    (target_multipole_lock.deref() + target_multipole_owned).eval();
            })
        }
    }
}

impl<T, U> TargetTranslation for FmmData<KiFmm<SingleNodeTree, T, U>>
where
    T: Kernel<T = f64> + KernelScale<T = f64> + std::marker::Sync + std::marker::Send,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
{
    fn l2l(&self, level: u64) {
        if let Some(targets) = self.fmm.tree().get_keys(level) {
            targets.par_iter().for_each(move |&target| {
                let source_local_arc = Arc::clone(self.locals.get(&target.parent()).unwrap());
                let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());
                let fmm = Arc::clone(&self.fmm);

                let operator_index = target.siblings().iter().position(|&x| x == target).unwrap();

                let source_local_lock = source_local_arc.lock().unwrap();

                let target_local_owned = fmm.l2l[operator_index].dot(&source_local_lock);
                let mut target_local_lock = target_local_arc.lock().unwrap();

                *target_local_lock.deref_mut() =
                    (target_local_lock.deref() + target_local_owned).eval();
            })
        }
    }

    fn m2p<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&target| {

                let fmm_arc = Arc::clone(&self.fmm);

                if let Some(points) = fmm_arc.tree().get_points(&target) {
                    let target_potential_arc = Arc::clone(self.potentials.get(&target).unwrap());
                    if let Some(w_list) = fmm_arc.get_w_list(&target) {
                        for source in w_list.iter() {
                            let source_multipole_arc =
                                Arc::clone(self.multipoles.get(source).unwrap());

                            let upward_equivalent_surface = source.compute_surface(
                                fmm_arc.tree().get_domain(),
                                fmm_arc.order(),
                                fmm_arc.alpha_inner,
                            );

                            let source_multipole_lock = source_multipole_arc.lock().unwrap();

                            let target_coordinates = points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let ntargets = target_coordinates.len() / self.fmm.kernel.space_dimension();

                            // Get into row major order
                            let target_coordinates = unsafe {
                                rlst_pointer_mat!['a, f64, target_coordinates.as_ptr(), (ntargets, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                            }.eval();

                            let mut target_potential = rlst_col_vec![f64, ntargets];

                            fmm_arc.kernel.evaluate_st(
                                EvalType::Value,
                                &upward_equivalent_surface[..],
                                target_coordinates.data(),
                                source_multipole_lock.data(),
                                target_potential.data_mut(),
                            );

                            let mut target_potential_lock = target_potential_arc.lock().unwrap();

                            *target_potential_lock.deref_mut() = (target_potential_lock.deref() + target_potential).eval();
                        }
                    }
                }
            }
)
        }
    }

    fn l2p<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let fmm_arc = Arc::clone(&self.fmm);
                let source_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());

                if let Some(target_points) = fmm_arc.tree().get_points(&leaf) {
                    let target_potential_arc = Arc::clone(self.potentials.get(&leaf).unwrap());
                    // Lookup data
                    let target_coordinates = target_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();
                    let ntargets = target_coordinates.len() / self.fmm.kernel.space_dimension();

                    // Get into row major order
                    let target_coordinates = unsafe {
                        rlst_pointer_mat!['a, f64, target_coordinates.as_ptr(), (ntargets, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                    }.eval();

                    let downward_equivalent_surface = leaf.compute_surface(
                        &fmm_arc.tree().domain,
                        fmm_arc.order,
                        fmm_arc.alpha_outer,
                    );

                    let source_local_lock = source_local_arc.lock().unwrap();

                    let mut target_potential = rlst_col_vec![f64, ntargets];

                    fmm_arc.kernel.evaluate_st(
                        EvalType::Value,
                        &downward_equivalent_surface[..],
                        target_coordinates.data(),
                        source_local_lock.data(),
                        target_potential.data_mut(),
                    );

                    let mut target_potential_lock = target_potential_arc.lock().unwrap();

                    *target_potential_lock.deref_mut() = (target_potential_lock.deref() + target_potential).eval();
                }
            })
        }
    }

    fn p2l<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());

                if let Some(x_list) = fmm_arc.get_x_list(&leaf) {
                    for source in x_list.iter() {
                        if let Some(source_points) = fmm_arc.tree().get_points(source) {
                            let source_coordinates = source_points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let nsources = source_coordinates.len() / self.fmm.kernel.space_dimension();

                            // Get into row major order
                            let source_coordinates = unsafe {
                                rlst_pointer_mat!['a, f64, source_coordinates.as_ptr(), (nsources, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                            }.eval();

                            let source_charges = self.charges.get(source).unwrap();

                            let downward_check_surface = leaf.compute_surface(
                                &fmm_arc.tree().domain,
                                fmm_arc.order,
                                fmm_arc.alpha_inner,
                            );

                            let ntargets = downward_check_surface.len() / fmm_arc.kernel.space_dimension();
                            let mut downward_check_potential = rlst_col_vec![f64, ntargets];

                            fmm_arc.kernel.evaluate_st(
                                EvalType::Value,
                                source_coordinates.data(),
                                &downward_check_surface[..],
                                &source_charges[..],
                                downward_check_potential.data_mut()
                            );


                            let mut target_local_lock = target_local_arc.lock().unwrap();

                            let target_local_owned = (fmm_arc.kernel.scale(leaf.level()) * fmm_arc.dc2e_inv.dot(&downward_check_potential)).eval();

                            *target_local_lock.deref_mut() = (target_local_lock.deref() + target_local_owned).eval();
                        }
                    }
                }
            })
        }
    }

    fn p2p<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&target| {
                let fmm_arc = Arc::clone(&self.fmm);

                if let Some(target_points) = fmm_arc.tree().get_points(&target) {
                    let target_potential_arc = Arc::clone(self.potentials.get(&target).unwrap());
                    let target_coordinates = target_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    let ntargets= target_coordinates.len() / self.fmm.kernel.space_dimension();

                    // Get into row major order
                    let target_coordinates = unsafe {
                        rlst_pointer_mat!['a, f64, target_coordinates.as_ptr(), (ntargets, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                    }.eval();

                    if let Some(u_list) = fmm_arc.get_u_list(&target) {
                        for source in u_list.iter() {
                            if let Some(source_points) = fmm_arc.tree().get_points(source) {
                                let source_coordinates = source_points
                                    .iter()
                                    .map(|p| p.coordinate)
                                    .flat_map(|[x, y, z]| vec![x, y, z])
                                    .collect_vec();

                                let nsources = source_coordinates.len() / self.fmm.kernel.space_dimension();

                                // Get into row major order
                                let source_coordinates = unsafe {
                                    rlst_pointer_mat!['a, f64, source_coordinates.as_ptr(), (nsources, fmm_arc.kernel.space_dimension()), (fmm_arc.kernel.space_dimension(), 1)]
                                }.eval();

                                let source_charges_arc =
                                    Arc::clone(self.charges.get(source).unwrap());

                                let mut target_potential = rlst_col_vec![f64, ntargets];

                                fmm_arc.kernel.evaluate_st(
                                    EvalType::Value,
                                    source_coordinates.data(),
                                    target_coordinates.data(),
                                    &source_charges_arc[..],
                                    target_potential.data_mut(),
                                );

                                let mut target_potential_lock =
                                    target_potential_arc.lock().unwrap();

                                *target_potential_lock.deref_mut() = (target_potential_lock.deref() + target_potential).eval();
                            }
                        }
                    }
                }
            })
        }
    }
}

/// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
impl<T> FieldTranslation for FmmData<KiFmm<SingleNodeTree, T, SvdFieldTranslationKiFmm<T>>>
where
    T: Kernel<T = f64> + KernelScale<T = f64> + std::marker::Sync + std::marker::Send + Default,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else { return };
        let mut transfer_vector_to_m2l =
            HashMap::<usize, Arc<Mutex<Vec<(MortonKey, MortonKey)>>>>::new();

        for tv in self.fmm.m2l.transfer_vectors.iter() {
            transfer_vector_to_m2l.insert(tv.hash, Arc::new(Mutex::new(Vec::new())));
        }

        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        targets.par_iter().enumerate().for_each(|(_i, &target)| {
            if let Some(v_list) = self.fmm.get_v_list(&target) {
                let calculated_transfer_vectors = v_list
                    .iter()
                    .map(|source| target.find_transfer_vector(source))
                    .collect::<Vec<usize>>();
                for (transfer_vector, &source) in
                    calculated_transfer_vectors.iter().zip(v_list.iter())
                {
                    let m2l_arc = Arc::clone(transfer_vector_to_m2l.get(transfer_vector).unwrap());
                    let mut m2l_lock = m2l_arc.lock().unwrap();
                    m2l_lock.push((source, target));
                }
            }
        });

        let mut transfer_vector_to_m2l_rw_lock =
            HashMap::<usize, Arc<RwLock<Vec<(MortonKey, MortonKey)>>>>::new();

        // Find all multipole expansions and allocate
        for (&transfer_vector, m2l_arc) in transfer_vector_to_m2l.iter() {
            transfer_vector_to_m2l_rw_lock.insert(
                transfer_vector,
                Arc::new(RwLock::new(m2l_arc.lock().unwrap().clone())),
            );
        }

        transfer_vector_to_m2l_rw_lock
            .par_iter()
            .for_each(|(transfer_vector, m2l_arc)| {
                let c_idx = self
                    .fmm
                    .m2l
                    .transfer_vectors
                    .iter()
                    .position(|x| x.hash == *transfer_vector)
                    .unwrap();

                let (nrows, _) = self.fmm.m2l.operator_data.c.shape();
                let top_left = (0, c_idx * self.fmm.m2l.k);
                let dim = (nrows, self.fmm.m2l.k);

                let c_sub = self.fmm.m2l.operator_data.c.block(top_left, dim);

                let m2l_rw = m2l_arc.read().unwrap();
                let mut multipoles = rlst_mat![f64, (self.fmm.m2l.k, m2l_rw.len())];

                for (i, (source, _)) in m2l_rw.iter().enumerate() {
                    let source_multipole_arc = Arc::clone(self.multipoles.get(source).unwrap());
                    let source_multipole_lock = source_multipole_arc.lock().unwrap();

                    // Compressed multipole
                    let compressed_source_multipole_owned = self
                        .fmm
                        .m2l
                        .operator_data
                        .st_block
                        .dot(&source_multipole_lock)
                        .eval();

                    let first = i * self.fmm.m2l.k;
                    let last = first + self.fmm.m2l.k;

                    let multipole_slice = multipoles.get_slice_mut(first, last);
                    multipole_slice.copy_from_slice(compressed_source_multipole_owned.data());
                }

                // Compute convolution
                let compressed_check_potential_owned = c_sub.dot(&multipoles);

                // Post process to find check potential
                let check_potential_owned = self
                    .fmm
                    .m2l
                    .operator_data
                    .u
                    .dot(&compressed_check_potential_owned)
                    .eval();

                // Compute local
                let locals_owned = (self.fmm.dc2e_inv.dot(&check_potential_owned)
                    * self.fmm.kernel.scale(level)
                    * self.m2l_scale(level))
                .eval();

                // Assign locals
                for (i, (_, target)) in m2l_rw.iter().enumerate() {
                    let target_local_arc = Arc::clone(self.locals.get(target).unwrap());
                    let mut target_local_lock = target_local_arc.lock().unwrap();

                    let top_left = (0, i);
                    let dim = (ncoeffs, 1);
                    let target_local_owned = locals_owned.block(top_left, dim);

                    *target_local_lock.deref_mut() =
                        (target_local_lock.deref() + target_local_owned).eval();
                }
            });
    }

    fn m2l_scale(&self, level: u64) -> f64 {
        if level < 2 {
            panic!("M2L only performed on level 2 and below")
        }

        if level == 2 {
            1. / 2.
        } else {
            2_f64.powf((level - 3) as f64)
        }
    }
}

/// Implement the multipole to local translation operator for an FFT accelerated KiFMM on a single node.
impl<T> FieldTranslation for FmmData<KiFmm<SingleNodeTree, T, FftFieldTranslationKiFmm<T>>>
where
    T: Kernel<T = f64> + KernelScale<T = f64> + std::marker::Sync + std::marker::Send + Default,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else { return };

        // Form signals to use for convolution first
        let start = Instant::now();

        let n = 2 * self.fmm.order - 1;
        let ntargets = targets.len();

        // Pad the signal
        let &(m, n, o) = &(n, n, n);

        let p = m + 1;
        let q = n + 1;
        let r = o + 1;
        let size = p * q * r;
        let size_real = p * q * (r / 2 + 1);
        let pad_size = (p - m, q - n, r - o);
        let pad_index = (p - m, q - n, r - o);
        let real_dim = q;
        let mut padded_signals = rlst_col_vec![f64, size * ntargets];

        let mut chunks = padded_signals.data_mut().par_chunks_exact_mut(size);

        let range = (0..chunks.len()).into_par_iter();
        range.zip(chunks).for_each(|(i, chunk)| {
            let fmm_arc = Arc::clone(&self.fmm);
            let target = targets[i];
            let source_multipole_arc = Arc::clone(self.multipoles.get(&target).unwrap());
            let source_multipole_lock = source_multipole_arc.lock().unwrap();
            let signal = fmm_arc
                .m2l
                .compute_signal(fmm_arc.order, source_multipole_lock.data());

            let mut padded_signal = pad3(&signal, pad_size, pad_index);

            chunk.copy_from_slice(padded_signal.get_data());
        });

        println!("data organisation time {:?}", start.elapsed().as_millis());

        let mut padded_signals_hat = rlst_col_vec![c64, size_real * ntargets];
        let start = Instant::now();
        rfft3_fftw_par_vec(&mut padded_signals, &mut padded_signals_hat, &[p, q, r]);
        println!("fft time {:?}", start.elapsed().as_millis());

        // Compute Hadamard product
        let start = Instant::now();
        // let mut global_check_potentials_hat = HashMap::new();
        // for target in targets.iter() {
        //     global_check_potentials_hat.insert(
        //         *target,
        //         Arc::new(Mutex::new(vec![Complex::<f64>::zero(); size_real])),
        //     );
        // }




        println!("data inst {:?}", start.elapsed().as_millis());

        let start = Instant::now();
        let kernel_data = &self.fmm.m2l.operator_data.kernel_data;


        let ntargets = targets.len();
        let mut global_check_potentials = vec![Complex::<f64>::zero(); size_real*ntargets];


        (0..size_real).into_par_iter().zip(global_check_potentials.par_chunks_exact_mut(ntargets)).for_each(|(freq, check_potential_chunk)| {

            // Extract frequency component of signal
            let padded_signal_freq = every_kth(padded_signals_hat.data(), freq, size_real);

            // Compute the Hadamard products over frequency chunks
            padded_signal_freq.chunks_exact(8).zip(targets.chunks_exact(8)).for_each(|(frequency_chunk, target_chunk)| {
                let parent = target_chunk[0].parent();
                let parent_neighbours = parent.neighbors();
                let parent_neighbours_children = parent_neighbours.iter().map(|pn| pn.children()).collect_vec();

                // For each halo position
                for (i, pnc) in parent_neighbours_children.iter().enumerate() {

                    // Find accumulation points of check potentials
                    check_potential_chunk[i] = Complex::<f64>::zero();

                    // For the current frequency, load the kernel chunk
                    let kernel_offset = 64*freq;
                    let kernel_chunk = &kernel_data[i][kernel_offset..kernel_offset+64];

                    // Compute hadamard products, and scatter to appropriate locations
                    // let mut result = vec![Complex::<f64>::zero(); 64];

                    // For each convolution in the halo
                    for (j, child) in pnc.iter().enumerate() {
                        let kernel_chunk_chunk = &kernel_chunk[j*8..(j+1)*8];
                        let tmp = kernel_chunk_chunk
                            .iter()
                            .zip(padded_signal_freq.iter())
                            .map(|(a, &b)| a*b).collect_vec();
                        // result[j*8..(j+1)*8].copy_from_slice(&tmp[..]);
                        // check_chunk.iter_mut().zip(tmp.iter()).for_each(|(c, t)| *c += t);
                    }

                    // Check which positions are accumulated to, this is algebraically defined for each halo position
                    // A subset of 'result' will be saved to check potential chunk

                }
            });
        });


        // let nsiblings = 8;
        // padded_signals_hat.data().par_chunks_exact(size_real*nsiblings).for_each(|sibling_set| {

        //     let nconvolutions = 189;
        //     // let nsiblings = 8;

        //     let mut result = rlst_mat![c64, (nconvolutions * nsiblings * size_real, 1)];

        //     hadamard_product_sibling(self.fmm.order, sibling_set, &&kernel_data[..], result.data_mut());

        // });


        // Iterate through sibling sets at a time, and compute hadamad products with halo items

        // println!("HERE foo {:?}", kernel_data[0].len());

        // for i in 0..kernel_data.len() {
        //     // println!("HERE {:?}", i);

        //     // Load a set of convolutions into memory
        //     let kernel_set = &kernel_data[i];

        //     // Compute hadamard products for each sibling set in targets in parallel, for this given kernel
        //     // set
        //     padded_signals_hat.data().par_chunks_exact(size_real*8).for_each(|sibling_set| {

        //         let nconvolutions = 64;
        //         let nsiblings = 8;

        //         let mut result = rlst_mat![c64, (nconvolutions * nsiblings * size_real, 1)];

        //         hadamard_product_sibling(self.fmm.order, sibling_set, &kernel_set[..], result.data_mut());

        //     })

        // }

        // let nsiblings = 8;
        // targets
        //     .par_chunks_exact(nsiblings)
        //     .zip(
        //         padded_signals_hat
        //             .data()
        //             .par_chunks_exact(nsiblings * size_real),
        //     )
        //     .for_each(|(siblings, sibling_signals)| {
        //         let mut halo_data = Vec::new();

        //         let parent = siblings[0].parent();

        //         let sentinel = MortonKey::default();

        //         let parent_neigbors_children = parent
        //             .all_neighbors()
        //             .iter()
        //             .flat_map(|p| {
        //                 if let Some(p) = p {
        //                     p.children()
        //                 } else {
        //                     vec![sentinel; 8]
        //                 }
        //             })
        //             .collect_vec();

        //         // Get all halo data
        //         for &pnc in parent_neigbors_children.iter() {
        //             if pnc != sentinel {
        //                 halo_data.push(Some(Arc::clone(
        //                     global_check_potentials_hat.get(&pnc).unwrap(),
        //                 )))
        //             } else {
        //                 halo_data.push(None)
        //             }
        //         }

                // Compute Hadamard products and scatter to halo
                // let mut tmp = rlst_mat![c64, (16 * 8 * size_real, 1)];
                // hadamard_product_sibling(self.fmm.order, sibling_signals, kernel_data.data(), tmp.data_mut());

                // println!("HERE {:?} {:?} {:?}", tmp.data()[0], sibling_signals[0], kernel_data.data()[0]);
            // });

        println!("hadamard products {:?}", start.elapsed());
    }

    fn m2l_scale(&self, level: u64) -> f64 {
        if level < 2 {
            panic!("M2L only performed on level 2 and below")
        }
        if level == 2 {
            1. / 2.
        } else {
            2_f64.powf((level - 3) as f64)
        }
    }
}

fn every_kth_mut<T>(slice: &mut [T], offset: usize, k: usize) -> Vec<&mut T> {
    if k == 0 {
        return Vec::new();
    }

    let mut result = Vec::new();
    // Start at the offset
    let mut remainder = &mut slice[offset..];

    while remainder.len() >= k {
        let (front, back) = remainder.split_at_mut(k);
        result.push(&mut front[k.saturating_sub(1)]);  // Using saturating_sub to prevent underflow
        remainder = back;
    }
    result
}

fn every_kth<T>(slice: &[T], offset: usize, k: usize) -> Vec<&T> {
    if k == 0 {
        return Vec::new();
    }

    let mut result = Vec::new();
    // Start at the offset
    let mut remainder = &slice[offset..];

    while remainder.len() >= k {
        let (front, back) = remainder.split_at(k);
        result.push(&front[k.saturating_sub(1)]);  // Using saturating_sub to prevent underflow
        remainder = back;
    }
    result
}