//! Implementation of field translations for each FMM.
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, RwLock},
};

use bempp_tools::Array3D;
use fftw::types::*;
use itertools::Itertools;
use rayon::prelude::*;

use bempp_field::{
    array::pad3,
    fft::{irfft3_fftw_par_vec, rfft3_fftw_par_vec},
    types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    arrays::Array3DAccess,
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, InteractionLists, SourceTranslation, TargetTranslation},
    kernel::{Kernel, KernelScale},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use rlst::{
    common::traits::*,
    dense::{rlst_col_vec, rlst_dynamic_mat, rlst_pointer_mat, traits::*, Dot, Shape},
};

use crate::types::{FmmData, KiFmm};

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
        let Some(targets) = self.fmm.tree().get_keys(level) else {
            return;
        };
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
                let mut multipoles = rlst_dynamic_mat![f64, (self.fmm.m2l.k, m2l_rw.len())];

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

                // println!("check potential svd {:?}", check_potential_owned.data());

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

#[derive(Clone, Debug, Copy)]
struct SendPtrMut<T> {
    raw: *mut T,
}

unsafe impl<T> Sync for SendPtrMut<T> {}

#[derive(Clone, Debug, Copy)]
struct SendPtr<T> {
    raw: *const T,
}

unsafe impl<T> Sync for SendPtr<T> {}

/// Implement the multipole to local translation operator for an FFT accelerated KiFMM on a single node.
impl<T> FieldTranslation for FmmData<KiFmm<SingleNodeTree, T, FftFieldTranslationKiFmm<T>>>
where
    T: Kernel<T = f64> + KernelScale<T = f64> + std::marker::Sync + std::marker::Send + Default,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else {
            return;
        };

        // Form signals to use for convolution first
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
        let mut padded_signals = rlst_col_vec![f64, size * ntargets];

        let chunks = padded_signals.data_mut().par_chunks_exact_mut(size);

        let range = (0..chunks.len()).into_par_iter();
        range.zip(chunks).for_each(|(i, chunk)| {
            let fmm_arc = Arc::clone(&self.fmm);
            let target = targets[i];
            let source_multipole_arc = Arc::clone(self.multipoles.get(&target).unwrap());
            let source_multipole_lock = source_multipole_arc.lock().unwrap();
            let signal = fmm_arc
                .m2l
                .compute_signal(fmm_arc.order, source_multipole_lock.data());

            let padded_signal = pad3(&signal, pad_size, pad_index);

            chunk.copy_from_slice(padded_signal.get_data());
        });

        let mut padded_signals_hat = rlst_col_vec![c64, size_real * ntargets];
        rfft3_fftw_par_vec(&mut padded_signals, &mut padded_signals_hat, &[p, q, r]);

        let kernel_data_halo = &self.fmm.m2l.operator_data.kernel_data_rearranged;
        let ntargets = targets.len();
        let nparents = ntargets / 8;
        let mut global_check_potentials_hat = rlst_col_vec![c64, size_real * ntargets];
        let mut global_check_potentials = rlst_col_vec![f64, size * ntargets];

        // Get check potentials in frequency order
        let mut global_check_potentials_hat_freq = vec![Vec::new(); size_real];

        unsafe {
            let ptr = global_check_potentials_hat.get_pointer_mut();
            for (i, elem) in global_check_potentials_hat_freq
                .iter_mut()
                .enumerate()
                .take(size_real)
            {
                for j in 0..ntargets {
                    let raw = ptr.offset((j * size_real + i).try_into().unwrap());
                    let send_ptr = SendPtrMut { raw };
                    elem.push(send_ptr);
                }
            }
        }

        // Get signals into frequency order
        let mut padded_signals_hat_freq = vec![Vec::new(); size_real];
        let zero = rlst_col_vec![c64, 8];
        unsafe {
            let ptr = padded_signals_hat.get_pointer();

            for (i, elem) in padded_signals_hat_freq
                .iter_mut()
                .enumerate()
                .take(size_real)
            {
                for j in 0..ntargets {
                    let raw = ptr.offset((j * size_real + i).try_into().unwrap());
                    let send_ptr = SendPtr { raw };
                    elem.push(send_ptr);
                }
                // put in a bunch of zeros at the end
                let ptr = zero.get_pointer();
                for _ in 0..8 {
                    let send_ptr = SendPtr { raw: ptr };
                    elem.push(send_ptr)
                }
            }
        }

        // Create a map between targets and index positions in vec of len 'ntargets'
        let mut target_map = HashMap::new();

        for (i, t) in targets.iter().enumerate() {
            target_map.insert(t, i);
        }

        // Find all the displacements used for saving results
        let mut all_displacements = Vec::new();
        targets.chunks_exact(8).for_each(|sibling_chunk| {
            // not in Morton order (refer to sort method when called on 'neighbours')
            let parent_neighbours: Vec<Option<MortonKey>> =
                sibling_chunk[0].parent().all_neighbors();

            let displacements = parent_neighbours
                .iter()
                .map(|pn| {
                    let mut tmp = Vec::new();
                    if let Some(pn) = pn {
                        if self.fmm.tree.keys_set.contains(pn) {
                            let mut children = pn.children();
                            children.sort();
                            for child in children {
                                // tmp.push(*target_map.get(&child).unwrap() as i64)
                                tmp.push(*target_map.get(&child).unwrap())
                            }
                        } else {
                            for i in 0..8 {
                                // tmp.push(-1 as i64)
                                tmp.push(ntargets + i)
                            }
                        }
                    } else {
                        for i in 0..8 {
                            tmp.push(ntargets + i)
                        }
                    }

                    assert!(tmp.len() == 8);
                    tmp
                })
                .collect_vec();
            all_displacements.push(displacements);
        });

        let scale = self.m2l_scale(level);

        (0..size_real).into_par_iter().for_each(|freq| {
            // Extract frequency component of signal (ntargets long)
            let padded_signal_freq = &padded_signals_hat_freq[freq];

            // Extract frequency components of save locations (ntargets long)
            let check_potential_freq = &global_check_potentials_hat_freq[freq];

            (0..nparents).for_each(|sibling_idx| {
                // lookup associated save locations for our current sibling set
                let save_locations =
                    &check_potential_freq[(sibling_idx * 8)..(sibling_idx + 1) * 8];
                let save_locations_raw = save_locations.iter().map(|s| s.raw).collect_vec();

                // for each halo position compute convolutions to a given sibling set
                for (i, kernel_data) in kernel_data_halo.iter().enumerate().take(26) {
                    let frequency_offset = 64 * freq;
                    let kernel_data_i = &kernel_data[frequency_offset..(frequency_offset + 64)];

                    // Find displacements for signal being translated
                    let displacements = &all_displacements[sibling_idx][i];

                    // Lookup signal to be translated if a translation is to be performed
                    let signal = &padded_signal_freq[(displacements[0])..=(displacements[7])];

                    for j in 0..8 {
                        let kernel_data_ij = &kernel_data_i[j * 8..(j + 1) * 8];
                        let sig = signal[j].raw;
                        unsafe {
                            save_locations_raw
                                .iter()
                                .zip(kernel_data_ij.iter())
                                .for_each(|(&sav, ker)| *sav += scale * *sig * ker)
                        }
                    } // inner loop
                }
            }); // over each sibling set
        });

        irfft3_fftw_par_vec(
            &mut global_check_potentials_hat,
            &mut global_check_potentials,
            &[p, q, r],
        );

        // Compute local expansion coefficients and save to data tree
        let (_, multi_indices) = MortonKey::surface_grid(self.fmm.order);

        let check_potentials = global_check_potentials
            .data()
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

        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
        let check_potentials = unsafe {
            rlst_pointer_mat!['a, f64, check_potentials.as_ptr(), (ncoeffs, ntargets), (1, ncoeffs)]
        };

        let locals =
            (self.fmm.dc2e_inv.dot(&check_potentials) * self.fmm.kernel.scale(level)).eval();

        for (i, target) in targets.iter().enumerate() {
            let target_local_arc = Arc::clone(self.locals.get(target).unwrap());
            let mut target_local_lock = target_local_arc.lock().unwrap();

            let top_left = (0, i);
            let dim = (ncoeffs, 1);
            let target_local_owned = locals.block(top_left, dim);

            *target_local_lock.deref_mut() =
                (target_local_lock.deref() + target_local_owned).eval();
        }
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
