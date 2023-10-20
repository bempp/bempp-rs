//! Implementation of field translations for each FMM.
use std::{
    collections::{HashMap, HashSet},
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
    array::{pad3, flip3},
    fft::{irfft3_fftw, irfft3_fftw_par_vec, rfft3_fftw, rfft3_fftw_par_vec},
    hadamard::hadamard_product_sibling,
    types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    arrays::Array3DAccess,
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, InteractionLists, SourceTranslation, TargetTranslation},
    kernel::{Kernel, KernelScale, self},
    tree::{Tree, MortonKeyInterface},
    types::EvalType,
};
use bempp_tree::{types::{morton::MortonKey, single_node::SingleNodeTree}, constants::ROOT, implementations::helpers::find_corners};
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
    raw: *mut T
}

unsafe impl<T> Sync for SendPtrMut<T> {}

#[derive(Clone, Debug, Copy)]
struct SendPtr<T> {
    raw: *const T
}

unsafe impl<T> Sync for SendPtr<T> {}

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
        let mut padded_signals = rlst_col_vec![f64, size * ntargets];

        let mut chunks = padded_signals.data_mut().par_chunks_exact_mut(size);

        let range = (0..chunks.len()).into_par_iter();
        range.zip(chunks).for_each(|(i, chunk)| {
            let fmm_arc = Arc::clone(&self.fmm);
            let target = targets[i];
            // let source_multipole_arc = Arc::clone(self.multipoles.get(&target).unwrap());
            // let source_multipole_lock = source_multipole_arc.lock().unwrap();
            // let signal = fmm_arc
            //     .m2l
            //     .compute_signal(fmm_arc.order, source_multipole_lock.data());

            // TO REMOVE
            let ncoeffs =  6*(self.fmm.order-1).pow(2) + 2;
            let mut source_multipole_lock = vec![1 as f64; ncoeffs];


            let signal = fmm_arc
                .m2l
                .compute_signal(fmm_arc.order, &source_multipole_lock);

            let mut padded_signal = pad3(&signal, pad_size, pad_index);

            chunk.copy_from_slice(padded_signal.get_data());
        });

        println!("data organisation time {:?}", start.elapsed().as_millis());

        let start = Instant::now();
        let mut padded_signals_hat = rlst_col_vec![c64, size_real * ntargets];
        rfft3_fftw_par_vec(&mut padded_signals, &mut padded_signals_hat, &[p, q, r]);
        println!("fft time {:?}", start.elapsed().as_millis());

        // TO REMOVE (fill with unique identifiers for each target)
        let mut padded_signals_hat = rlst_col_vec![c64, size_real*ntargets];
        for i in 0..ntargets {
            let tmp =  vec![Complex::new(i as f64, 0 as f64); size_real];
            padded_signals_hat.data_mut()[i*size_real..(i+1)*size_real].copy_from_slice(&tmp);
        }


        let start = Instant::now();
        let kernel_data = &self.fmm.m2l.operator_data.kernel_data_rearranged;
        let ntargets = targets.len();
        let mut global_check_potentials_hat = rlst_col_vec![c64, size_real*ntargets];
        let mut global_check_potentials = rlst_col_vec![f64, size*ntargets];

        // Get check potentials in frequency order
        let mut global_check_potentials_hat_freq = vec![Vec::new(); size_real];

        unsafe {
            let ptr = global_check_potentials_hat.get_pointer_mut();
            for i in 0..size_real {
                for j in 0..ntargets {
                    let raw = ptr.offset((j*size_real+i).try_into().unwrap());
                    let send_ptr = SendPtrMut{raw};
                    global_check_potentials_hat_freq[i].push(send_ptr);
                }
            }
        }

        // Get signals into frequency order
        let mut padded_signals_hat_freq = vec![Vec::new(); size_real];
        unsafe {
            let ptr = padded_signals_hat.get_pointer();
            for i in 0..size_real {
                for j in 0..ntargets {
                    let raw = ptr.offset((j*size_real+i).try_into().unwrap());
                    let send_ptr = SendPtr{raw};
                    padded_signals_hat_freq[i].push(send_ptr);
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
            let parent_neighbours: Vec<Option<MortonKey>> = sibling_chunk[0].parent().all_neighbors();

            let displacements = parent_neighbours.iter().map(|pn| {
                let mut tmp = Vec::new();
                if let Some(pn) = pn {
                    if self.fmm.tree.keys_set.contains(pn) {
                        for child in pn.children() {
                            tmp.push(*target_map.get(&child).unwrap() as i64)
                        }
                    } else {
                        for _ in 0..8 {
                            tmp.push(-1 as i64)
                        }
                    }
                } else {
                    for _ in 0..8 {
                        tmp.push(-1 as i64)
                    }
                }
                tmp
            })
            .collect_vec();
            all_displacements.push(displacements);
        });

        println!("data inst {:?}", start.elapsed().as_millis());
        println!("displacements {:?}, {:?} {:?}", all_displacements.len(), all_displacements[0].len(), all_displacements[0][0].len());

        let scale = self.m2l_scale(level);

        (0..size_real)
            .into_par_iter()
            // .into_iter()
            .for_each(|(freq)| {

            // Extract frequency component of signal (ntargets long)
            let padded_signal_freq = &padded_signals_hat_freq[freq];

            // Extract frequency components of save locations (ntargets long)
            let check_potential_freq = &global_check_potentials_hat_freq[freq];

            // println!("ntargets {:?}={:?}={:?} freq {:?}", targets.len(), padded_signal_freq.len(), check_potential_freq.len(), freq);

            // Iterate over sets of siblings
            padded_signal_freq
                    .chunks_exact(8)
                    .enumerate()
                    .for_each(|(sibling_idx, signal_sibling_chunk)| {

                        // if sibling_idx == 1 {
                        //     unsafe {
                        //         println!("freq {:?} kernel chunk {:?}", freq, signal_sibling_chunk.iter().map(|s| *s.raw).collect_vec());
                        //     }
                        // }
                        // Index locations of save locations for this sibling set.
                        let displacements = &all_displacements[sibling_idx];

                        // if freq == 0 && sibling_idx == 1 {
                        //     // println!("displacements {:?}", displacements.iter().map(|d| d[0]).collect_vec());
                        //     println!("displacements {:?}", displacements);
                        //     println!("parent neighbours {:?}", targets[sibling_idx*8].parent().all_neighbors());
                        //     println!("siblings {:?}", targets[sibling_idx*8].parent().neighbors().len());
                        //     println!("")
                        // }

                        if freq == 0 {
                            unsafe {
                                println!("sibling idx {:?} SIBLING SIGNAL CHUNK {:?}", sibling_idx*8, signal_sibling_chunk.iter().map(|s| *s.raw).collect_vec())
                            }

                        }

                        // For each halo position
                        for i in 0..26 {

                            // For the current frequency, load the kernel chunk, for the current halo position
                            let frequency_offset = 64*freq;

                            // Find the kernel evaluations for the set of siblings in this halo position
                            // expected to be in order [s1, ..., s8, s1, ..., s8, ....] for each halo child position
                            let kernel_chunk = &kernel_data[i][frequency_offset..frequency_offset+64];

                            // if sibling_idx == 15 {
                            //     // unsafe {
                            //     //     println!("halo pos {:?} freq {:?} kernel chunk {:?}", i, freq, kernel_chunk.iter().map(|s| *s).collect_vec());
                            //     // }
                            //         println!("displacements [{:?}] {:?}", i, displacements[i])
                            // }

                            // println!("HERE frq = {:?} i={:?} {:?}", freq, i, kernel_chunk);

                            // Grab the displacements
                            if displacements[i][0] > -1 {

                                let save_locations = &check_potential_freq[(displacements[i][0] as usize)..=(displacements[i][7] as usize)];

                                // For each possible convolution in this halo position
                                for j in 0..8 {

                                    // I want this to be of form [s1,..,s8] for this freq
                                    let kernel_chunk_chunk = &kernel_chunk[j*8..(j+1)*8];

                                    // if freq == 1 && sibling_idx == 0 {
                                    //     println!("KERNEL CHUNK j={:?} {:?}", j, kernel_chunk_chunk);
                                    // }

                                    unsafe {
                                        save_locations.iter().zip(kernel_chunk_chunk.iter()).zip(signal_sibling_chunk.iter())
                                            .for_each(|((sav, ker), &sig)| {
                                                *sav.raw +=  (scale * ker * *sig.raw);
                                            });
                                    }
                                }

                            }
                        }
                    });
        });

        println!("hadamard products {:?}", start.elapsed());

        //// TEST CODE
        // if level == 3 {
        //     let tgt_idx = 455;
        //     let tmp = &global_check_potentials_hat.data()[tgt_idx*size_real..(tgt_idx+1)*size_real];
        //     println!("fft check pot hat {:?}", tmp.iter().map(|c| c.re).collect_vec());
        //     println!("fft check pot hat {:?}", tmp.iter().map(|c| c.im).collect_vec());

        // }
        ////////////////////////////////////////////

        let start = Instant::now();
        irfft3_fftw_par_vec(&mut global_check_potentials_hat, &mut global_check_potentials, &[p, q, r]);
        println!("ifft time {:?}", start.elapsed().as_millis());


        /// TEST CODE TO BE REMOVED
        if level == 3 {

            let mut full_v_lists = Vec::new();
            for (i, tgt) in targets.iter().enumerate() {
                let mut v_list: Vec<MortonKey> = tgt
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| !tgt.is_adjacent(pnc))
                    .collect();

                let mut counter = 0;
                for source in v_list.iter() {
                    if self.fmm.tree.keys_set.contains(source) {
                        counter += 1;
                    }
                }

                if counter == 189 {
                    full_v_lists.push(i);
                }
            }

            // println!("full v lists {:?}", full_v_lists);

            let tgt_idx = 0;
            let tgt_idx = full_v_lists[7];
            let tgt = targets[tgt_idx];
            let target_check_surface = tgt.compute_surface(&self.fmm.tree().domain, self.fmm.order, self.fmm.alpha_inner);
            let n_target_check_surface = target_check_surface.len() / 3;
            let ncoeffs = 6 * (self.fmm.order - 1 ).pow(2) + 2;
            let mut tst_pot = vec![0f64; ncoeffs];

            let mut v_list: Vec<MortonKey> = tgt
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| !tgt.is_adjacent(pnc))
                .collect();

            let n_corners = 8;

            let mut test_check_potential_hat = vec![Complex::<f64>::zero(); size_real];

            for source in v_list.iter() {
                if self.fmm.tree.keys_set.contains(source) {

                    let source_equivalent_surface = source.compute_surface(&self.fmm.tree().domain, self.fmm.order, self.fmm.alpha_inner);
                    let multipole = vec![1.0; ncoeffs];
                    let mut direct = vec![0f64; ncoeffs];

                    self.fmm.m2l.kernel.evaluate_st(
                        EvalType::Value,
                        &source_equivalent_surface[..],
                        &target_check_surface[..],
                        &multipole[..],
                        &mut direct[..],
                    );

                    tst_pot.iter_mut().zip(direct.iter()).for_each(|(t, d)| *t+= d);

                    // Compute FFT kernels for each source box
                    let conv_point_corner_index = 7;
                    let corners = find_corners(&source_equivalent_surface[..]);
                    let conv_point_corner = [
                        corners[conv_point_corner_index],
                        corners[n_corners + conv_point_corner_index],
                        corners[2*n_corners + conv_point_corner_index],
                    ];

                    let (conv_grid, _) = source.convolution_grid(
                        self.fmm.order, &self.fmm.tree.domain, self.fmm.alpha_inner, &conv_point_corner, conv_point_corner_index);

                    // Calculate Green's fct evaluations with respect to a 'kernel point' on the target box
                    let kernel_point_index = 0;
                    let kernel_point = [
                        target_check_surface[kernel_point_index],
                        target_check_surface[n_target_check_surface + kernel_point_index],
                        target_check_surface[2*n_target_check_surface + kernel_point_index],
                    ];

                    // Compute Green's fct evaluations
                    let kernel = self.fmm.m2l.compute_kernel(self.fmm.order, &conv_grid, kernel_point);
                    let (m, n, o) = kernel.shape();
                    let p = m + 1;
                    let q = n + 1;
                    let r: usize = o + 1;

                    let padded_kernel = pad3(&kernel, (p-m, q-n, r-o), (0, 0, 0));
                    let mut padded_kernel = flip3(&padded_kernel);

                    // Compute FFT of padded kernel
                    let mut padded_kernel_hat = Array3D::<c64>::new((p, q, r/2 + 1));
                    rfft3_fftw(padded_kernel.get_data_mut(), padded_kernel_hat.get_data_mut(), &[p, q, r]);

                    // Compute FFT of the representative signal
                    // let signal = self.fmm.m2l.compute_signal(self.fmm.order, &multipole[..]);
                    // let &(m, n, o) = signal.shape();
                    // let p = m + 1;
                    // let q = n + 1;
                    // let r = o + 1;
                    // let pad_size = (p - m, q - n, r - o);
                    // let pad_index = (p - m, q - n, r - o);
                    // let mut padded_signal = pad3(&signal, pad_size, pad_index);
                    // let mut padded_signal_hat = Array3D::<c64>::new((p, q, r / 2 + 1));

                    // rfft3_fftw(
                    //     padded_signal.get_data_mut(),
                    //     padded_signal_hat.get_data_mut(),
                    //     &[p, q, r],
                    // );
                    let padded_signal_hat = &padded_signals_hat.data()[tgt_idx*size_real..(tgt_idx+1)*size_real];
                    // Compute convolution
                    let hadamard_product = padded_signal_hat
                        // .get_data()
                        .iter()
                        .zip(padded_kernel_hat.get_data().iter())
                        .map(|(a, b)| a * b)
                        .collect_vec();

                    test_check_potential_hat.iter_mut().zip(hadamard_product.iter()).for_each(|(t, h)| *t += h);
                    // let mut hadamard_product = Array3D::from_data(hadamard_product, (p, q, r / 2 + 1));
                }
            }

            // let mut test_check_potential = vec![0f64; size];
            let mut test_check_potential = Array3D::new((p, q, r));
            // println!("fft check pot hat {:?}", test_check_potential_hat.iter().map(|c| c.re).collect_vec());
            // println!("fft check pot hat {:?}", test_check_potential_hat.iter().map(|c| c.im).collect_vec());

            irfft3_fftw(&mut test_check_potential_hat[..], test_check_potential.get_data_mut(), &[p, q, r]);
            let (_, multi_indices) = MortonKey::surface_grid(self.fmm.order);

            let mut tmp = Vec::new();
            let ntargets = multi_indices.len() / 3;
            let xs = &multi_indices[0..ntargets];
            let ys = &multi_indices[ntargets..2 * ntargets];
            let zs = &multi_indices[2 * ntargets..];

            for i in 0..ntargets {
                let val = test_check_potential.get(zs[i], ys[i], xs[i]).unwrap();
                tmp.push(*val);
            }

            // println!("test check pot hat {:?}", test_check_potential_hat);
            // println!("test check pot hat {:?}", tmp);
            // let tst_global_check_potentials_hat = &global_check_potentials_hat.data()[tgt_idx*size_real..(tgt_idx+1)*size_real];
            // println!("found {:?}", &test_check_potential.get_data());
            // println!("FOO {:?}", &test_check_potential.get_data()[0..10]);
            // println!("test check pot hat {:?}", &global_check_potentials.data()[tgt_idx*size..(tgt_idx+1)*size]);
            // println!("direct pot {:?}", tst_pot);

            let fft_pot = &global_check_potentials.data()[tgt_idx*size..(tgt_idx+1)*size];

            let m = 2*self.fmm.order - 1;
            let p = m + 1;
            let mut potentials = Array3D::new((p, p, p));
            potentials.get_data_mut().copy_from_slice(fft_pot);

            let (_, multi_indices) = MortonKey::surface_grid(self.fmm.order);

            let mut tmp = Vec::new();
            let ntargets = multi_indices.len() / 3;
            let xs = &multi_indices[0..ntargets];
            let ys = &multi_indices[ntargets..2 * ntargets];
            let zs = &multi_indices[2 * ntargets..];

            for i in 0..ntargets {
                let val = potentials.get(zs[i], ys[i], xs[i]).unwrap();
                tmp.push(*val);
            }

            // println!("fft pot {:?}", tmp)
        }

        //////////////////////////////////``


        // Compute local expansion coefficients and save to data tree
        let start = Instant::now();

        let (_, multi_indices) = MortonKey::surface_grid(self.fmm.order);
        let check_potentials = global_check_potentials.data().chunks_exact(size).map(|chunk| {
            let m = 2*self.fmm.order - 1;
            let p = m + 1;
            let mut potentials = Array3D::new((p, p, p));
            potentials.get_data_mut().copy_from_slice(chunk);

            let (_, multi_indices) = MortonKey::surface_grid(self.fmm.order);

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
        }).flat_map(|c| c).collect_vec();

        // println!("check potentials {:?}", &check_potentials[0..8]);
        println!("local time {:?}", start.elapsed().as_millis());
        // println!("check potentials {:?} {:?}", &global_check_potentials.data()[0..10], &global_check_potentials_hat.data()[0..10]);

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
