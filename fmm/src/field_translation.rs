// Implementation of field translations
use std::{
    collections::HashMap,
    ops::{Deref, Mul, DerefMut},
    sync::{Arc, Mutex, RwLock, MutexGuard},
};

use bempp_tools::Array3D;
use num::Complex;
use itertools::Itertools;
use rayon::prelude::*;

use bempp_field::{types::{SvdFieldTranslationKiFmm, FftFieldTranslationNaiveKiFmm}, helpers::{pad3, rfft3, irfft3}};
use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, InteractionLists, SourceTranslation, TargetTranslation},
    kernel::{Kernel, KernelScale},
    tree::Tree,
    types::EvalType,
    arrays::Array3DAccess
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};
use rlst::{
    common::traits::*,
    common::tools::PrettyPrint,
    dense::{rlst_col_vec, rlst_mat, rlst_pointer_mat, traits::*, Dot, Shape},
};

use crate::types::{FmmData, KiFmm};

impl<T, U> SourceTranslation for FmmData<KiFmm<SingleNodeTree, T, U>>
where
    T: Kernel<T = f64> + KernelScale<T = f64> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
{
    // TODO: Change back to multithreading over the leaves once Timo has merged trait changes to
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

                    for i in 0..leaf_multipole_lock.shape().0 {
                        leaf_multipole_lock[[i, 0]] += leaf_multipole_owned[[i, 0]];
                    }
                }
            });
        }
    }

    fn m2m<'a>(&self, level: u64) {
        // Parallelise over nodes at a given level
        if let Some(sources) = self.fmm.tree().get_keys(level) {
            sources.par_iter().for_each(move |&source| {
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

                let operator_index = source.siblings().iter().position(|&x| x == source).unwrap();
                let source_multipole_arc = Arc::clone(self.multipoles.get(&source).unwrap());
                let target_multipole_arc =
                    Arc::clone(self.multipoles.get(&source.parent()).unwrap());
                let fmm_arc = Arc::clone(&self.fmm);

                let source_multipole_lock = source_multipole_arc.lock().unwrap();

                let target_multipole_owned =
                    fmm_arc.m2m[operator_index].dot(&source_multipole_lock);

                let mut target_multipole_lock = target_multipole_arc.lock().unwrap();

                for i in 0..ncoeffs {
                    target_multipole_lock[[i, 0]] += target_multipole_owned[[i, 0]];
                }
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
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
                let source_local_arc = Arc::clone(self.locals.get(&target.parent()).unwrap());
                let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());
                let fmm = Arc::clone(&self.fmm);

                let operator_index = target.siblings().iter().position(|&x| x == target).unwrap();

                let source_local_lock = source_local_arc.lock().unwrap();

                let target_local_owned = fmm.l2l[operator_index].dot(&source_local_lock);
                let mut target_local_lock = target_local_arc.lock().unwrap();

                for i in 0..ncoeffs {
                    target_local_lock[[i, 0]] += target_local_owned[[i, 0]];
                }
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

                            for i in 0..ntargets {
                                target_potential_lock[[i, 0]] += target_potential[[i, 0]];
                            }
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

                    for i in 0..ntargets {
                        target_potential_lock[[i, 0]] += target_potential[[i, 0]];
                    }
                }
            })
        }
    }

    fn p2l<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

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

                            for i in 0..ncoeffs {
                                target_local_lock[[i, 0]] += target_local_owned[[i, 0]];
                            }
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

                                for i in 0..ntargets {
                                    target_potential_lock[[i, 0]] += target_potential[[i, 0]];
                                }
                            }
                        }
                    }
                }
            })
        }
    }
}

impl<T> FieldTranslation for FmmData<KiFmm<SingleNodeTree, T, SvdFieldTranslationKiFmm<T>>>
where
    T: Kernel<T = f64> + KernelScale<T = f64> + std::marker::Sync + std::marker::Send + Default,
{
    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else { return };
        let mut transfer_vector_to_m2l =
            HashMap::<usize, Arc<Mutex<Vec<(MortonKey, MortonKey)>>>>::new();

        for tv in self.fmm.m2l.transfer_vectors.iter() {
            transfer_vector_to_m2l.insert(tv.vector, Arc::new(Mutex::new(Vec::new())));
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
                    .position(|x| x.vector == *transfer_vector)
                    .unwrap();

                let (nrows, _) = self.fmm.m2l.m2l.2.shape();
                let top_left = (0, c_idx * self.fmm.m2l.k);
                let dim = (nrows, self.fmm.m2l.k);

                let c_sub = self.fmm.m2l.m2l.2.block(top_left, dim);

                let m2l_rw = m2l_arc.read().unwrap();
                let mut multipoles = rlst_mat![f64, (self.fmm.m2l.k, m2l_rw.len())];

                for (i, (source, _)) in m2l_rw.iter().enumerate() {
                    let source_multipole_arc = Arc::clone(self.multipoles.get(source).unwrap());
                    let source_multipole_lock = source_multipole_arc.lock().unwrap();

                    // Compressed multipole
                    let compressed_source_multipole_owned =
                        self.fmm.m2l.m2l.1.dot(&source_multipole_lock).eval();

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
                    .m2l
                    .0
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

                    for i in 0..target_local_lock.shape().0 {
                        target_local_lock[[i, 0]] += target_local_owned[[i, 0]];
                    }
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


impl<T> FieldTranslation for FmmData<KiFmm<SingleNodeTree, T, FftFieldTranslationNaiveKiFmm<T>>>
where
    T: Kernel<T = f64> + KernelScale<T = f64> + std::marker::Sync + std::marker::Send + Default
{

    fn m2l<'a>(&self, level: u64) {
        let Some(targets) = self.fmm.tree().get_keys(level) else { return };
        
        targets.par_iter().for_each(move |&target| {
            if let Some(v_list) = self.fmm.get_v_list(&target) {
                let fmm_arc = Arc::clone(&self.fmm);
                let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());

                for source in v_list.iter() {

                    let transfer_vector = target.find_transfer_vector(source);

                    // Locate correct precomputed FFT of kernel
                    let k_idx = fmm_arc
                        .m2l
                        .transfer_vectors
                        .iter()
                        .position(|x| x.vector == transfer_vector)
                        .unwrap();

                    // Compute FFT of signal
                    let source_multipole_arc = Arc::clone(self.multipoles.get(source).unwrap());
                    
                    let source_multipole_lock = source_multipole_arc.lock().unwrap();

                    let signal = fmm_arc.m2l.compute_signal(fmm_arc.order, source_multipole_lock.data());

                    // 1. Pad the signal
                    let &(m, n, o) = signal.shape();

                    let p = 2*m;
                    let q = 2*n;
                    let r = 2*o;

                    // TODO: Look carefully how to pad upper left
                    let pad_size = (p-m, q-n, r-o);
                    let pad_index = (p-m, q-n, r-o);
                    let real_dim = q;

                    let padded_signal = pad3(&signal, pad_size, pad_index);

                    let padded_signal_hat = rfft3(&padded_signal);
                    let &(m_, n_, o_) = padded_signal_hat.shape();
                    let len_padded_signal_hat = m_*n_*o_;

                    // 2. Compute the convolution to find the check potential
                    let padded_kernel_hat = &fmm_arc.m2l.m2l[k_idx];
                    let &(m_, n_, o_) = padded_kernel_hat.shape();
                    let len_padded_kernel_hat= m_*n_*o_;
                    
                    // Compute Hadamard product
                    let padded_signal_hat = unsafe {
                        rlst_pointer_mat!['a, Complex<f64>, padded_signal_hat.get_data().as_ptr(), (len_padded_signal_hat, 1), (1,1)]
                    };
                    
                    let padded_kernel_hat= unsafe {
                        rlst_pointer_mat!['a, Complex<f64>, padded_kernel_hat.get_data().as_ptr(), (len_padded_kernel_hat, 1), (1,1)]
                    };

                    let check_potential_hat = padded_kernel_hat.cmp_wise_product(padded_signal_hat).eval();

                    // 3.1 Compute iFFT to find check potentials
                    let check_potential_hat = Array3D::from_data(check_potential_hat.data().to_vec(), (m_, n_, o_));
                    
                    let check_potential = irfft3(&check_potential_hat, real_dim);

                    // Filter check potentials
                    let mut filtered_check_potentials: Array3D<f64> = Array3D::new((m+1, n+1, o+1));
                    for i in (p-m-1)..p {
                        for j in (q-n-1)..q {
                            for k in (r-o-1)..r {
                                let i_= i - (p-m-1);
                                let j_ = j - (q-n-1);
                                let k_ = k - (r-o-1);
                                *filtered_check_potentials.get_mut(i_, j_, k_).unwrap() = *check_potential.get(i, j, k).unwrap();
                            }
                        }
                    }

                    let (_, target_surface_idxs) = target.surface_grid(fmm_arc.order);
                    let mut tmp = Vec::new();
                    let ntargets = target_surface_idxs.len() / fmm_arc.kernel.space_dimension();
                    let xs = &target_surface_idxs[0..ntargets];
                    let ys = &target_surface_idxs[ntargets..2*ntargets];
                    let zs = &target_surface_idxs[2*ntargets..];

                    for i in 0..ntargets {
                        let val = filtered_check_potentials.get(xs[i], ys[i], zs[i]).unwrap();
                        tmp.push(*val);
                    }

                    let check_potential = unsafe {
                        rlst_pointer_mat!['a, f64, tmp.as_ptr(), (ntargets, 1), (1,1)]
                    };

                    // Finally, compute local coefficients from check potential
                    let target_local_owned = (self.m2l_scale(target.level()) 
                        * fmm_arc.kernel.scale(target.level()) 
                        * fmm_arc.dc2e_inv.dot(&check_potential)).eval();


                    let mut target_local_lock = target_local_arc.lock().unwrap();
                    *target_local_lock.deref_mut() = (target_local_lock.deref() + target_local_owned).eval();
                }
            }
        })
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