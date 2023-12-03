//! Implementation of Source and Target translations, as well as Source to Target translation.
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use itertools::Itertools;
use num::Float;
use rayon::prelude::*;

use bempp_traits::{
    field::FieldTranslationData,
    fmm::{Fmm, InteractionLists, TargetTranslation},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::types::single_node::SingleNodeTree;

use rlst::{
    common::traits::*,
    dense::{rlst_col_vec, rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer},
};

use crate::types::{FmmDataHashmap, KiFmmHashMap};

impl<T, U, V> TargetTranslation for FmmDataHashmap<KiFmmHashMap<SingleNodeTree<V>, T, U, V>, V>
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
    fn l2l(&self, level: u64) {
        if let Some(targets) = self.fmm.tree().get_keys(level) {
            targets.par_iter().for_each(move |&target| {
                let source_local_arc = Arc::clone(self.locals.get(&target.parent()).unwrap());
                let target_local_arc = Arc::clone(self.locals.get(&target).unwrap());

                let operator_index = target.siblings().iter().position(|&x| x == target).unwrap();

                let source_local_lock = source_local_arc.lock().unwrap();

                let target_local_owned = self.fmm.l2l[operator_index].dot(&source_local_lock);
                let mut target_local_lock = target_local_arc.lock().unwrap();

                *target_local_lock.deref_mut() =
                    (target_local_lock.deref() + target_local_owned).eval();
            })
        }
    }

    fn m2p<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_all_leaves() {
            targets.par_iter().for_each(move |&target| {


                if let Some(points) = self.fmm.tree().get_points(&target) {
                    let target_potential_arc = Arc::clone(self.potentials.get(&target).unwrap());
                    if let Some(w_list) = self.fmm.get_w_list(&target) {
                        for source in w_list.iter() {
                            let source_multipole_arc =
                                Arc::clone(self.multipoles.get(source).unwrap());

                            let upward_equivalent_surface = source.compute_surface(
                                self.fmm.tree().get_domain(),
                                self.fmm.order(),
                                self.fmm.alpha_inner,
                            );

                            let source_multipole_lock = source_multipole_arc.lock().unwrap();

                            let target_coordinates = points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let ntargets = target_coordinates.len() / self.fmm.kernel.space_dimension();

                            let target_coordinates = unsafe {
                                rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, self.fmm.kernel.space_dimension()), (self.fmm.kernel.space_dimension(), 1)]
                            }.eval();


                            let mut target_potential = rlst_col_vec![V, ntargets];

                            self.fmm.kernel.evaluate_st(
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
        if let Some(targets) = self.fmm.tree().get_all_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let source_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());

                if let Some(target_points) = self.fmm.tree().get_points(&leaf) {
                    let target_potential_arc = Arc::clone(self.potentials.get(&leaf).unwrap());
                    // Lookup data
                    let target_coordinates = target_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();
                    let ntargets = target_coordinates.len() / self.fmm.kernel.space_dimension();

                    let target_coordinates = unsafe {
                        rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, self.fmm.kernel.space_dimension()), (self.fmm.kernel.space_dimension(), 1)]
                    }.eval();

                    let downward_equivalent_surface = leaf.compute_surface(
                        &self.fmm.tree().domain,
                        self.fmm.order,
                        self.fmm.alpha_outer,
                    );

                    let source_local_lock = source_local_arc.lock().unwrap();

                    let mut target_potential = rlst_col_vec![V, ntargets];

                    self.fmm.kernel.evaluate_st(
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
        if let Some(targets) = self.fmm.tree().get_all_leaves() {
            targets.par_iter().for_each(move |&leaf| {
                let target_local_arc = Arc::clone(self.locals.get(&leaf).unwrap());

                if let Some(x_list) = self.fmm.get_x_list(&leaf) {
                    for source in x_list.iter() {
                        if let Some(source_points) = self.fmm.tree().get_points(source) {
                            let source_coordinates = source_points
                                .iter()
                                .map(|p| p.coordinate)
                                .flat_map(|[x, y, z]| vec![x, y, z])
                                .collect_vec();

                            let nsources = source_coordinates.len() / self.fmm.kernel.space_dimension();

                            let source_coordinates = unsafe {
                                rlst_pointer_mat!['a, V, source_coordinates.as_ptr(), (nsources, self.fmm.kernel.space_dimension()), (self.fmm.kernel.space_dimension(), 1)]
                            }.eval();

                            let source_charges = self.charges.get(source).unwrap();

                            let downward_check_surface = leaf.compute_surface(
                                &self.fmm.tree().domain,
                                self.fmm.order,
                                self.fmm.alpha_inner,
                            );

                            let ntargets = downward_check_surface.len() / self.fmm.kernel.space_dimension();
                            let mut downward_check_potential = rlst_col_vec![V, ntargets];

                            self.fmm.kernel.evaluate_st(
                                EvalType::Value,
                                source_coordinates.data(),
                                &downward_check_surface[..],
                                &source_charges[..],
                                downward_check_potential.data_mut()
                            );


                            let mut target_local_lock = target_local_arc.lock().unwrap();
                            let mut tmp = self.fmm.dc2e_inv_1.dot(&self.fmm.dc2e_inv_2.dot(&downward_check_potential)).eval();
                            tmp.data_mut().iter_mut().for_each(|d| *d *=  self.fmm.kernel.scale(leaf.level()));
                            let target_local_owned =  tmp;
                            *target_local_lock.deref_mut() = (target_local_lock.deref() + target_local_owned).eval();
                        }
                    }
                }
            })
        }
    }

    fn p2p<'a>(&self) {
        if let Some(targets) = self.fmm.tree().get_all_leaves() {
            targets.par_iter().for_each(move |&target| {

                if let Some(target_points) = self.fmm.tree().get_points(&target) {
                    let target_potential_arc = Arc::clone(self.potentials.get(&target).unwrap());
                    let target_coordinates = target_points
                        .iter()
                        .map(|p| p.coordinate)
                        .flat_map(|[x, y, z]| vec![x, y, z])
                        .collect_vec();

                    let ntargets= target_coordinates.len() / self.fmm.kernel.space_dimension();

                    let target_coordinates = unsafe {
                        rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, self.fmm.kernel.space_dimension()), (self.fmm.kernel.space_dimension(), 1)]
                    }.eval();

                    if let Some(u_list) = self.fmm.get_u_list(&target) {

                        for source in u_list.iter() {
                            if let Some(source_points) = self.fmm.tree().get_points(source) {
                                let source_coordinates = source_points
                                    .iter()
                                    .map(|p| p.coordinate)
                                    .flat_map(|[x, y, z]| vec![x, y, z])
                                    .collect_vec();

                                let nsources = source_coordinates.len() / self.fmm.kernel.space_dimension();

                                let source_coordinates = unsafe {
                                    rlst_pointer_mat!['a, V, source_coordinates.as_ptr(), (nsources, self.fmm.kernel.space_dimension()), (self.fmm.kernel.space_dimension(), 1)]
                                }.eval();

                                let source_charges_arc =
                                    Arc::clone(self.charges.get(source).unwrap());

                                let mut target_potential = rlst_col_vec![V, ntargets];

                                self.fmm.kernel.evaluate_st(
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


                        let target_potential_lock =
                            target_potential_arc.lock().unwrap();

                        if target.morton == 3 {
                            println!("local result {:?}", target_potential_lock.data());
                        }
                    }
                }
            })
        }
    }
}
