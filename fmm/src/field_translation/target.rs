//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.

use std::{collections::HashSet, ops::Add};

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
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::{
    constants::L2L_MAX_CHUNK_SIZE,
    types::{FmmData, FmmDataSparse, KiFmmLinear, FmmDataAdaptive},
    helpers::find_chunk_size
};

use rlst::{
    common::traits::*,
    dense::{rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer, rlst_dynamic_mat, rlst_col_vec},
};


impl<T, U, V> TargetTranslation for FmmData<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
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
        if let Some(parent_sources) = self.fmm.tree().get_keys(level - 1) {
            if let Some(_child_targets) = self.fmm.tree().get_keys(level) {
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
                let nsiblings = 8;

                let nsources = parent_sources.len();
                let min_parent = &parent_sources[0];
                let max_parent = &parent_sources[nsources - 1];

                let min_parent_idx = self.fmm.tree().key_to_index.get(min_parent).unwrap();
                let max_parent_idx = self.fmm.tree().key_to_index.get(max_parent).unwrap();

                let parent_locals =
                    &self.locals[min_parent_idx * ncoeffs..(max_parent_idx + 1) * ncoeffs];

                let child_locals = &self.level_locals[level as usize];

                let mut max_chunk_size = 8_i32.pow((level).try_into().unwrap()) as usize;

                if max_chunk_size > L2L_MAX_CHUNK_SIZE {
                    max_chunk_size = L2L_MAX_CHUNK_SIZE;
                }
                let chunk_size = find_chunk_size(nsources, max_chunk_size);

                parent_locals
                    .par_chunks_exact(ncoeffs*chunk_size)
                    .zip(child_locals.par_chunks_exact(chunk_size*nsiblings))
                    .for_each(|(parent_local_chunk, children_locals)| {

                        let parent_local_chunk = unsafe { rlst_pointer_mat!['a, V, parent_local_chunk.as_ptr(), (ncoeffs, chunk_size), (1, ncoeffs)] };

                        for i in 0..8 {
                            // Evaluate all L2L for this position in local chunk
                            let tmp = self.fmm.l2l[i].dot(&parent_local_chunk).eval();

                            // Assign for each child in this chunk at this position
                            for j in 0..chunk_size {
                                let chunk_displacement = j*nsiblings;
                                let child_displacement = chunk_displacement + i;
                                let mut ptr = children_locals[child_displacement].raw;

                                unsafe {
                                    for k in 0..ncoeffs {
                                        *ptr += tmp.data()[(j*ncoeffs)+k];
                                        ptr = ptr.add(1);
                                    }
                                }

                            }
                        }
                    });
            }
        }
    }

    fn m2p<'a>(&self) {
      
    }

    fn l2p<'a>(&self) {
        if let Some(_leaves) = self.fmm.tree().get_all_leaves() {
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
            let dim = self.fmm.kernel.space_dimension();
            let surface_size = ncoeffs * dim;

            self.leaf_upward_surfaces
                .par_chunks_exact(surface_size)
                .zip(self.leaf_locals.into_par_iter())
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers)
                .for_each(
                    |(
                        ((leaf_downward_equivalent_surface, local_ptr), charge_index_pointer),
                        potential_send_ptr,
                    )| {
                        let target_coordinates = &coordinates
                            [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                        let ntargets = target_coordinates.len() / dim;
                        let target_coordinates = unsafe {
                            rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, dim), (dim, 1)]
                        }.eval();

                        let local_expansion =
                            unsafe { rlst_pointer_mat!['a, V, local_ptr.raw, (ncoeffs, 1), (1, ncoeffs) ]};


                        // Compute direct
                        if ntargets > 0 {
                            let result = unsafe { std::slice::from_raw_parts_mut(potential_send_ptr.raw, ntargets)};

                            self.fmm.kernel.evaluate_st(
                                EvalType::Value,
                                leaf_downward_equivalent_surface,
                                target_coordinates.data(),
                                local_expansion.data(),
                                result,
                            );

                        }
                    },
                );
        }
    }

    fn p2p<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let dim = self.fmm.kernel.space_dimension();
            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

            leaves
                .par_iter()
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers)
                .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                    let targets =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = targets.len() / dim;
                    let targets = unsafe {
                        rlst_pointer_mat!['a, V, targets.as_ptr(), (ntargets, dim), (dim, 1)]
                    }.eval();

                    if ntargets > 0 {

                        if let Some(u_list) = self.fmm.get_u_list(leaf) {

                            let u_list_indices = u_list
                                .iter()
                                .filter_map(|k| self.fmm.tree().get_leaf_index(k));

                            let charges = u_list_indices
                                .clone()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &self.charges[index_pointer.0..index_pointer.1]
                                })
                                .collect_vec();

                            let sources_coordinates = u_list_indices
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
                                })
                                .collect_vec();

                            for (&charges, sources) in charges.iter().zip(sources_coordinates) {
                                let nsources = sources.len() / dim;
                                let sources = unsafe {
                                    rlst_pointer_mat!['a, V, sources.as_ptr(), (nsources, dim), (dim, 1)]
                                }.eval();


                                if nsources > 0 {
                                    let result = unsafe { std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)};
                                    self.fmm.kernel.evaluate_st(
                                        EvalType::Value,
                                        sources.data(),
                                        targets.data(),
                                        charges,
                                        result,
                                    );
                                }
                            }
                        }
                    }
                })
        }
    }
}

impl<T, U, V> TargetTranslation for FmmDataSparse<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
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
        if let Some(child_targets) = self.fmm.tree().get_keys(level) {
            let nsiblings = 8;
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let parent_sources: HashSet<MortonKey> =
                child_targets.iter().map(|source| source.parent()).collect();
            let mut parent_sources = parent_sources.into_iter().collect_vec();
            parent_sources.sort();
            let nparents = parent_sources.len();
            let mut parent_locals = Vec::new();
            for parent in parent_sources.iter() {
                let parent_index_pointer = *self.level_index_pointer[(level - 1) as usize]
                    .get(parent)
                    .unwrap();
                let parent_local = self.level_locals[(level - 1) as usize][parent_index_pointer];
                parent_locals.push(parent_local);
            }

            let mut max_chunk_size = nparents;
            if max_chunk_size > L2L_MAX_CHUNK_SIZE {
                max_chunk_size = L2L_MAX_CHUNK_SIZE
            }
            let chunk_size = find_chunk_size(nparents, max_chunk_size);

            let child_locals = &self.level_locals[level as usize];

            parent_locals
                .par_chunks_exact(chunk_size)
                .zip(child_locals.par_chunks_exact(nsiblings*chunk_size))
                .for_each(|(parent_local_pointer, child_local_pointers)| {

                    let mut parent_locals = rlst_dynamic_mat![V, (ncoeffs, chunk_size)];
                    for chunk_idx in 0..chunk_size {
                        let tmp = unsafe { rlst_pointer_mat!['a, V, parent_local_pointer[chunk_idx].raw, (ncoeffs, 1), (1, ncoeffs)] };
                        parent_locals.data_mut()[chunk_idx*ncoeffs..(chunk_idx+1)*ncoeffs].copy_from_slice(tmp.data());
                    }                    

                    for i in 0..8 {
                        let tmp = self.fmm.l2l[i].dot(&parent_locals).eval();

                        for j in 0..chunk_size {
                            let chunk_displacement = j*nsiblings;
                            let child_displacement = chunk_displacement + i;
                            let mut ptr = child_local_pointers[child_displacement].raw;
                            unsafe {
                                for k in 0..ncoeffs {
                                    *ptr += tmp.data()[(j*ncoeffs)+k];
                                    ptr = ptr.add(1);
                                }
                            }
                        }
                    }
                });

        }
    }

    fn m2p<'a>(&self) {}

    fn l2p<'a>(&self) {
        if let Some(_leaves) = self.fmm.tree().get_all_leaves() {
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
            let dim = self.fmm.kernel.space_dimension();
            let surface_size = ncoeffs * dim;

            self.leaf_upward_surfaces
                .par_chunks_exact(surface_size)
                .zip(self.leaf_locals.into_par_iter())
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers)
                .for_each(
                    |(
                        ((leaf_downward_equivalent_surface, local_ptr), charge_index_pointer),
                        potential_send_ptr,
                    )| {
                        let target_coordinates = &coordinates
                            [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                        let ntargets = target_coordinates.len() / dim;
                        let target_coordinates = unsafe {
                            rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, dim), (dim, 1)]
                        }.eval();

                        let local_expansion =
                            unsafe { rlst_pointer_mat!['a, V, local_ptr.raw, (ncoeffs, 1), (1, ncoeffs) ]};


                        // Compute direct
                        if ntargets > 0 {
                            let result = unsafe { std::slice::from_raw_parts_mut(potential_send_ptr.raw, ntargets)};

                            self.fmm.kernel.evaluate_st(
                                EvalType::Value,
                                leaf_downward_equivalent_surface,
                                target_coordinates.data(),
                                local_expansion.data(),
                                result,
                            );

                        }
                    },
                );
        }
    }

    fn p2p<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let dim = self.fmm.kernel.space_dimension();

            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

            leaves
                .par_iter()
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers)
                .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                    let targets =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = targets.len() / dim;
                    let targets = unsafe {
                        rlst_pointer_mat!['a, V, targets.as_ptr(), (ntargets, dim), (dim, 1)]
                    }.eval();

                    if ntargets > 0 {

                        if let Some(u_list) = self.fmm.get_u_list(leaf) {

                            let u_list_indices = u_list
                                .iter()
                                .filter_map(|k| self.fmm.tree().get_leaf_index(k));

                            let charges = u_list_indices
                                .clone()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &self.charges[index_pointer.0..index_pointer.1]
                                })
                                .collect_vec();

                            let sources_coordinates = u_list_indices
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
                                })
                                .collect_vec();

                            for (&charges, sources) in charges.iter().zip(sources_coordinates) {
                                let nsources = sources.len() / dim;
                                let sources = unsafe {
                                    rlst_pointer_mat!['a, V, sources.as_ptr(), (nsources, dim), (dim, 1)]
                                }.eval();


                                if nsources > 0 {
                                    let result = unsafe { std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)};
                                    self.fmm.kernel.evaluate_st(
                                        EvalType::Value,
                                        sources.data(),
                                        targets.data(),
                                        charges,
                                        result,
                                    );
                                }
                            }
                        }
                    }
                })
        }
    }
}

impl<T, U, V> TargetTranslation for FmmDataAdaptive<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
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
        if let Some(child_targets) = self.fmm.tree().get_keys(level) {
            let nsiblings = 8;
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let mut child_targets = child_targets.iter().cloned().collect_vec();
            child_targets.sort();
            
            let parent_sources: HashSet<MortonKey> =
                child_targets.iter().map(|source| source.parent()).collect();
            let mut parent_sources = parent_sources.into_iter().collect_vec();
            parent_sources.sort();

            let mut parent_locals = Vec::new();
            for parent in parent_sources.iter() {
                // let parent_index_pointer = *self.level_index_pointer[(level - 1) as usize]
                //     .get(parent)
                //     .unwrap();
                // let parent_local = self.level_locals[(level - 1) as usize][parent_index_pointer];
                // parent_locals.push(parent_local);

                if let Some(parent_index_pointer) = self.level_index_pointer[(level-1) as usize]
                    .get(parent) {
                        let child_local = self.level_locals[(level-1) as usize][*parent_index_pointer];
                        parent_locals.push(child_local);
                    } else {
                        println!("not found parent {:?}", self.fmm.tree().get_all_keys_set().contains(parent))
                    }
            }

            let child_locals = &self.level_locals[level as usize];

            assert_eq!(child_locals.len(), child_targets.len());
            let mut child_locals = Vec::new();
            for child in child_targets.iter() {
                if let Some(child_index_pointer) = self.level_index_pointer[level as usize]
                    .get(child) {
                        let child_local = self.level_locals[level as usize][*child_index_pointer];
                        child_locals.push(child_local);
                    } else {
                        println!("not found {:?}", self.fmm.tree().get_all_keys_set().contains(child))
                    }
            } 
            assert!(parent_locals.len() == child_locals.len() / 8);

            parent_locals
                .into_par_iter()
                .zip(child_locals.par_chunks_exact(nsiblings))
                .for_each(|(parent_local_pointer, child_local_pointers)| {

                    let parent_local = unsafe { rlst_pointer_mat!['a, V, parent_local_pointer.raw, (ncoeffs, 1), (1, ncoeffs)] };
                    
                    for i in 0..8 {
                        let tmp = self.fmm.l2l[i].dot(&parent_local).eval();
                        let child_local = unsafe { std::slice::from_raw_parts_mut(child_local_pointers[i].raw, ncoeffs)};
                        child_local.iter_mut().zip(tmp.data()).for_each(|(c, p)| *c += *p);
                    }
                });

            

        }
    }

    fn m2p<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let dim = self.fmm.kernel.space_dimension();
            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
            
            leaves
                .par_iter()
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers)
                .for_each(|((leaf, charge_index_pointer),potential_send_pointer)| {

                    let targets = &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = targets.len() / dim;
                    let targets = unsafe {
                        rlst_pointer_mat!['a, V, targets.as_ptr(), (ntargets, dim), (dim, 1)]
                    }.eval();

                    if ntargets > 0 {
                        if let Some(w_list) = self.fmm.get_w_list(leaf) {
                            
                            let result = unsafe { std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)};

                            let w_list_indices = w_list
                                .iter()
                                .filter_map(|k| self.fmm.tree().get_leaf_index(k)).collect_vec();
                            // println!("leaf {:?} w list {:?}={:?}", leaf.anchor, w_list.len(), w_list_indices.len());


                            for &source_index in w_list_indices {
                                let multipole_send_ptr = self.leaf_multipoles[source_index];
                                let multipole = unsafe { std::slice::from_raw_parts(multipole_send_ptr.raw, ncoeffs)};
                                let surface = &self.leaf_upward_surfaces[source_index * ncoeffs * dim .. (source_index+1) * ncoeffs * dim];

                                self.fmm.kernel.evaluate_st(
                                    EvalType::Value,
                                    surface, 
                                    targets.data(),
                                    multipole, 
                                    result
                                )
                            }
                        }
                    }
                })
        }
    }

    fn l2p<'a>(&self) {
        if let Some(_leaves) = self.fmm.tree().get_all_leaves() {
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
            let dim = self.fmm.kernel.space_dimension();
            let surface_size = ncoeffs * dim;

            self.leaf_upward_surfaces
                .par_chunks_exact(surface_size)
                .zip(self.leaf_locals.into_par_iter())
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers)
                .for_each(
                    |(
                        ((leaf_downward_equivalent_surface, local_ptr), charge_index_pointer),
                        potential_send_ptr,
                    )| {
                        let target_coordinates = &coordinates
                            [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                        let ntargets = target_coordinates.len() / dim;
                        let target_coordinates = unsafe {
                            rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, dim), (dim, 1)]
                        }.eval();

                        let local_expansion =
                            unsafe { rlst_pointer_mat!['a, V, local_ptr.raw, (ncoeffs, 1), (1, ncoeffs) ]};


                        // Compute direct
                        if ntargets > 0 {
                            let result = unsafe { std::slice::from_raw_parts_mut(potential_send_ptr.raw, ntargets)};

                            self.fmm.kernel.evaluate_st(
                                EvalType::Value,
                                leaf_downward_equivalent_surface,
                                target_coordinates.data(),
                                local_expansion.data(),
                                result,
                            );

                        }
                    },
                );
        }
    }

    // fn p2l<'a>(&self) {

    //     if let Some(leaves) = self.fmm.tree().get_all_keys() { 
    //         let dim = self.fmm.kernel.space_dimension();
    //         let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
    //         let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
    //         let surface_size = ncoeffs * dim;

    //         // leaves
    //         //     .par_iter()
    //         //     .zip(&self.charge_index_pointer)
    //         //     .zip(&self.potentials_send_pointers)
    //         //     .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
    //         //         let targets =
    //         //             &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
    //         //         let ntargets = targets.len() / dim;
    //         //         let targets = unsafe {
    //         //             rlst_pointer_mat!['a, V, targets.as_ptr(), (ntargets, dim), (dim, 1)]
    //         //         }.eval();

    //         //         if ntargets > 0 {

    //         //             if let Some(u_list) = self.fmm.get_x_list(leaf) {

    //         //                 let u_list_indices = u_list
    //         //                     .iter()
    //         //                     .filter_map(|k| self.fmm.tree().get_leaf_index(k));

    //         //                 let charges = u_list_indices
    //         //                     .clone()
    //         //                     .map(|&idx| {
    //         //                         let index_pointer = &self.charge_index_pointer[idx];
    //         //                         &self.charges[index_pointer.0..index_pointer.1]
    //         //                     })
    //         //                     .collect_vec();

    //         //                 let sources_coordinates = u_list_indices
    //         //                     .into_iter()
    //         //                     .map(|&idx| {
    //         //                         let index_pointer = &self.charge_index_pointer[idx];
    //         //                         &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
    //         //                     })
    //         //                     .collect_vec();

    //         //                 for (&charges, sources) in charges.iter().zip(sources_coordinates) {
    //         //                     let nsources = sources.len() / dim;
    //         //                     let sources = unsafe {
    //         //                         rlst_pointer_mat!['a, V, sources.as_ptr(), (nsources, dim), (dim, 1)]
    //         //                     }.eval();


    //         //                     if nsources > 0 {
    //         //                         let result = unsafe { std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)};
    //         //                         self.fmm.kernel.evaluate_st(
    //         //                             EvalType::Value,
    //         //                             sources.data(),
    //         //                             targets.data(),
    //         //                             charges,
    //         //                             result,
    //         //                         );
    //         //                     }
    //         //                 }
    //         //             }
    //         //         }
    //         //     }) 

    //         // leaves
    //         //     .iter()
    //         //     .zip(self.leaf_downward_surfaces.chunks_exact(surface_size))
    //         //     .zip(&self.leaf_locals)
    //         //     .for_each(|((leaf, leaf_downward_surface), local_send_ptr)| {

    //         //         if let Some(x_list) = self.fmm.get_x_list(leaf) {
    //         //             let x_list_indices = x_list
    //         //                 .iter()
    //         //                 .filter_map(|k| self.fmm.tree().get_leaf_index(k));

    //         //             let charges = x_list_indices
    //         //                 .clone()
    //         //                 .map(|&idx| {
    //         //                     let index_pointer = &self.charge_index_pointer[idx];
    //         //                     &self.charges[index_pointer.0..index_pointer.1]
    //         //                 })
    //         //                 .collect_vec();

    //         //             let sources_coordinates = x_list_indices
    //         //                 .into_iter()
    //         //                 .map(|&idx| {
    //         //                     let index_pointer = &self.charge_index_pointer[idx];
    //         //                     &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
    //         //                 })
    //         //                 .collect_vec();

    //         //             let target_local = unsafe { std::slice::from_raw_parts_mut(local_send_ptr.raw, ncoeffs)};

    //         //             for (&charges, sources) in charges.iter().zip(sources_coordinates) {
    //         //                 let nsources = sources.len() / dim;
    //         //                 let sources = unsafe {
    //         //                     rlst_pointer_mat!['a, V, sources.as_ptr(), (nsources, dim), (dim, 1)]
    //         //                 }.eval();

    //         //                 if nsources > 0 {
    //         //                     let mut check_potential = rlst_col_vec![V, ncoeffs];
    //         //                     self.fmm.kernel.evaluate_st(
    //         //                         EvalType::Value, 
    //         //                      sources.data(), 
    //         //                         leaf_downward_surface, 
    //         //                         charges, 
    //         //                         check_potential.data_mut()
    //         //                     );

    //         //                     let scale = self.fmm.kernel().scale(leaf.level());
    //         //                     let mut tmp = self.fmm.dc2e_inv_1.dot(&self.fmm.dc2e_inv_2.dot(&check_potential));
    //         //                     tmp.data_mut().iter_mut().for_each(|val| *val *= scale);
    //         //                     target_local.iter_mut().zip(tmp.data()).for_each(|(r, &t)| *r +=  t );
    //         //                 }
    //         //             }
    //         //         }
    //         //     });

    //     }
    // }


    fn p2p<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let dim = self.fmm.kernel.space_dimension();

            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

            leaves
                .par_iter()
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers)
                .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                    let targets =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = targets.len() / dim;
                    let targets = unsafe {
                        rlst_pointer_mat!['a, V, targets.as_ptr(), (ntargets, dim), (dim, 1)]
                    }.eval();

                    if ntargets > 0 {

                        if let Some(u_list) = self.fmm.get_u_list(leaf) {


                            let u_list_indices = u_list
                                .iter()
                                .filter_map(|k| self.fmm.tree().get_leaf_index(k));

                            let charges = u_list_indices
                                .clone()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &self.charges[index_pointer.0..index_pointer.1]
                                })
                                .collect_vec();

                            let sources_coordinates = u_list_indices
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
                                })
                                .collect_vec();

                            for (&charges, sources) in charges.iter().zip(sources_coordinates) {
                                let nsources = sources.len() / dim;
                                let sources = unsafe {
                                    rlst_pointer_mat!['a, V, sources.as_ptr(), (nsources, dim), (dim, 1)]
                                }.eval();


                                if nsources > 0 {
                                    let result = unsafe { std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)};
                                    self.fmm.kernel.evaluate_st(
                                        EvalType::Value,
                                        sources.data(),
                                        targets.data(),
                                        charges,
                                        result,
                                    );
                                }
                            }
                        }
                    }
                })
        }
    }
}
