//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.

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

use crate::{
    constants::P2M_MAX_CHUNK_SIZE,
    types::{FmmDataLinear, KiFmmLinear},
};

use rlst::{
    common::traits::*,
    dense::{rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer},
};

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
        if let Some(parent_sources) = self.fmm.tree().get_keys(level - 1) {
            if let Some(_child_targets) = self.fmm.tree().get_keys(level) {
                let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

                let nsources = parent_sources.len();
                let min_parent = &parent_sources[0];
                let max_parent = &parent_sources[nsources - 1];

                let min_parent_idx = self.fmm.tree().key_to_index.get(min_parent).unwrap();
                let max_parent_idx = self.fmm.tree().key_to_index.get(max_parent).unwrap();

                let parent_locals =
                    &self.locals[min_parent_idx * ncoeffs..(max_parent_idx + 1) * ncoeffs];

                let child_locals = &self.level_locals[level as usize];

                let mut max_chunk_size = 8_i32.pow((level).try_into().unwrap()) as usize;

                if max_chunk_size > P2M_MAX_CHUNK_SIZE {
                    max_chunk_size = P2M_MAX_CHUNK_SIZE;
                }
                let chunk_size = find_chunk_size(nsources, max_chunk_size);
                let nsiblings = 8;

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

    fn p2l<'a>(&self) {}

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
