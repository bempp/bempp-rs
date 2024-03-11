use std::collections::HashSet;

use bempp_field::constants::NSIBLINGS;
use itertools::Itertools;
use num::Float;
use rayon::prelude::*;

use bempp_traits::{
    field::SourceToTargetData,
    fmm::TargetTranslation,
    kernel::Kernel,
    tree::{FmmTree, Tree},
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};
use rlst_dense::{
    array::empty_array,
    rlst_array_from_slice2, rlst_dynamic_array2,
    traits::{MultIntoResize, RawAccess, RawAccessMut},
    types::RlstScalar,
};

use crate::{
    builder::FmmEvalType, constants::L2L_MAX_CHUNK_SIZE, fmm::KiFmm, helpers::find_chunk_size,
};

impl<T, U, V, W> TargetTranslation for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTree<W>, NodeIndex = MortonKey> + Send + Sync,
    U: SourceToTargetData<V> + Send + Sync,
    V: Kernel<T = W>,
    W: RlstScalar<Real = W> + Float + Default,
{
    fn l2l(&self, level: u64) {
        let Some(child_targets) = self.tree.get_target_tree().get_keys(level) else {
            return;
        };

        let parent_sources: HashSet<MortonKey> =
            child_targets.iter().map(|source| source.parent()).collect();
        let mut parent_sources = parent_sources.into_iter().collect_vec();
        parent_sources.sort();
        let nparents = parent_sources.len();
        let mut parent_locals = Vec::new();
        for parent in parent_sources.iter() {
            let parent_index_pointer = *self.level_index_pointer_locals[(level - 1) as usize]
                .get(parent)
                .unwrap();
            let parent_local = &self.level_locals[(level - 1) as usize][parent_index_pointer][0];
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
            .zip(child_locals.par_chunks_exact(NSIBLINGS * chunk_size))
            .for_each(|(parent_local_pointer_chunk, child_local_pointers_chunk)| {
                let mut parent_locals = rlst_dynamic_array2!(W, [self.ncoeffs, chunk_size]);
                for (chunk_idx, parent_local_pointer) in parent_local_pointer_chunk
                    .iter()
                    .enumerate()
                    .take(chunk_size)
                {
                    parent_locals.data_mut()
                        [chunk_idx * self.ncoeffs..(chunk_idx + 1) * self.ncoeffs]
                        .copy_from_slice(unsafe {
                            std::slice::from_raw_parts_mut(parent_local_pointer.raw, self.ncoeffs)
                        });
                }

                for i in 0..NSIBLINGS {
                    let tmp = empty_array::<W, 2>()
                        .simple_mult_into_resize(self.target_data[i].view(), parent_locals.view());

                    for j in 0..chunk_size {
                        let chunk_displacement = j * NSIBLINGS;
                        let child_displacement = chunk_displacement + i;
                        let child_local = unsafe {
                            std::slice::from_raw_parts_mut(
                                child_local_pointers_chunk[child_displacement][0].raw,
                                self.ncoeffs,
                            )
                        };
                        child_local
                            .iter_mut()
                            .zip(&tmp.data()[j * self.ncoeffs..(j + 1) * self.ncoeffs])
                            .for_each(|(l, t)| *l += *t);
                    }
                }
            });
    }

    fn l2p(&self) {
        let Some(_leaves) = self.tree.get_target_tree().get_all_leaves() else {
            return;
        };

        let coordinates = self.tree.get_target_tree().get_all_coordinates().unwrap();
        let surface_size = self.ncoeffs * self.dim;

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                self.leaf_upward_surfaces_targets
                    .par_chunks_exact(surface_size)
                    .zip(self.leaf_locals.into_par_iter())
                    .zip(&self.charge_index_pointer_targets)
                    .zip(&self.potentials_send_pointers)
                    .for_each(
                        |(
                            ((leaf_downward_equivalent_surface, leaf_locals), charge_index_pointer),
                            potential_send_ptr,
                        )| {
                            let target_coordinates_row_major = &coordinates[charge_index_pointer.0
                                * self.dim
                                ..charge_index_pointer.1 * self.dim];
                            let ntargets = target_coordinates_row_major.len() / self.dim;

                            // Compute direct
                            if ntargets > 0 {
                                let target_coordinates_row_major = rlst_array_from_slice2!(
                                    W,
                                    target_coordinates_row_major,
                                    [ntargets, self.dim],
                                    [self.dim, 1]
                                );
                                let mut target_coordinates_col_major =
                                    rlst_dynamic_array2!(W, [ntargets, self.dim]);
                                target_coordinates_col_major
                                    .fill_from(target_coordinates_row_major.view());

                                let result = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        potential_send_ptr.raw,
                                        ntargets * self.eval_size,
                                    )
                                };

                                self.kernel.evaluate_st(
                                    self.kernel_eval_type,
                                    leaf_downward_equivalent_surface,
                                    target_coordinates_col_major.data(),
                                    unsafe {
                                        std::slice::from_raw_parts_mut(
                                            leaf_locals[0].raw,
                                            self.ncoeffs,
                                        )
                                    },
                                    result,
                                );
                            }
                        },
                    );
            }

            FmmEvalType::Matrix(nmatvec) => {
                let nleaves = self.tree.get_target_tree().get_nleaves().unwrap();
                for i in 0..nmatvec {
                    self.leaf_upward_surfaces_targets
                        .par_chunks_exact(surface_size)
                        .zip(&self.leaf_locals)
                        .zip(&self.charge_index_pointer_targets)
                        .zip(&self.potentials_send_pointers[i * nleaves..(i + 1) * nleaves])
                        .for_each(
                            |(
                                (
                                    (leaf_downward_equivalent_surface, leaf_locals),
                                    charge_index_pointer,
                                ),
                                potential_send_ptr,
                            )| {
                                let target_coordinates_row_major =
                                    &coordinates[charge_index_pointer.0 * self.dim
                                        ..charge_index_pointer.1 * self.dim];
                                let ntargets = target_coordinates_row_major.len() / self.dim;

                                if ntargets > 0 {
                                    let target_coordinates_row_major = rlst_array_from_slice2!(
                                        W,
                                        target_coordinates_row_major,
                                        [ntargets, self.dim],
                                        [self.dim, 1]
                                    );
                                    let mut target_coordinates_col_major =
                                        rlst_dynamic_array2!(W, [ntargets, self.dim]);
                                    target_coordinates_col_major
                                        .fill_from(target_coordinates_row_major.view());

                                    let local_expansion_ptr = leaf_locals[i].raw;
                                    let local_expansion = unsafe {
                                        std::slice::from_raw_parts(
                                            local_expansion_ptr,
                                            self.ncoeffs,
                                        )
                                    };
                                    let result = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            potential_send_ptr.raw,
                                            ntargets * self.eval_size,
                                        )
                                    };

                                    self.kernel.evaluate_st(
                                        self.kernel_eval_type,
                                        leaf_downward_equivalent_surface,
                                        target_coordinates_col_major.data(),
                                        local_expansion,
                                        result,
                                    );
                                }
                            },
                        )
                }
            }
        }
    }

    fn m2p(&self) {}

    fn p2p(&self) {
        let Some(leaves) = self.tree.get_target_tree().get_all_leaves() else {
            return;
        };

        let all_target_coordinates = self.tree.get_target_tree().get_all_coordinates().unwrap();
        let all_source_coordinates = self.tree.get_source_tree().get_all_coordinates().unwrap();
        let n_all_source_coordinates = all_source_coordinates.len() / self.dim;

        match self.fmm_eval_type {
            FmmEvalType::Vector => leaves
                .par_iter()
                .zip(&self.charge_index_pointer_targets)
                .zip(&self.potentials_send_pointers)
                .for_each(
                    |((leaf, charge_index_pointer_targets), potential_send_pointer)| {
                        let target_coordinates_row_major = &all_target_coordinates
                            [charge_index_pointer_targets.0 * self.dim
                                ..charge_index_pointer_targets.1 * self.dim];
                        let ntargets = target_coordinates_row_major.len() / self.dim;

                        if ntargets > 0 {
                            let target_coordinates_row_major = rlst_array_from_slice2!(
                                W,
                                target_coordinates_row_major,
                                [ntargets, self.dim],
                                [self.dim, 1]
                            );
                            let mut target_coordinates_col_major =
                                rlst_dynamic_array2!(W, [ntargets, self.dim]);
                            target_coordinates_col_major
                                .fill_from(target_coordinates_row_major.view());

                            if let Some(u_list) = self.tree.get_near_field(leaf) {
                                let u_list_indices = u_list
                                    .iter()
                                    .filter_map(|k| self.tree.get_source_tree().get_leaf_index(k));

                                let charges = u_list_indices
                                    .clone()
                                    .map(|&idx| {
                                        let index_pointer = &self.charge_index_pointer_sources[idx];
                                        &self.charges[index_pointer.0..index_pointer.1]
                                    })
                                    .collect_vec();

                                let sources_coordinates = u_list_indices
                                    .into_iter()
                                    .map(|&idx| {
                                        let index_pointer = &self.charge_index_pointer_sources[idx];
                                        &all_source_coordinates
                                            [index_pointer.0 * self.dim..index_pointer.1 * self.dim]
                                    })
                                    .collect_vec();

                                for (&charges, source_coordinates_row_major) in
                                    charges.iter().zip(sources_coordinates)
                                {
                                    let nsources = source_coordinates_row_major.len() / self.dim;

                                    if nsources > 0 {
                                        let source_coordinates_row_major = rlst_array_from_slice2!(
                                            W,
                                            source_coordinates_row_major,
                                            [nsources, self.dim],
                                            [self.dim, 1]
                                        );
                                        let mut source_coordinates_col_major =
                                            rlst_dynamic_array2!(W, [nsources, self.dim]);
                                        source_coordinates_col_major
                                            .fill_from(source_coordinates_row_major.view());

                                        let result = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                potential_send_pointer.raw,
                                                ntargets * self.eval_size,
                                            )
                                        };

                                        self.kernel.evaluate_st(
                                            self.kernel_eval_type,
                                            source_coordinates_col_major.data(),
                                            target_coordinates_col_major.data(),
                                            charges,
                                            result,
                                        )
                                    }
                                }
                            }
                        }
                    },
                ),

            FmmEvalType::Matrix(nmatvec) => {
                let nleaves = self.tree.get_target_tree().get_nleaves().unwrap();

                for i in 0..nmatvec {
                    leaves
                        .par_iter()
                        .zip(&self.charge_index_pointer_targets)
                        .zip(&self.potentials_send_pointers[i * nleaves..(i + 1) * nleaves])
                        .for_each(|((leaf, charge_index_pointer), potential_send_ptr)| {
                            let target_coordinates_row_major = &all_target_coordinates
                                [charge_index_pointer.0 * self.dim
                                    ..charge_index_pointer.1 * self.dim];
                            let ntargets = target_coordinates_row_major.len() / self.dim;

                            if ntargets > 0 {
                                let target_coordinates_row_major = rlst_array_from_slice2!(
                                    W,
                                    target_coordinates_row_major,
                                    [ntargets, self.dim],
                                    [self.dim, 1]
                                );
                                let mut target_coordinates_col_major =
                                    rlst_dynamic_array2!(W, [ntargets, self.dim]);
                                target_coordinates_col_major
                                    .fill_from(target_coordinates_row_major.view());

                                if let Some(u_list) = self.tree.get_near_field(leaf) {
                                    let u_list_indices = u_list.iter().filter_map(|k| {
                                        self.tree.get_source_tree().get_leaf_index(k)
                                    });

                                    let charge_vec_displacement = i * n_all_source_coordinates;
                                    let charges = u_list_indices
                                        .clone()
                                        .map(|&idx| {
                                            let index_pointer =
                                                &self.charge_index_pointer_sources[idx];
                                            &self.charges[charge_vec_displacement + index_pointer.0
                                                ..charge_vec_displacement + index_pointer.1]
                                        })
                                        .collect_vec();

                                    let sources_coordinates = u_list_indices
                                        .into_iter()
                                        .map(|&idx| {
                                            let index_pointer =
                                                &self.charge_index_pointer_sources[idx];
                                            &all_source_coordinates[index_pointer.0 * self.dim
                                                ..index_pointer.1 * self.dim]
                                        })
                                        .collect_vec();

                                    for (&charges, source_coordinates_row_major) in
                                        charges.iter().zip(sources_coordinates)
                                    {
                                        let nsources =
                                            source_coordinates_row_major.len() / self.dim;
                                        let source_coordinates_row_major = rlst_array_from_slice2!(
                                            W,
                                            source_coordinates_row_major,
                                            [nsources, self.dim],
                                            [self.dim, 1]
                                        );
                                        let mut source_coordinates_col_major =
                                            rlst_dynamic_array2!(W, [nsources, self.dim]);
                                        source_coordinates_col_major
                                            .fill_from(source_coordinates_row_major.view());

                                        if nsources > 0 {
                                            let result = unsafe {
                                                std::slice::from_raw_parts_mut(
                                                    potential_send_ptr.raw,
                                                    ntargets * self.eval_size,
                                                )
                                            };

                                            self.kernel.evaluate_st(
                                                self.kernel_eval_type,
                                                source_coordinates_col_major.data(),
                                                target_coordinates_col_major.data(),
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
    }
}
