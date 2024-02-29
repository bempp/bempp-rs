//! Local field translations for uniform and adaptive Kernel Indepenent FMMs
use std::collections::HashSet;

use itertools::Itertools;
use num::Float;
use rayon::prelude::*;

use bempp_field::field::ncoeffs;
use bempp_traits::{
    field::SourceToTargetData,
    fmm::{Fmm, InteractionLists, TargetTranslation},
    kernel::{Kernel, ScaleInvariantHomogenousKernel},
    tree::Tree,
    types::{EvalType, Scalar},
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::{
    constants::L2L_MAX_CHUNK_SIZE,
    helpers::find_chunk_size,
    types::{FmmDataAdaptive, FmmDataUniform, FmmDataUniformMatrix, KiFmm, KiFmmMatrix},
};

use rlst_dense::{
    array::empty_array,
    rlst_array_from_slice2, rlst_dynamic_array2,
    traits::{MultIntoResize, RawAccess, RawAccessMut},
};

impl<T, U, V> TargetTranslation for FmmDataUniform<KiFmm<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V>
        + ScaleInvariantHomogenousKernel<T = V>
        + std::marker::Send
        + std::marker::Sync,
    U: SourceToTargetData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V>
        + Float
        + Default
        + std::marker::Sync
        + std::marker::Send
        + rlst_blis::interface::gemm::Gemm,
{
    fn l2l<'a>(&self, level: u64) {
        let Some(child_targets) = self.fmm.tree().get_keys(level) else {
            return;
        };

        let nsiblings = 8;
        let ncoeffs = ncoeffs(self.fmm.order);

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
            .zip(child_locals.par_chunks_exact(nsiblings * chunk_size))
            .for_each(|(parent_local_pointer_chunk, child_local_pointers_chunk)| {
                let mut parent_locals = rlst_dynamic_array2!(V, [ncoeffs, chunk_size]);
                for (chunk_idx, parent_local_pointer) in parent_local_pointer_chunk
                    .iter()
                    .enumerate()
                    .take(chunk_size)
                {
                    parent_locals.data_mut()[chunk_idx * ncoeffs..(chunk_idx + 1) * ncoeffs]
                        .copy_from_slice(unsafe {
                            std::slice::from_raw_parts_mut(parent_local_pointer.raw, ncoeffs)
                        });
                }

                for i in 0..nsiblings {
                    let tmp = empty_array::<V, 2>()
                        .simple_mult_into_resize(self.fmm.l2l[i].view(), parent_locals.view());

                    for j in 0..chunk_size {
                        let chunk_displacement = j * nsiblings;
                        let child_displacement = chunk_displacement + i;
                        let child_local = unsafe {
                            std::slice::from_raw_parts_mut(
                                child_local_pointers_chunk[child_displacement].raw,
                                ncoeffs,
                            )
                        };
                        child_local
                            .iter_mut()
                            .zip(&tmp.data()[j * ncoeffs..(j + 1) * ncoeffs])
                            .for_each(|(l, t)| *l += *t);
                    }
                }
            });
    }

    fn m2p<'a>(&self) {}

    fn l2p<'a>(&self) {
        let Some(_leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let ncoeffs = ncoeffs(self.fmm.order);

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
                    let target_coordinates_row_major =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = target_coordinates_row_major.len() / dim;

                    // Compute direct
                    if ntargets > 0 {
                        let target_coordinates_row_major = rlst_array_from_slice2!(
                            V,
                            target_coordinates_row_major,
                            [ntargets, dim],
                            [dim, 1]
                        );
                        let mut target_coordinates_col_major =
                            rlst_dynamic_array2!(V, [ntargets, dim]);
                        target_coordinates_col_major.fill_from(target_coordinates_row_major.view());

                        let result = unsafe {
                            std::slice::from_raw_parts_mut(potential_send_ptr.raw, ntargets)
                        };

                        self.fmm.kernel.evaluate_st(
                            EvalType::Value,
                            leaf_downward_equivalent_surface,
                            target_coordinates_col_major.data(),
                            unsafe { std::slice::from_raw_parts_mut(local_ptr.raw, ncoeffs) },
                            result,
                        );
                    }
                },
            );
    }

    fn p2p<'a>(&self) {
        let Some(leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let dim = self.fmm.kernel.space_dimension();

        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

        leaves
            .par_iter()
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers)
            .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                let target_coordinates_row_major =
                    &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                let ntargets = target_coordinates_row_major.len() / dim;

                if ntargets > 0 {
                    let target_coordinates_row_major = rlst_array_from_slice2!(
                        V,
                        target_coordinates_row_major,
                        [ntargets, dim],
                        [dim, 1]
                    );
                    let mut target_coordinates_col_major = rlst_dynamic_array2!(V, [ntargets, dim]);
                    target_coordinates_col_major.fill_from(target_coordinates_row_major.view());

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

                        for (&charges, source_coordinates_row_major) in
                            charges.iter().zip(sources_coordinates)
                        {
                            let nsources = source_coordinates_row_major.len() / dim;

                            if nsources > 0 {
                                let source_coordinates_row_major = rlst_array_from_slice2!(
                                    V,
                                    source_coordinates_row_major,
                                    [nsources, dim],
                                    [dim, 1]
                                );
                                let mut source_coordinates_col_major =
                                    rlst_dynamic_array2!(V, [nsources, dim]);
                                source_coordinates_col_major
                                    .fill_from(source_coordinates_row_major.view());

                                let result = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        potential_send_pointer.raw,
                                        ntargets,
                                    )
                                };
                                self.fmm.kernel.evaluate_st(
                                    EvalType::Value,
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

impl<T, U, V> TargetTranslation for FmmDataAdaptive<KiFmm<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V>
        + ScaleInvariantHomogenousKernel<T = V>
        + std::marker::Send
        + std::marker::Sync,
    U: SourceToTargetData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V>
        + Float
        + Default
        + std::marker::Sync
        + std::marker::Send
        + rlst_blis::interface::gemm::Gemm,
{
    fn l2l<'a>(&self, level: u64) {
        let Some(child_targets) = self.fmm.tree().get_keys(level) else {
            return;
        };

        let nsiblings = 8;
        let ncoeffs = ncoeffs(self.fmm.order);

        let mut child_targets = child_targets.iter().cloned().collect_vec();
        child_targets.sort();

        let parent_sources: HashSet<MortonKey> =
            child_targets.iter().map(|source| source.parent()).collect();
        let mut parent_sources = parent_sources.into_iter().collect_vec();
        parent_sources.sort();

        let mut parent_locals = Vec::new();
        for parent in parent_sources.iter() {
            let parent_index_pointer = *self.level_index_pointer[(level - 1) as usize]
                .get(parent)
                .unwrap();
            let parent_local = self.level_locals[(level - 1) as usize][parent_index_pointer];
            parent_locals.push(parent_local);
        }

        let mut child_locals = Vec::new();
        for child in child_targets.iter() {
            let child_index_pointer = self.level_index_pointer[level as usize].get(child).unwrap();
            let child_local = self.level_locals[level as usize][*child_index_pointer];
            child_locals.push(child_local);
        }

        parent_locals
            .into_par_iter()
            .zip(child_locals.par_chunks_exact(nsiblings))
            .for_each(|(parent_local_pointer, child_local_pointers)| {
                let parent_local = rlst_array_from_slice2!(
                    V,
                    unsafe { std::slice::from_raw_parts(parent_local_pointer.raw, ncoeffs) },
                    [ncoeffs, 1]
                );

                for (i, child_local_pointer) in child_local_pointers.iter().enumerate().take(8) {
                    let tmp = empty_array::<V, 2>()
                        .simple_mult_into_resize(self.fmm.l2l[i].view(), parent_local.view());
                    let child_local =
                        unsafe { std::slice::from_raw_parts_mut(child_local_pointer.raw, ncoeffs) };
                    child_local
                        .iter_mut()
                        .zip(tmp.data())
                        .for_each(|(c, p)| *c += *p);
                }
            });
    }

    fn m2p<'a>(&self) {
        let Some(leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let dim = self.fmm.kernel.space_dimension();
        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

        let ncoeffs = ncoeffs(self.fmm.order);

        leaves
            .par_iter()
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers)
            .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                let target_coordinates_row_major =
                    &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                let ntargets = target_coordinates_row_major.len() / dim;

                if ntargets > 0 {
                    let target_coordinates_row_major = rlst_array_from_slice2!(
                        V,
                        target_coordinates_row_major,
                        [ntargets, dim],
                        [dim, 1]
                    );
                    let mut target_coordinates_col_major = rlst_dynamic_array2!(V, [ntargets, dim]);
                    target_coordinates_col_major.fill_from(target_coordinates_row_major.view());

                    if let Some(w_list) = self.fmm.get_w_list(leaf) {
                        let result = unsafe {
                            std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)
                        };

                        let w_list_indices = w_list
                            .iter()
                            .filter_map(|k| self.fmm.tree().get_leaf_index(k))
                            .collect_vec();

                        for &source_index in w_list_indices {
                            let multipole_send_ptr = self.leaf_multipoles[source_index];
                            let multipole = unsafe {
                                std::slice::from_raw_parts(multipole_send_ptr.raw, ncoeffs)
                            };
                            let surface = &self.leaf_upward_surfaces
                                [source_index * ncoeffs * dim..(source_index + 1) * ncoeffs * dim];

                            self.fmm.kernel.evaluate_st(
                                EvalType::Value,
                                surface,
                                target_coordinates_col_major.data(),
                                multipole,
                                result,
                            )
                        }
                    }
                }
            });
    }

    fn l2p<'a>(&self) {
        let Some(_leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let ncoeffs = ncoeffs(self.fmm.order);

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
                    let target_coordinates_row_major =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = target_coordinates_row_major.len() / dim;

                    // Compute direct
                    if ntargets > 0 {
                        let target_coordinates_row_major = rlst_array_from_slice2!(
                            V,
                            target_coordinates_row_major,
                            [ntargets, dim],
                            [dim, 1]
                        );
                        let mut target_coordinates_col_major =
                            rlst_dynamic_array2!(V, [ntargets, dim]);
                        target_coordinates_col_major.fill_from(target_coordinates_row_major.view());

                        let result = unsafe {
                            std::slice::from_raw_parts_mut(potential_send_ptr.raw, ntargets)
                        };

                        self.fmm.kernel.evaluate_st(
                            EvalType::Value,
                            leaf_downward_equivalent_surface,
                            target_coordinates_col_major.data(),
                            unsafe { std::slice::from_raw_parts_mut(local_ptr.raw, ncoeffs) },
                            result,
                        );
                    }
                },
            );
    }

    fn p2p<'a>(&self) {
        let Some(leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let dim = self.fmm.kernel.space_dimension();
        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
        leaves
            .par_iter()
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers)
            .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                let target_coordinates_row_major =
                    &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                let ntargets = target_coordinates_row_major.len() / dim;

                if ntargets > 0 {
                    let target_coordinates_row_major = rlst_array_from_slice2!(
                        V,
                        target_coordinates_row_major,
                        [ntargets, dim],
                        [dim, 1]
                    );
                    let mut target_coordinates_col_major = rlst_dynamic_array2!(V, [ntargets, dim]);
                    target_coordinates_col_major.fill_from(target_coordinates_row_major.view());

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

                        for (&charges, source_coordinates_row_major) in
                            charges.iter().zip(sources_coordinates)
                        {
                            let nsources = source_coordinates_row_major.len() / dim;

                            if nsources > 0 {
                                let source_coordinates_row_major = rlst_array_from_slice2!(
                                    V,
                                    source_coordinates_row_major,
                                    [nsources, dim],
                                    [dim, 1]
                                );
                                let mut source_coordinates_col_major =
                                    rlst_dynamic_array2!(V, [nsources, dim]);
                                source_coordinates_col_major
                                    .fill_from(source_coordinates_row_major.view());

                                let result = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        potential_send_pointer.raw,
                                        ntargets,
                                    )
                                };
                                self.fmm.kernel.evaluate_st(
                                    EvalType::Value,
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

impl<T, U, V> TargetTranslation for FmmDataUniformMatrix<KiFmmMatrix<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V>
        + ScaleInvariantHomogenousKernel<T = V>
        + std::marker::Send
        + std::marker::Sync,
    U: SourceToTargetData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V>
        + Float
        + Default
        + std::marker::Sync
        + std::marker::Send
        + rlst_blis::interface::gemm::Gemm,
{
    fn l2l<'a>(&self, level: u64) {
        let Some(child_targets) = self.fmm.tree().get_keys(level) else {
            return;
        };

        let nsiblings = 8;

        let parent_sources: HashSet<MortonKey> =
            child_targets.iter().map(|source| source.parent()).collect();
        let mut parent_sources = parent_sources.into_iter().collect_vec();
        parent_sources.sort();
        let nparents = parent_sources.len();
        let mut parent_locals = vec![Vec::new(); nparents];
        for (parent_idx, parent) in parent_sources.iter().enumerate() {
            for charge_vec_idx in 0..self.ncharge_vectors {
                let parent_index_pointer = *self.level_index_pointer[(level - 1) as usize]
                    .get(parent)
                    .unwrap();
                let parent_local =
                    self.level_locals[(level - 1) as usize][parent_index_pointer][charge_vec_idx];
                parent_locals[parent_idx].push(parent_local);
            }
        }

        let child_locals = &self.level_locals[level as usize];

        parent_locals
            .into_par_iter()
            .zip(child_locals.par_chunks_exact(nsiblings))
            .for_each(|(parent_local_pointers, child_locals_pointers)| {
                let mut parent_locals =
                    rlst_dynamic_array2!(V, [self.ncoeffs, self.ncharge_vectors]);

                for (charge_vec_idx, parent_local_pointer) in parent_local_pointers
                    .iter()
                    .enumerate()
                    .take(self.ncharge_vectors)
                {
                    let tmp = unsafe {
                        std::slice::from_raw_parts(parent_local_pointer.raw, self.ncoeffs)
                    };
                    parent_locals.data_mut()
                        [charge_vec_idx * self.ncoeffs..(charge_vec_idx + 1) * self.ncoeffs]
                        .copy_from_slice(tmp);
                }

                for (i, child_locals_i) in child_locals_pointers.iter().enumerate().take(nsiblings)
                {
                    let result_i = empty_array::<V, 2>()
                        .simple_mult_into_resize(self.fmm.l2l[i].view(), parent_locals.view());

                    for (j, child_locals_ij) in
                        child_locals_i.iter().enumerate().take(self.ncharge_vectors)
                    {
                        let child_locals_ij = unsafe {
                            std::slice::from_raw_parts_mut(child_locals_ij.raw, self.ncoeffs)
                        };
                        let result_ij = &result_i.data()[j * self.ncoeffs..(j + 1) * self.ncoeffs];
                        child_locals_ij
                            .iter_mut()
                            .zip(result_ij.iter())
                            .for_each(|(c, r)| *c += *r);
                    }
                }
            });
    }

    fn m2p<'a>(&self) {}

    fn l2p<'a>(&self) {
        let Some(_leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let surface_size = self.ncoeffs * self.fmm.kernel.space_dimension();
        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
        let dim = self.fmm.kernel.space_dimension();

        for i in 0..self.ncharge_vectors {
            self.leaf_upward_surfaces
                .par_chunks_exact(surface_size)
                .zip(&self.leaf_locals)
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers[i * self.nleaves..(i + 1) * self.nleaves])
                .for_each(
                    |(
                        ((leaf_downward_equivalent_surface, leaf_locals), charge_index_pointer),
                        potential_send_pointer,
                    )| {
                        let target_coordinates_row_major = &coordinates
                            [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                        let ntargets = target_coordinates_row_major.len() / dim;

                        if ntargets > 0 {
                            let target_coordinates_row_major = rlst_array_from_slice2!(
                                V,
                                target_coordinates_row_major,
                                [ntargets, dim],
                                [dim, 1]
                            );
                            let mut target_coordinates_col_major =
                                rlst_dynamic_array2!(V, [ntargets, dim]);
                            target_coordinates_col_major
                                .fill_from(target_coordinates_row_major.view());

                            let local_expansion_ptr = leaf_locals[i].raw;
                            let local_expansion = unsafe {
                                std::slice::from_raw_parts(local_expansion_ptr, self.ncoeffs)
                            };
                            let result = unsafe {
                                std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)
                            };

                            self.fmm.kernel().evaluate_st(
                                EvalType::Value,
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

    fn p2p<'a>(&self) {
        let Some(leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let dim = self.fmm.kernel.space_dimension();
        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
        let ncoordinates = coordinates.len() / dim;

        for i in 0..self.ncharge_vectors {
            leaves
                .par_iter()
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers[i * self.nleaves..(i + 1) * self.nleaves])
                .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                    let target_coordinates_row_major =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = target_coordinates_row_major.len() / dim;

                    if ntargets > 0 {
                        let target_coordinates_row_major = rlst_array_from_slice2!(
                            V,
                            target_coordinates_row_major,
                            [ntargets, dim],
                            [dim, 1]
                        );
                        let mut target_coordinates_col_major =
                            rlst_dynamic_array2!(V, [ntargets, dim]);
                        target_coordinates_col_major.fill_from(target_coordinates_row_major.view());

                        if let Some(u_list) = self.fmm.get_u_list(leaf) {
                            let u_list_indices = u_list
                                .iter()
                                .filter_map(|k| self.fmm.tree().get_leaf_index(k));

                            let charge_vec_displacement = i * ncoordinates;
                            let charges = u_list_indices
                                .clone()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &self.charges[charge_vec_displacement + index_pointer.0
                                        ..charge_vec_displacement + index_pointer.1]
                                })
                                .collect_vec();

                            let sources_coordinates = u_list_indices
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
                                })
                                .collect_vec();

                            for (&charges, source_coordinates_row_major) in
                                charges.iter().zip(sources_coordinates)
                            {
                                let nsources = source_coordinates_row_major.len() / dim;
                                let source_coordinates_row_major = rlst_array_from_slice2!(
                                    V,
                                    source_coordinates_row_major,
                                    [nsources, dim],
                                    [dim, 1]
                                );
                                let mut source_coordinates_col_major =
                                    rlst_dynamic_array2!(V, [nsources, dim]);
                                source_coordinates_col_major
                                    .fill_from(source_coordinates_row_major.view());

                                if nsources > 0 {
                                    let result = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            potential_send_pointer.raw,
                                            ntargets,
                                        )
                                    };
                                    self.fmm.kernel.evaluate_st(
                                        EvalType::Value,
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
