//! Multipole field translations
use std::collections::HashSet;

use itertools::Itertools;
use num::Float;
use rayon::prelude::*;

use bempp_traits::{
    field::SourceToTargetData,
    fmm::SourceTranslation,
    kernel::Kernel,
    tree::{FmmTree, Tree},
    types::EvalType,
};
use bempp_tree::types::single_node::SingleNodeTreeNew;
use rlst_blis::interface::gemm::Gemm;

use crate::{
    builder::FmmEvaluationMode,
    constants::{M2M_MAX_CHUNK_SIZE, NSIBLINGS, P2M_MAX_CHUNK_SIZE},
    fmm::KiFmm,
    helpers::find_chunk_size,
};
use bempp_traits::types::Scalar;
use rlst_dense::{
    array::empty_array,
    rlst_array_from_slice2, rlst_dynamic_array2,
    traits::{MultIntoResize, RawAccess, RawAccessMut},
};

impl<T, U, V, W> SourceTranslation for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTreeNew<W>> + Send + Sync,
    U: SourceToTargetData<V> + Send + Sync,
    V: Kernel<T = W> + Send + Sync,
    W: Scalar<Real = W> + Default + Send + Sync + Gemm + Float,
{
    fn p2m(&self) {
        let Some(_leaves) = self.tree.get_source_tree().get_all_leaves() else {
            return;
        };

        let nleaves = self.tree.get_source_tree().get_nleaves().unwrap();
        let dim = self.kernel.space_dimension();
        let surface_size = self.ncoeffs * dim;
        let coordinates = self.tree.get_source_tree().get_all_coordinates().unwrap();
        let ncoordinates = coordinates.len();

        match self.eval_mode {
            FmmEvaluationMode::Vector => {
                let mut check_potentials = rlst_dynamic_array2!(W, [nleaves * self.ncoeffs, 1]);

                // Compute check potential for each box
                check_potentials
                    .data_mut()
                    .par_chunks_exact_mut(self.ncoeffs)
                    .zip(self.leaf_upward_surfaces.par_chunks_exact(surface_size))
                    .zip(&self.charge_index_pointer)
                    .for_each(
                        |((check_potential, upward_check_surface), charge_index_pointer)| {
                            let charges =
                                &self.charges[charge_index_pointer.0..charge_index_pointer.1];
                            let coordinates_row_major = &coordinates
                                [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];

                            let nsources = coordinates_row_major.len() / dim;
                            if nsources > 0 {
                                let coordinates_row_major = rlst_array_from_slice2!(
                                    W,
                                    coordinates_row_major,
                                    [nsources, dim],
                                    [dim, 1]
                                );
                                let mut coordinates_col_major =
                                    rlst_dynamic_array2!(W, [nsources, dim]);
                                coordinates_col_major.fill_from(coordinates_row_major.view());

                                self.kernel.evaluate_st(
                                    EvalType::Value,
                                    coordinates_col_major.data(),
                                    upward_check_surface,
                                    charges,
                                    check_potential,
                                );
                            }
                        },
                    );

                // Use check potentials to compute the multipole expansion
                let chunk_size = find_chunk_size(nleaves, P2M_MAX_CHUNK_SIZE);
                check_potentials
                    .data()
                    .par_chunks_exact(self.ncoeffs * chunk_size)
                    .zip(self.leaf_multipoles.par_chunks_exact(chunk_size))
                    .zip(
                        self.source_scales
                            .par_chunks_exact(self.ncoeffs * chunk_size),
                    )
                    .for_each(|((check_potential, multipole_ptrs), scale)| {
                        let check_potential =
                            rlst_array_from_slice2!(W, check_potential, [self.ncoeffs, chunk_size]);
                        let scale = rlst_array_from_slice2!(W, scale, [self.ncoeffs, chunk_size]);

                        let mut cmp_prod = rlst_dynamic_array2!(W, [self.ncoeffs, chunk_size]);
                        cmp_prod.fill_from(check_potential * scale);

                        let tmp = empty_array::<W, 2>().simple_mult_into_resize(
                            self.uc2e_inv_1.view(),
                            empty_array::<W, 2>()
                                .simple_mult_into_resize(self.uc2e_inv_2.view(), cmp_prod),
                        );

                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(chunk_size)
                        {
                            let multipole = unsafe {
                                std::slice::from_raw_parts_mut(multipole_ptr[0].raw, self.ncoeffs)
                            };
                            multipole
                                .iter_mut()
                                .zip(&tmp.data()[i * self.ncoeffs..(i + 1) * self.ncoeffs])
                                .for_each(|(m, t)| *m += *t);
                        }
                    })
            }

            FmmEvaluationMode::Matrix(nmatvecs) => {
                let mut check_potentials = rlst_dynamic_array2!(W, [nleaves * nmatvecs, 1]);

                // Compute the check potential for each box for each charge vector
                check_potentials
                    .data_mut()
                    .par_chunks_exact_mut(self.ncoeffs * nmatvecs)
                    .zip(self.leaf_upward_surfaces.par_chunks_exact(surface_size))
                    .zip(&self.charge_index_pointer)
                    .for_each(
                        |((check_potential, upward_check_surface), charge_index_pointer)| {
                            let coordinates_row_major = &coordinates
                                [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                            let nsources = coordinates_row_major.len() / dim;

                            if nsources > 0 {
                                for i in 0..nmatvecs {
                                    let charge_vec_displacement = i * ncoordinates;
                                    let charges_i = &self.charges[charge_vec_displacement
                                        + charge_index_pointer.0
                                        ..charge_vec_displacement + charge_index_pointer.1];
                                    let check_potential_i = &mut check_potential
                                        [i * self.ncoeffs..(i + 1) * self.ncoeffs];

                                    let coordinates_mat = rlst_array_from_slice2!(
                                        W,
                                        coordinates_row_major,
                                        [nsources, dim],
                                        [dim, 1]
                                    );
                                    let mut coordinates_col_major =
                                        rlst_dynamic_array2!(W, [nsources, dim]);
                                    coordinates_col_major.fill_from(coordinates_mat.view());

                                    self.kernel.evaluate_st(
                                        EvalType::Value,
                                        coordinates_col_major.data(),
                                        upward_check_surface,
                                        charges_i,
                                        check_potential_i,
                                    );
                                }
                            }
                        },
                    );

                // Compute multipole expansions
                check_potentials
                    .data()
                    .par_chunks_exact(self.ncoeffs * nmatvecs)
                    .zip(self.leaf_multipoles.into_par_iter())
                    .zip(self.source_scales.par_chunks_exact(self.ncoeffs))
                    .for_each(|((check_potential, multipole_ptrs), scale)| {
                        let check_potential =
                            rlst_array_from_slice2!(W, check_potential, [self.ncoeffs, nmatvecs]);

                        let mut scaled_check_potential =
                            rlst_dynamic_array2!(W, [self.ncoeffs, nmatvecs]);

                        scaled_check_potential.fill_from(check_potential);
                        scaled_check_potential.scale_in_place(scale[0]);

                        let tmp = empty_array::<W, 2>().simple_mult_into_resize(
                            self.uc2e_inv_1.view(),
                            empty_array::<W, 2>().simple_mult_into_resize(
                                self.uc2e_inv_2.view(),
                                scaled_check_potential.view(),
                            ),
                        );

                        for (i, multipole_ptr) in multipole_ptrs.iter().enumerate().take(nmatvecs) {
                            let multipole = unsafe {
                                std::slice::from_raw_parts_mut(multipole_ptr.raw, self.ncoeffs)
                            };
                            multipole
                                .iter_mut()
                                .zip(&tmp.data()[i * self.ncoeffs..(i + 1) * self.ncoeffs])
                                .for_each(|(m, t)| *m += *t);
                        }
                    })
            }
        }
    }

    fn m2m(&self, level: u64) {
        let Some(child_sources) = self.tree.get_source_tree().get_keys(level) else {
            return;
        };

        let nchild_sources = self.tree.get_source_tree().get_nkeys(level).unwrap();
        let min = self.tree.get_source_tree().get_node_index(0).unwrap();
        let max = self
            .tree
            .get_source_tree()
            .get_node_index(nchild_sources - 1)
            .unwrap();
        let min_idx = self.tree.get_source_tree().get_index(min).unwrap();
        let max_idx = self.tree.get_source_tree().get_index(max).unwrap();

        let parent_targets: HashSet<_> =
            child_sources.iter().map(|source| source.parent()).collect();

        let mut parent_targets = parent_targets.into_iter().collect_vec();

        parent_targets.sort();
        let nparents = parent_targets.len();

        match self.eval_mode {
            FmmEvaluationMode::Vector => {
                let mut parent_multipoles = Vec::new();
                for parent in parent_targets.iter() {
                    let &parent_index_pointer = self.level_index_pointer_multipoles
                        [(level - 1) as usize]
                        .get(parent)
                        .unwrap();
                    let parent_multipole =
                        &self.level_multipoles[(level - 1) as usize][parent_index_pointer][0];
                    parent_multipoles.push(parent_multipole);
                }

                let child_multipoles =
                    &self.multipoles[min_idx * self.ncoeffs..(max_idx + 1) * self.ncoeffs];

                let mut max_chunk_size = nparents;
                if max_chunk_size > M2M_MAX_CHUNK_SIZE {
                    max_chunk_size = M2M_MAX_CHUNK_SIZE
                }
                let chunk_size = find_chunk_size(nparents, max_chunk_size);

                child_multipoles
                    .par_chunks_exact(NSIBLINGS * self.ncoeffs * chunk_size)
                    .zip(parent_multipoles.par_chunks_exact(chunk_size))
                    .for_each(
                        |(child_multipoles_chunk, parent_multipole_pointers_chunk)| {
                            let child_multipoles_chunk_mat = rlst_array_from_slice2!(
                                W,
                                child_multipoles_chunk,
                                [self.ncoeffs * NSIBLINGS, chunk_size]
                            );

                            let parent_multipoles_chunk = empty_array::<W, 2>()
                                .simple_mult_into_resize(
                                    self.source_data.view(),
                                    child_multipoles_chunk_mat,
                                );

                            for (chunk_idx, parent_multipole_pointer) in
                                parent_multipole_pointers_chunk
                                    .iter()
                                    .enumerate()
                                    .take(chunk_size)
                            {
                                let parent_multipole = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        parent_multipole_pointer.raw,
                                        self.ncoeffs,
                                    )
                                };

                                parent_multipole
                                    .iter_mut()
                                    .zip(
                                        &parent_multipoles_chunk.data()[chunk_idx * self.ncoeffs
                                            ..(chunk_idx + 1) * self.ncoeffs],
                                    )
                                    .for_each(|(p, t)| *p += *t);
                            }
                        },
                    )
            }

            FmmEvaluationMode::Matrix(nmatvecs) => {
                let mut parent_multipoles = vec![Vec::new(); nparents];

                for (parent_idx, parent) in parent_targets.iter().enumerate() {
                    for charge_vec_idx in 0..nmatvecs {
                        let parent_index_pointer = *self.level_index_pointer_multipoles
                            [(level - 1) as usize]
                            .get(parent)
                            .unwrap();
                        let parent_multipole = self.level_multipoles[(level - 1) as usize]
                            [parent_index_pointer][charge_vec_idx];
                        parent_multipoles[parent_idx].push(parent_multipole);
                    }
                }

                let min_key_displacement = min_idx * self.ncoeffs * nmatvecs;
                let max_key_displacement = (max_idx + 1) * self.ncoeffs * nmatvecs;

                let child_multipoles = &self.multipoles[min_key_displacement..max_key_displacement];

                child_multipoles
                    .par_chunks_exact(nmatvecs * self.ncoeffs * NSIBLINGS)
                    .zip(parent_multipoles.into_par_iter())
                    .for_each(|(child_multipoles, parent_multipole_pointers)| {
                        for i in 0..NSIBLINGS {
                            let sibling_displacement = i * self.ncoeffs * nmatvecs;

                            let child_multipoles_i = rlst_array_from_slice2!(
                                W,
                                &child_multipoles[sibling_displacement
                                    ..sibling_displacement + self.ncoeffs * nmatvecs],
                                [self.ncoeffs, nmatvecs]
                            );

                            let result_i = empty_array::<W, 2>().simple_mult_into_resize(
                                self.source_data_vec[i].view(),
                                child_multipoles_i,
                            );

                            for (j, send_ptr) in
                                parent_multipole_pointers.iter().enumerate().take(nmatvecs)
                            {
                                let raw = send_ptr.raw;
                                let parent_multipole_j =
                                    unsafe { std::slice::from_raw_parts_mut(raw, self.ncoeffs) };
                                let result_ij =
                                    &result_i.data()[j * self.ncoeffs..(j + 1) * self.ncoeffs];
                                parent_multipole_j
                                    .iter_mut()
                                    .zip(result_ij.iter())
                                    .for_each(|(p, r)| *p += *r);
                            }
                        }
                    });
            }
        }
    }
}
