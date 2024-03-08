//! Multipole to Local field translations for uniform and adaptive Kernel Indepenent FMMs
use bempp_field::types::BlasFieldTranslationKiFmm;
use bempp_traits::fmm::Fmm;
use bempp_traits::tree::FmmTree;
use itertools::Itertools;
use num::Float;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use bempp_field::helpers::ncoeffs_kifmm;
use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData},
    kernel::Kernel,
    tree::Tree,
    types::Scalar,
};
use bempp_tree::types::single_node::SingleNodeTreeNew;

use crate::builder::FmmEvalType;
use crate::field_translation::target;
use crate::fmm::KiFmm;
use crate::traits::FmmScalar;

use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_array_from_slice2, rlst_dynamic_array2,
    traits::{MatrixSvd, MultIntoResize, RawAccess, RawAccessMut},
};

impl<T, U, V> KiFmm<V, BlasFieldTranslationKiFmm<U, T>, T, U>
where
    T: Kernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: FmmScalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree<Tree = SingleNodeTreeNew<U>>,
{
    fn displacements(&self, level: u64) -> Vec<Mutex<Vec<i64>>> {
        let sources = self.tree.get_source_tree().get_keys(level).unwrap();
        let nsources = sources.len();

        let targets = self.tree.get_target_tree().get_keys(level).unwrap();
        let ntargets = targets.len();

        let all_displacements = vec![vec![-1i64; ntargets]; 316];
        let all_displacements = all_displacements.into_iter().map(Mutex::new).collect_vec();

        targets.into_par_iter().enumerate().for_each(|(j, target)| {
            let v_list = target
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| {
                    !target.is_adjacent(pnc)
                        && self
                            .tree
                            .get_source_tree()
                            .get_all_keys_set()
                            .unwrap()
                            .contains(pnc)
                })
                .collect_vec();

            let transfer_vectors = v_list
                .iter()
                .map(|source| source.find_transfer_vector(target))
                .collect_vec();

            let mut transfer_vectors_map = HashMap::new();
            for (i, v) in transfer_vectors.iter().enumerate() {
                transfer_vectors_map.insert(v, i);
            }

            let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().cloned().collect();

            for (i, tv) in self
                .source_to_target_data
                .transfer_vectors
                .iter()
                .enumerate()
            {
                let mut all_displacements_lock = all_displacements[i].lock().unwrap();

                if transfer_vectors_set.contains(&tv.hash) {
                    let target = &v_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                    let target_index = self.level_index_pointer[level as usize]
                        .get(target)
                        .unwrap();
                    all_displacements_lock[j] = *target_index as i64;
                }
            }
        });

        all_displacements
    }
}

/// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
impl<T, U, V> SourceToTarget for KiFmm<V, BlasFieldTranslationKiFmm<U, T>, T, U>
where
    T: Kernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: FmmScalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree<Tree = SingleNodeTreeNew<U>>,
{
    fn m2l(&self, level: u64) {
        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let Some(sources) = self.tree.get_source_tree().get_keys(level) else {
                    return;
                };
                let Some(targets) = self.tree.get_target_tree().get_keys(level) else {
                    return;
                };

                // Number of sources at this level
                let nsources = sources.len();

                // Lookup multipole data from source tree
                let multipoles = rlst_array_from_slice2!(
                    U,
                    unsafe {
                        std::slice::from_raw_parts(
                            self.level_multipoles[level as usize][0][0].raw,
                            self.ncoeffs * nsources,
                        )
                    },
                    [self.ncoeffs, nsources]
                );

                // Compute SVD compressed multipole expansions at this level
                rlst_blis::interface::threading::enable_threading();
                let mut compressed_multipoles = empty_array::<U, 2>().simple_mult_into_resize(
                    self.source_to_target_data.operator_data.st_block.view(),
                    multipoles,
                );
                rlst_blis::interface::threading::disable_threading();
            }
            FmmEvalType::Matrix(nmatvec) => {}
        }
    }

    fn p2l(&self, _level: u64) {}
}
