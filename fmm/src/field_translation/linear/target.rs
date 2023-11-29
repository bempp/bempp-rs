//! kiFMM based on simple linear data structures that minimises memory allocations, maximises cache re-use.

use itertools::Itertools;
use num::Float;
use rayon::prelude::*;
use std::collections::HashMap;

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

// Try this two different ways, ignoring w,x lists and also including them
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
        if let Some(sources) = self.fmm.tree().get_keys(level) {
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let nsources = sources.len();
            let min = &sources[0];
            let max = &sources[nsources - 1];
            let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
            let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

            let locals = &self.locals[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

            let nsiblings = 8;
            let mut max_chunk_size = 8_i32.pow((level - 1).try_into().unwrap()) as usize;

            if max_chunk_size > P2M_MAX_CHUNK_SIZE {
                max_chunk_size = P2M_MAX_CHUNK_SIZE;
            }
            let chunk_size = find_chunk_size(nsources, max_chunk_size);
            locals
                .par_chunks_exact(nsiblings * ncoeffs*chunk_size)
                .zip(self.level_multipoles[(level + 1) as usize].par_chunks_exact(chunk_size))
                .for_each(|(multipole_chunk, parent)| {

                    unsafe {
                        let tmp = rlst_pointer_mat!['a, V, multipole_chunk.as_ptr(), (ncoeffs*nsiblings, chunk_size), (1, ncoeffs*nsiblings)];
                        let tmp = self.fmm.l2l.dot(&tmp).eval();

                        for i in 0..chunk_size {
                            let mut ptr = parent[i].raw;
                            for j in 0..ncoeffs {
                                *ptr += tmp.data()[(i*ncoeffs)+j];
                                ptr = ptr.add(1)
                            }
                        }
                    }
                })
        }
    }

    fn m2p<'a>(&self) {}

    fn l2p<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let nleaves = leaves.len();
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

            let surface_size = ncoeffs * self.fmm.kernel.space_dimension();
        }
    }

    fn p2l<'a>(&self) {}

    fn p2p<'a>(&self) {
        if let Some(leaves) = self.fmm.tree().get_all_leaves() {
            let dim = self.fmm.kernel.space_dimension();

            let mut target_map = HashMap::new();

            for (i, k) in leaves.iter().enumerate() {
                target_map.insert(k, i);
            }

            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

            leaves
                .par_iter()
                .enumerate()
                .zip(&self.charge_index_pointer)
                .for_each(|((i, leaf), charge_index_pointer)| {
                    let targets =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = targets.len() / dim;

                    if ntargets > 0 {
                        let mut local_result = vec![V::zero(); ntargets];
                        let mut result = self.potentials_send_pointers[i].raw;

                        if let Some(u_list) = self.fmm.get_u_list(leaf) {
                            let u_list_indices = u_list.iter().filter_map(|k| target_map.get(k));

                            let charges = u_list_indices
                                .clone()
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    let charges = &self.charges[index_pointer.0..index_pointer.1];
                                    charges
                                })
                                .collect_vec();

                            let coordinates = u_list_indices
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    let coords =
                                        &coordinates[index_pointer.0 * dim..index_pointer.1 * dim];
                                    coords
                                })
                                .collect_vec();

                            for (&charges, coords) in charges.iter().zip(coordinates) {
                                let nsources = coords.len() / dim;

                                if nsources > 0 {
                                    self.fmm.kernel.evaluate_st(
                                        EvalType::Value,
                                        coords,
                                        targets,
                                        charges,
                                        &mut local_result,
                                    )
                                }
                            }
                            // Save to global locations
                            for res in local_result.iter() {
                                unsafe {
                                    *result += *res;
                                    result = result.add(1);
                                }
                            }
                        }
                    }
                })
        }
    }
}
