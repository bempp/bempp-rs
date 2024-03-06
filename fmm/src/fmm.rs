//! Implementation of FmmData and Fmm traits.
use num::Float;
use rlst_blis::interface::gemm::Gemm;
use rlst_common::types::Scalar;
use std::collections::HashMap;

use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array2,
    traits::{MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, Shape},
};

use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData, SourceToTargetHomogenousScaleInvariant},
    fmm::{Fmm, SourceTranslation, TargetTranslation},
    kernel::{HomogenousKernel, Kernel},
    tree::{FmmTree, Tree},
};

use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTreeNew};

use crate::{
    builder::FmmEvaluationMode,
    types::{C2EType, SendPtrMut},
};

pub fn ncoeffs(expansion_order: usize) -> usize {
    6 * (expansion_order - 1).pow(2) + 2
}

/// Combines the old datatree + Fmm structs into a single storage of metadata
pub struct KiFmm<
    T: FmmTree<Tree = SingleNodeTreeNew<W>>,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: Scalar<Real = W> + Default + Float,
> {
    pub tree: T,
    pub source_to_target_data: U,
    pub kernel: V,
    pub expansion_order: usize,
    pub ncoeffs: usize,
    pub eval_mode: FmmEvaluationMode,
    pub source_data_vec: Vec<C2EType<W>>,
    pub eval_size: usize,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_1: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_2: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_1: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_2: C2EType<W>,

    /// The multipole to multipole operator matrices, each index is associated with a child box (in sequential Morton order),
    pub source_data: C2EType<W>,

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub target_data: Vec<C2EType<W>>,

    /// The multipole expansion data at each box.
    pub multipoles: Vec<W>,

    /// Multipole expansions at leaf level
    pub leaf_multipoles: Vec<Vec<SendPtrMut<W>>>,

    /// Multipole expansions at each level
    pub level_multipoles: Vec<Vec<Vec<SendPtrMut<W>>>>,

    /// The local expansion at each box
    pub locals: Vec<W>,

    /// Local expansions at the leaf level
    pub leaf_locals: Vec<Vec<SendPtrMut<W>>>,

    /// The local expansion data at each level.
    pub level_locals: Vec<Vec<Vec<SendPtrMut<W>>>>,

    /// index pointers to each key at a given level, indexed by level.
    pub level_index_pointer_locals: Vec<HashMap<MortonKey, usize>>,

    /// index pointers to each key at a given level, indexed by level.
    pub level_index_pointer_multipoles: Vec<HashMap<MortonKey, usize>>,

    /// The evaluated potentials at each leaf box.
    pub potentials: Vec<W>,

    /// The evaluated potentials at each leaf box.
    pub potentials_send_pointers: Vec<SendPtrMut<W>>,

    /// All upward surfaces
    pub upward_surfaces: Vec<W>,

    /// All downward surfaces
    pub downward_surfaces: Vec<W>,

    /// Leaf upward surfaces
    pub leaf_upward_surfaces: Vec<W>,

    /// Leaf downward surfaces
    pub leaf_downward_surfaces: Vec<W>,

    /// The charge data at each leaf box.
    pub charges: Vec<W>,

    /// Index pointer between leaf keys and charges
    pub charge_index_pointer: Vec<(usize, usize)>,

    /// Scales of each leaf operator
    pub target_scales: Vec<W>,

    /// Scales of each leaf operator
    pub source_scales: Vec<W>,

    /// Global indices of each charge
    pub global_indices: Vec<usize>,
}

// TODO: Remove scale function

// impl<T, U, V> SourceToTargetHomogenousScaleInvariant<U>
//     for NewKiFmm<V, FftFieldTranslationKiFmmNew<U, T>, T, U>
// where
//     T: HomogenousKernel<T = U> + std::marker::Send + std::marker::Sync + Default,
//     U: Scalar<Real = U>
//         + Float
//         + Default
//         + std::marker::Send
//         + std::marker::Sync
//         + Fft
//         + rlst_blis::interface::gemm::Gemm,
//     Complex<U>: Scalar,
//     Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
//     V: FmmTree,
// {
//     fn s2t_scale(&self, level: u64) -> U {
//         if level < 2 {
//             panic!("M2L only perfomed on level 2 and below")
//         }

//         if level == 2 {
//             U::from(1. / 2.).unwrap()
//         } else {
//             let two = U::from(2.0).unwrap();
//             Scalar::powf(two, U::from(level - 3).unwrap())
//         }
//     }
// }

impl<T, U, V, W> Fmm for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTreeNew<W>, NodeIndex = MortonKey> + Send + Sync,
    U: SourceToTargetData<V> + Send + Sync,
    V: Kernel<T = W> + Send + Sync,
    W: Scalar<Real = W> + Default + Send + Sync + Gemm + Float,
{
    type NodeIndex = T::NodeIndex;
    type Precision = W;

    fn get_multipole_data(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        if let Some(index) = self.tree.get_source_tree().get_index(key) {
            match self.eval_mode {
                FmmEvaluationMode::Vector => {
                    Some(&self.multipoles[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvaluationMode::Matrix(nmatvecs) => Some(
                    &self.multipoles
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn get_local_data(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        if let Some(index) = self.tree.get_target_tree().get_index(key) {
            match self.eval_mode {
                FmmEvaluationMode::Vector => {
                    Some(&self.locals[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvaluationMode::Matrix(nmatvecs) => Some(
                    &self.locals
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn get_potential_data(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>> {
        if let Some(&leaf_idx) = self.tree.get_target_tree().get_leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer[leaf_idx];
            let ntargets = r - l;

            match self.eval_mode {
                FmmEvaluationMode::Vector => Some(vec![
                    &self.potentials[l * self.eval_size..r * self.eval_size],
                ]),
                FmmEvaluationMode::Matrix(nmatvecs) => {
                    let nleaves = self.tree.get_target_tree().get_nleaves().unwrap();
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let potentials_pointer =
                            self.potentials_send_pointers[eval_idx * nleaves + leaf_idx].raw;
                        slices.push(unsafe {
                            std::slice::from_raw_parts(
                                potentials_pointer,
                                ntargets * self.eval_size,
                            )
                        });
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn evaluate(&self) {
        // Upward pass
        {
            self.p2m();
            for level in (1..=self.tree.get_source_tree().get_depth()).rev() {
                self.m2m(level);
            }
        }

        // Downward pass
        {}
    }
}

impl<T, U, V, W> Default for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTreeNew<W>> + Default,
    U: SourceToTargetData<V> + Default,
    V: HomogenousKernel + Default,
    W: Scalar<Real = W> + Default + Float,
{
    fn default() -> Self {
        let uc2e_inv_1 = rlst_dynamic_array2!(W, [1, 1]);
        let uc2e_inv_2 = rlst_dynamic_array2!(W, [1, 1]);
        let dc2e_inv_1 = rlst_dynamic_array2!(W, [1, 1]);
        let dc2e_inv_2 = rlst_dynamic_array2!(W, [1, 1]);
        let source = rlst_dynamic_array2!(W, [1, 1]);
        KiFmm {
            tree: T::default(),
            source_to_target_data: U::default(),
            kernel: V::default(),
            expansion_order: 0,
            eval_mode: FmmEvaluationMode::Vector,
            eval_size: 0,
            ncoeffs: 0,
            uc2e_inv_1,
            uc2e_inv_2,
            dc2e_inv_1,
            dc2e_inv_2,
            source_data: source,
            source_data_vec: Vec::default(),
            target_data: Vec::default(),
            multipoles: Vec::default(),
            locals: Vec::default(),
            leaf_multipoles: Vec::default(),
            level_multipoles: Vec::default(),
            leaf_locals: Vec::default(),
            level_locals: Vec::default(),
            level_index_pointer_locals: Vec::default(),
            level_index_pointer_multipoles: Vec::default(),
            potentials: Vec::default(),
            potentials_send_pointers: Vec::default(),
            upward_surfaces: Vec::default(),
            downward_surfaces: Vec::default(),
            leaf_upward_surfaces: Vec::default(),
            leaf_downward_surfaces: Vec::default(),
            charges: Vec::default(),
            charge_index_pointer: Vec::default(),
            source_scales: Vec::default(),
            target_scales: Vec::default(),
            global_indices: Vec::default(),
        }
    }
}

mod test {

    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::{constants::ROOT, implementations::helpers::points_fixture};

    use crate::{builder::KiFmmBuilderSingleNode, constants::ALPHA_INNER};
    use bempp_field::types::{FftFieldTranslationKiFmmNew, SvdFieldTranslationKiFmm};

    use super::*;

    #[test]

    fn test_upward_pass() {
        let npoints = 10000;
        let nvecs = 1;
        let sources = points_fixture::<f64>(npoints, None, None);
        let targets = points_fixture::<f64>(npoints, None, None);
        let tmp = vec![1.0; npoints * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [npoints, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let n_crit = Some(100);
        let expansion_order = 10;
        let sparse = false;

        let fmm = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, &charges, n_crit, sparse)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmmNew::default(),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm.evaluate();

        let multipole = fmm.get_multipole_data(&ROOT).unwrap();
        let upward_equivalent_surface =
            ROOT.compute_surface(fmm.tree.get_domain(), fmm.expansion_order, ALPHA_INNER);
        let test_point = vec![100000., 0., 0.];
        let mut expected = vec![0.];
        let mut found = vec![0.];

        fmm.kernel.evaluate_st(
            bempp_traits::types::EvalType::Value,
            sources.data(),
            &test_point,
            charges.data(),
            &mut expected,
        );

        fmm.kernel.evaluate_st(
            bempp_traits::types::EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            multipole,
            &mut found,
        );

        let abs_error = num::Float::abs(expected[0] - found[0]);
        let rel_error = abs_error / expected[0];
        assert!(rel_error < 1e-10);
    }
}
