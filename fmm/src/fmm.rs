//! Implementation of FmmData and Fmm traits.
use std::collections::HashMap;

use rlst_dense::rlst_dynamic_array2;

use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData},
    fmm::{Fmm, SourceTranslation, TargetTranslation},
    kernel::Kernel,
    tree::{FmmTree, Tree},
    types::EvalType,
};

use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTreeNew};

use crate::{
    builder::FmmEvalType,
    traits::FmmScalar,
    types::{C2EType, SendPtrMut},
};

/// Combines the old datatree + Fmm structs into a single storage of metadata
pub struct KiFmm<
    T: FmmTree<Tree = SingleNodeTreeNew<W>>,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: FmmScalar,
> {
    pub tree: T,
    pub source_to_target_data: U,
    pub kernel: V,
    pub expansion_order: usize,
    pub ncoeffs: usize,
    pub fmm_eval_type: FmmEvalType,
    pub kernel_eval_type: EvalType,
    pub source_data_vec: Vec<C2EType<W>>,
    pub eval_size: usize,
    pub charge_index_pointer_sources: Vec<(usize, usize)>,
    pub charge_index_pointer_targets: Vec<(usize, usize)>,
    pub dim: usize,

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
    pub leaf_upward_surfaces_sources: Vec<W>,
    pub leaf_upward_surfaces_targets: Vec<W>,

    /// Leaf downward surfaces
    pub leaf_downward_surfaces: Vec<W>,

    /// The charge data at each leaf box.
    pub charges: Vec<W>,

    /// Scales of each leaf operator
    pub target_scales: Vec<W>,

    /// Scales of each leaf operator
    pub source_scales: Vec<W>,

    /// Global indices of each charge
    pub global_indices: Vec<usize>,
}

impl<T, U, V, W> Fmm for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTreeNew<W>, NodeIndex = MortonKey> + Send + Sync,
    U: SourceToTargetData<V> + Send + Sync,
    V: Kernel<T = W> + Send + Sync,
    W: FmmScalar,
    Self: SourceToTarget,
{
    type NodeIndex = T::NodeIndex;
    type Precision = W;
    type Kernel = V;
    type Tree = T;

    fn get_dim(&self) -> usize {
        self.dim
    }

    fn get_multipole(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        if let Some(index) = self.tree.get_source_tree().get_index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.multipoles[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.multipoles
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn get_local(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        if let Some(index) = self.tree.get_target_tree().get_index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.locals[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.locals
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn get_potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>> {
        if let Some(&leaf_idx) = self.tree.get_target_tree().get_leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];
            let ntargets = r - l;

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.eval_size..r * self.eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
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

    fn get_expansion_order(&self) -> usize {
        self.expansion_order
    }

    fn get_kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn get_tree(&self) -> &Self::Tree {
        &self.tree
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
        {
            for level in 2..=self.tree.get_target_tree().get_depth() {
                if level > 2 {
                    self.l2l(level);
                }
                self.m2l(level);
                self.p2l(level)
            }

            // Leaf level computations
            self.m2p();
            self.p2p();
            self.l2p();
        }
    }
}

impl<T, U, V, W> Default for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTreeNew<W>> + Default,
    U: SourceToTargetData<V> + Default,
    V: Kernel + Default,
    W: FmmScalar,
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
            fmm_eval_type: FmmEvalType::Vector,
            kernel_eval_type: EvalType::Value,
            eval_size: 0,
            dim: 0,
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
            leaf_upward_surfaces_sources: Vec::default(),
            leaf_upward_surfaces_targets: Vec::default(),
            leaf_downward_surfaces: Vec::default(),
            charges: Vec::default(),
            charge_index_pointer_sources: Vec::default(),
            charge_index_pointer_targets: Vec::default(),
            source_scales: Vec::default(),
            target_scales: Vec::default(),
            global_indices: Vec::default(),
        }
    }
}

#[cfg(test)]
mod test {

    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::{constants::ROOT, implementations::helpers::points_fixture};
    use rlst_dense::array::Array;
    use rlst_dense::base_array::BaseArray;
    use rlst_dense::data_container::VectorContainer;
    use rlst_dense::rlst_array_from_slice2;
    use rlst_dense::traits::{RawAccess, RawAccessMut};

    use crate::{builder::KiFmmBuilderSingleNode, constants::ALPHA_INNER, tree::SingleNodeFmmTree};
    use bempp_field::types::FftFieldTranslationKiFmm;

    use super::*;

    fn test_root_multipole_laplace_single_node<T: FmmScalar>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let multipole = fmm.get_multipole(&ROOT).unwrap();
        let upward_equivalent_surface = ROOT.compute_surface(
            fmm.get_tree().get_domain(),
            fmm.get_expansion_order(),
            T::from(ALPHA_INNER).unwrap(),
        );
        let test_point = vec![T::from(100000.).unwrap(), T::zero(), T::zero()];
        let mut expected = vec![T::zero()];
        let mut found = vec![T::zero()];

        fmm.get_kernel().evaluate_st(
            bempp_traits::types::EvalType::Value,
            sources.data(),
            &test_point,
            charges.data(),
            &mut expected,
        );

        fmm.get_kernel().evaluate_st(
            bempp_traits::types::EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            multipole,
            &mut found,
        );

        let abs_error = num::Float::abs(expected[0] - found[0]);
        let rel_error = abs_error / expected[0];
        assert!(rel_error <= threshold);
    }

    fn test_single_node_laplace_fmm<T: FmmScalar>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {

        let leaf_idx = 2;
        let leaf: MortonKey = fmm.get_tree().get_target_tree().get_all_leaves().unwrap()[leaf_idx];
        let potential = fmm.get_potential(&leaf).unwrap()[0];

        let leaf_targets = fmm
            .get_tree()
            .get_target_tree()
            .get_coordinates(&leaf)
            .unwrap();

        let ntargets = leaf_targets.len() / fmm.get_dim();
        let mut direct = vec![T::zero(); ntargets];

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(T, leaf_targets, [ntargets, fmm.get_dim()], [fmm.get_dim(), 1]);
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.get_dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.get_kernel().evaluate_st(
            EvalType::Value,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );
        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = num::Float::abs(d - p);
            let rel_error = abs_error / p;
            assert!(rel_error <= threshold)
        });


    }


    // fn test_root_multipole_laplace_single_node_matrix<T: FmmScalar>(
    //     fmm: Box<
    //         dyn Fmm<
    //             Precision = T,
    //             NodeIndex = MortonKey,
    //             Kernel = Laplace3dKernel<T>,
    //             Tree = SingleNodeFmmTree<T>,
    //         >,
    //     >,
    //     sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
    //     charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
    //     threshold: T,
    // ) {
    //     let multipole = fmm.get_multipole(&ROOT).unwrap();
    //     let upward_equivalent_surface = ROOT.compute_surface(
    //         fmm.get_tree().get_domain(),
    //         fmm.get_expansion_order(),
    //         T::from(ALPHA_INNER).unwrap(),
    //     );
    //     let test_point = vec![T::from(100000.).unwrap(), T::zero(), T::zero()];

    //     let [nsources, nmatvecs] = charges.shape();

    //     let mut expected = vec![T::zero(); nmatvecs];
    //     let mut found = vec![T::zero(); nmatvecs];

    //     let ncoeffs = ncoeffs_kifmm(fmm.get_expansion_order());

    //     for eval_idx in 0..nmatvecs {
    //         fmm.get_kernel().evaluate_st(
    //             bempp_traits::types::EvalType::Value,
    //             sources.data(),
    //             &test_point,
    //             &charges.data()[eval_idx * nsources..(eval_idx + 1) * nsources],
    //             &mut expected[eval_idx..eval_idx + 1],
    //         );
    //     }

    //     for eval_idx in 0..nmatvecs {
    //         let multipole_i = &multipole[eval_idx * ncoeffs..(eval_idx + 1) * ncoeffs];
    //         fmm.get_kernel().evaluate_st(
    //             bempp_traits::types::EvalType::Value,
    //             &upward_equivalent_surface,
    //             &test_point,
    //             &multipole_i,
    //             &mut found[eval_idx..eval_idx + 1],
    //         );
    //     }
    //     for (&a, &b) in expected.iter().zip(found.iter()) {
    //         let abs_error = num::Float::abs(b - a);
    //         let rel_error = abs_error / a;
    //         assert!(rel_error <= threshold);
    //         assert!(false);
    //     }
    // }

    #[test]
    fn test_upward_pass_vector() {
        // Setup random sources and targets
        let nsources = 10000;
        let ntargets = 9000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(0));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = false;
        // let svd_threshold = Some(1e-5);

        // Charge data
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let fmm_fft = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, &charges, n_crit, sparse)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate();

        // let fmm_svd = KiFmmBuilderSingleNode::new()
        //     .tree(&sources, &targets, &charges, n_crit, sparse)
        //     .parameters(
        //         expansion_order,
        //         Laplace3dKernel::new(),
        //         bempp_traits::types::EvalType::Value,
        //         BlasFieldTranslationKiFmm::new(svd_threshold),
        //     )
        //     .unwrap()
        //     .build()
        //     .unwrap();
        // fmm_svd.evaluate();

        let fmm_fft = Box::new(fmm_fft);
        // let fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node(fmm_fft, &sources, &charges, 1e-5);
        // test_root_multipole_laplace_single_node(fmm_svd, &sources, &charges, 1e-5);
    }

    #[test]
    fn test_fmm_vector() {
        // Setup random sources and targets
        let nsources = 50000;
        let ntargets = 10000;
        // let min = Some(0.1);
        // let max = Some(0.9);
        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(3));
        // FMM parameters
        let n_crit = Some(10);
        let expansion_order = 7;
        let sparse = true;
        let threshold = 1e-6;

        // Charge data
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let fmm_fft = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, &charges, n_crit, sparse)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate();

        let fmm_fft = Box::new(fmm_fft);
        test_single_node_laplace_fmm(fmm_fft, &sources, &charges, threshold);


        // let fmm_svd = KiFmmBuilderSingleNode::new()
        //     .tree(&sources, &targets, &charges, n_crit, sparse)
        //     .parameters(
        //         expansion_order,
        //         Laplace3dKernel::new(),
        //         bempp_traits::types::EvalType::Value,
        //         BlasFieldTranslationKiFmm::new(svd_threshold),
        //     )
        //     .unwrap()
        //     .build()
        //     .unwrap();
        // fmm_svd.evaluate();
        // let fmm_svd = Box::new(fmm_svd);
        // test_root_multipole_laplace_single_node(fmm_svd, &sources, &charges, 1e-5);

        // let fmm_fft = Box::new(fmm_fft);
        // test_root_multipole_laplace_single_node(fmm_fft, &sources, &charges, 1e-5);
    }

    // #[test]
    // fn test_upward_pass_matrix() {
    //     // Setup random sources and targets
    //     let npoints = 10000;
    //     let sources = points_fixture::<f64>(npoints, None, None, Some(0));
    //     let targets = points_fixture::<f64>(npoints, None, None, Some(1));

    //     // FMM parameters
    //     let n_crit = Some(100);
    //     let expansion_order = 7;
    //     let sparse = false;
    //     let svd_threshold = Some(1e-5);

    //     // Charge data
    //     let nvecs = 3;
    //     let mut charges = rlst_dynamic_array2!(f64, [npoints, nvecs]);

    //     charges
    //         .data_mut()
    //         .chunks_exact_mut(npoints)
    //         .enumerate()
    //         .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (i + 1) as f64));

    //     let fmm_fft = KiFmmBuilderSingleNode::new()
    //         .tree(&sources, &targets, &charges, n_crit, sparse)
    //         .parameters(
    //             expansion_order,
    //             Laplace3dKernel::new(),
    //             bempp_traits::types::EvalType::Value,
    //             FftFieldTranslationKiFmm::default(),
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();

    //     let fmm_svd = KiFmmBuilderSingleNode::new()
    //         .tree(&sources, &targets, &charges, n_crit, sparse)
    //         .parameters(
    //             expansion_order,
    //             Laplace3dKernel::new(),
    //             bempp_traits::types::EvalType::Value,
    //             BlasFieldTranslationKiFmm::new(svd_threshold),
    //         )
    //         .unwrap()
    //         .build()
    //         .unwrap();

    //     fmm_fft.evaluate();
    //     fmm_svd.evaluate();

    //     let fmm_fft = Box::new(fmm_fft);
    //     let fmm_svd = Box::new(fmm_svd);
    //     test_root_multipole_laplace_single_node_matrix(fmm_fft, &sources, &charges, 1e-5);
    //     test_root_multipole_laplace_single_node_matrix(fmm_svd, &sources, &charges, 1e-5);
    // }
}
