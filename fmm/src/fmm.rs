//! Implementation of FmmData and Fmm traits.
use std::collections::HashMap;

use num::Float;
use rlst_dense::{rlst_dynamic_array2, types::RlstScalar};

use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData},
    fmm::{Fmm, SourceTranslation, TargetTranslation},
    kernel::Kernel,
    tree::{FmmTree, Tree},
    types::EvalType,
};

use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::{
    builder::FmmEvalType,
    types::{C2EType, SendPtrMut},
};

/// Combines the old datatree + Fmm structs into a single storage of metadata
pub struct KiFmm<
    T: FmmTree<Tree = SingleNodeTree<W>>,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: RlstScalar<Real = W> + Float + Default,
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
    pub leaf_upward_surfaces_sources: Vec<W>,
    pub leaf_upward_surfaces_targets: Vec<W>,

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
    T: FmmTree<Tree = SingleNodeTree<W>, NodeIndex = MortonKey> + Send + Sync,
    U: SourceToTargetData<V> + Send + Sync,
    V: Kernel<T = W> + Send + Sync,
    W: RlstScalar<Real = W> + Float + Default,
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
    T: FmmTree<Tree = SingleNodeTree<W>> + Default,
    U: SourceToTargetData<V> + Default,
    V: Kernel + Default,
    W: RlstScalar<Real = W> + Float + Default,
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

// Dummy implementation that simply calls a direct evaluation
pub struct KiFmmDummy<T, U, V>
where
    T: FmmTree<Tree = SingleNodeTree<U>>,
    U: RlstScalar<Real = U> + Float + Default,
    V: Kernel<T = U> + Send + Sync,
{
    pub tree: T,
    pub charges: Vec<U>,
    pub potentials: Vec<U>,
    pub expansion_order: usize,
    pub kernel: V,
    pub kernel_eval_type: EvalType,
    pub fmm_eval_type: FmmEvalType,
    pub eval_size: usize,
    pub charge_index_pointer_targets: Vec<(usize, usize)>,
}

impl<T, U, V> Fmm for KiFmmDummy<T, U, V>
where
    T: FmmTree<Tree = SingleNodeTree<U>, NodeIndex = MortonKey>,
    U: RlstScalar<Real = U> + Float + Default,
    V: Kernel<T = U> + Send + Sync,
{
    type NodeIndex = T::NodeIndex;
    type Precision = U;
    type Tree = T;
    type Kernel = V;

    fn get_dim(&self) -> usize {
        3
    }

    fn get_multipole(&self, _key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        None
    }

    fn get_local(&self, _key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        None
    }

    fn get_potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>> {
        let ntarget_coordinates = self
            .tree
            .get_target_tree()
            .get_all_coordinates()
            .unwrap()
            .len()
            / self.get_dim();

        if let Some(&leaf_idx) = self.tree.get_target_tree().get_leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.eval_size..r * self.eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let vec_displacement = eval_idx * ntarget_coordinates;
                        let slice = &self.potentials[vec_displacement + l..vec_displacement + r];
                        slices.push(slice);
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn evaluate(&self) {
        let all_target_coordinates = self.tree.get_target_tree().get_all_coordinates().unwrap();
        let ntarget_coordinates = all_target_coordinates.len() / self.get_dim();
        let all_source_coordinates = self.tree.get_source_tree().get_all_coordinates().unwrap();
        let nsource_coordinates = all_target_coordinates.len() / self.get_dim();

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let charges = &self.charges;
                let res = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.potentials.as_ptr() as *mut U,
                        ntarget_coordinates,
                    )
                };
                self.kernel.evaluate_st(
                    self.kernel_eval_type,
                    all_source_coordinates,
                    all_target_coordinates,
                    charges,
                    res,
                )
            }

            FmmEvalType::Matrix(nmatvec) => {
                for i in 0..nmatvec {
                    let charges_i =
                        &self.charges[i * nsource_coordinates..(i + 1) * nsource_coordinates];
                    let res_i = unsafe {
                        std::slice::from_raw_parts_mut(
                            self.potentials.as_ptr().add(ntarget_coordinates) as *mut U,
                            ntarget_coordinates,
                        )
                    };
                    self.kernel.evaluate_st(
                        self.kernel_eval_type,
                        all_source_coordinates,
                        all_target_coordinates,
                        charges_i,
                        res_i,
                    )
                }
            }
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
}

#[cfg(test)]
mod test {

    use bempp_field::constants::ALPHA_INNER;
    use bempp_field::helpers::ncoeffs_kifmm;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::{constants::ROOT, implementations::helpers::points_fixture};
    use num::Float;
    use rlst_dense::array::Array;
    use rlst_dense::base_array::BaseArray;
    use rlst_dense::data_container::VectorContainer;
    use rlst_dense::rlst_array_from_slice2;
    use rlst_dense::traits::{RawAccess, RawAccessMut, Shape};

    use crate::{builder::KiFmmBuilderSingleNode, tree::SingleNodeFmmTree};
    use bempp_field::types::{BlasFieldTranslationKiFmm, FftFieldTranslationKiFmm};

    use super::*;

    fn test_root_multipole_laplace_single_node<T: RlstScalar<Real = T> + Float + Default>(
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

    fn test_single_node_laplace_fmm<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let eval_size;
        match eval_type {
            EvalType::Value => eval_size = 1,
            EvalType::ValueDeriv => eval_size = 4,
        };

        let leaf_idx = 0;
        let leaf: MortonKey = fmm.get_tree().get_target_tree().get_all_leaves().unwrap()[leaf_idx];
        let potential = fmm.get_potential(&leaf).unwrap()[0];

        let leaf_targets = fmm
            .get_tree()
            .get_target_tree()
            .get_coordinates(&leaf)
            .unwrap();

        let ntargets = leaf_targets.len() / fmm.get_dim();
        let mut direct = vec![T::zero(); ntargets * eval_size];

        let leaf_coordinates_row_major = rlst_array_from_slice2!(
            T,
            leaf_targets,
            [ntargets, fmm.get_dim()],
            [fmm.get_dim(), 1]
        );
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.get_dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.get_kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = num::Float::abs(d - p);
            let rel_error = abs_error / p;
            println!("err {:?} d {:?} \np {:?}", rel_error, direct, potential);
            assert!(rel_error <= threshold)
        });
    }

    fn test_single_node_laplace_fmm_matrix<T: RlstScalar<Real = T> + Float + Default>(
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
        let leaf_idx = 0;
        let leaf: MortonKey = fmm.get_tree().get_target_tree().get_all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm
            .get_tree()
            .get_target_tree()
            .get_coordinates(&leaf)
            .unwrap();

        let ntargets = leaf_targets.len() / fmm.get_dim();

        let leaf_coordinates_row_major = rlst_array_from_slice2!(
            T,
            leaf_targets,
            [ntargets, fmm.get_dim()],
            [fmm.get_dim(), 1]
        );

        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.get_dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        let [nsources, nmatvec] = charges.shape();

        for i in 0..nmatvec {
            let potential_i = fmm.get_potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let mut direct_i = vec![T::zero(); ntargets];
            fmm.get_kernel().evaluate_st(
                EvalType::Value,
                sources.data(),
                leaf_coordinates_col_major.data(),
                charges_i,
                &mut direct_i,
            );

            println!(
                "i {:?} \n direct_i {:?}\n potential_i {:?}",
                i, direct_i, potential_i
            );
            direct_i.iter().zip(potential_i).for_each(|(&d, &p)| {
                let abs_error = num::Float::abs(d - p);
                let rel_error = abs_error / p;
                assert!(rel_error <= threshold)
            });
        }
    }

    fn test_root_multipole_laplace_single_node_matrix<T: RlstScalar<Real = T> + Float + Default>(
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

        let [nsources, nmatvecs] = charges.shape();

        let mut expected = vec![T::zero(); nmatvecs];
        let mut found = vec![T::zero(); nmatvecs];

        let ncoeffs = ncoeffs_kifmm(fmm.get_expansion_order());

        for eval_idx in 0..nmatvecs {
            fmm.get_kernel().evaluate_st(
                bempp_traits::types::EvalType::Value,
                sources.data(),
                &test_point,
                &charges.data()[eval_idx * nsources..(eval_idx + 1) * nsources],
                &mut expected[eval_idx..eval_idx + 1],
            );
        }

        for eval_idx in 0..nmatvecs {
            let multipole_i = &multipole[eval_idx * ncoeffs..(eval_idx + 1) * ncoeffs];
            fmm.get_kernel().evaluate_st(
                bempp_traits::types::EvalType::Value,
                &upward_equivalent_surface,
                &test_point,
                &multipole_i,
                &mut found[eval_idx..eval_idx + 1],
            );
        }
        for (&a, &b) in expected.iter().zip(found.iter()) {
            let abs_error = num::Float::abs(b - a);
            let rel_error = abs_error / a;
            assert!(rel_error <= threshold);
        }
    }

    #[test]
    fn test_upward_pass_vector() {
        // Setup random sources and targets
        let nsources = 10000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;

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

        let svd_threshold = Some(1e-5);
        let fmm_svd = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, &charges, n_crit, sparse)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                BlasFieldTranslationKiFmm::new(svd_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate();

        let fmm_fft = Box::new(fmm_fft);
        let fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node(fmm_fft, &sources, &charges, 1e-5);
        test_root_multipole_laplace_single_node(fmm_svd, &sources, &charges, 1e-5);
    }

    #[test]
    fn test_fmm_vector() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));
        // FMM parameters
        let n_crit = Some(10);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;
        let threshold_deriv = 1e-4;
        let threshold_deriv_blas = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        // fmm with fft based field translation
        {
            // Evaluate potentials
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
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_laplace_fmm(fmm_fft, eval_type, &sources, &charges, threshold_pot);

            // Evaluate potentials + derivatives
            let fmm_fft = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, &charges, n_crit, sparse)
                .parameters(
                    expansion_order,
                    Laplace3dKernel::new(),
                    bempp_traits::types::EvalType::ValueDeriv,
                    FftFieldTranslationKiFmm::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_laplace_fmm(fmm_fft, eval_type, &sources, &charges, threshold_deriv);
        }

        // fmm with BLAS based field translation
        {
            // Evaluate potentials
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, &charges, n_crit, sparse)
                .parameters(
                    expansion_order,
                    Laplace3dKernel::new(),
                    bempp_traits::types::EvalType::Value,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let eval_type = fmm_blas.kernel_eval_type;
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm(fmm_blas, eval_type, &sources, &charges, threshold_pot);

            // Evaluate potentials + derivatives
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, &charges, n_crit, sparse)
                .parameters(
                    expansion_order,
                    Laplace3dKernel::new(),
                    bempp_traits::types::EvalType::ValueDeriv,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let eval_type = fmm_blas.kernel_eval_type;
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv_blas,
            );
        }
    }

    #[test]
    fn test_fmm_matrix() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));
        // FMM parameters
        let n_crit = Some(10);
        let expansion_order = 6;
        let sparse = true;
        let threshold = 1e-5;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);

        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (1 + i) as f64));

        // fmm with blas based field translation
        {
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, &charges, n_crit, sparse)
                .parameters(
                    expansion_order,
                    Laplace3dKernel::new(),
                    bempp_traits::types::EvalType::Value,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();

            let fmm_blas = Box::new(fmm_blas);
            test_single_node_laplace_fmm_matrix(fmm_blas, &sources, &charges, threshold);
        }
    }

    #[test]
    fn test_upward_pass_matrix() {
        // Setup random sources and targets
        let npoints = 10000;
        let sources = points_fixture::<f64>(npoints, None, None, Some(0));
        let targets = points_fixture::<f64>(npoints, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = false;
        let threshold = 1e-5;
        let svd_threshold = Some(1e-5);

        // Charge data
        let nvecs = 3;
        let mut charges = rlst_dynamic_array2!(f64, [npoints, nvecs]);

        charges
            .data_mut()
            .chunks_exact_mut(npoints)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|elem| *elem += (i + 1) as f64));

        let fmm_blas = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, &charges, n_crit, sparse)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                BlasFieldTranslationKiFmm::new(svd_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_blas.evaluate();

        let fmm_blas = Box::new(fmm_blas);
        test_root_multipole_laplace_single_node_matrix(fmm_blas, &sources, &charges, threshold);
    }
}
