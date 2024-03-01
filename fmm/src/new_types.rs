use std::{collections::HashMap, thread::LocalKey};

use bempp_field::{
    fft::Fft,
    types::{FftFieldTranslationKiFmm, FftFieldTranslationKiFmmNew, SvdFieldTranslationKiFmm},
};
use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData, SourceToTargetHomogenousScaleInvariant},
    fmm::{Fmm, NewFmm, SourceTranslation, TargetTranslation},
    kernel::{Kernel, ScaleInvariantHomogenousKernel},
    tree::{FmmTree, Tree},
    types::EvalType,
};
use bempp_tree::types::{
    domain::Domain,
    morton::MortonKey,
    single_node::{SingleNodeTree, SingleNodeTreeNew},
};
use cauchy::Scalar;
use num::{traits::real::Real, Complex, Float};
use rlst_blis::interface::gemm::Gemm;
use rlst_dense::{
    array::Array, base_array::BaseArray, data_container::VectorContainer, rlst_dynamic_array2,
    traits::MatrixSvd,
};

use crate::types::{C2EType, SendPtrMut};

pub fn ncoeffs(expansion_order: usize) -> usize {
    6 * (expansion_order - 1).pow(2) + 2
}

/// Combines the old datatree + Fmm structs into a single storage of metadata
pub struct NewKiFmm<T: FmmTree, U: SourceToTargetData<V>, V: Kernel, W: Scalar + Default> {
    pub tree: T,
    pub field_translation_data: U,
    pub kernel: V,
    pub expansion_order: usize,
    pub ncoeffs: usize,

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
    pub source: C2EType<W>,

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub target: Vec<C2EType<W>>,

    /// The multipole expansion data at each box.
    pub multipoles: Vec<U>,

    /// Multipole expansions at leaf level
    pub leaf_multipoles: Vec<SendPtrMut<U>>,

    /// Multipole expansions at each level
    pub level_multipoles: Vec<Vec<SendPtrMut<U>>>,

    /// The local expansion at each box
    pub locals: Vec<U>,

    /// Local expansions at the leaf level
    pub leaf_locals: Vec<SendPtrMut<U>>,

    /// The local expansion data at each level.
    pub level_locals: Vec<Vec<SendPtrMut<U>>>,

    /// Index pointers to each key at a given level, indexed by level.
    pub level_index_pointer: Vec<HashMap<MortonKey, usize>>,

    /// The evaluated potentials at each leaf box.
    pub potentials: Vec<U>,

    /// The evaluated potentials at each leaf box.
    pub potentials_send_pointers: Vec<SendPtrMut<U>>,

    /// All upward surfaces
    pub upward_surfaces: Vec<U>,

    /// All downward surfaces
    pub downward_surfaces: Vec<U>,

    /// Leaf upward surfaces
    pub leaf_upward_surfaces: Vec<U>,

    /// Leaf downward surfaces
    pub leaf_downward_surfaces: Vec<U>,

    /// The charge data at each leaf box.
    pub charges: Vec<U>,

    /// Index pointer between leaf keys and charges
    pub charge_index_pointer: Vec<(usize, usize)>,

    /// Scales of each leaf operator
    pub scales: Vec<U>,

    /// Global indices of each charge
    pub global_indices: Vec<usize>,
}

pub struct SingleNodeFmmTree<T: Float + Default + Scalar<Real = T>> {
    pub source_tree: SingleNodeTreeNew<T>,
    pub target_tree: SingleNodeTreeNew<T>,
}

#[derive(Default)]
pub struct KiFmmBuilderSingleNode<T, U, V>
where
    T: SourceToTargetData<V>,
    U: Float + Default + Scalar<Real = U>,
    V: Kernel,
{
    tree: Option<SingleNodeFmmTree<U>>,
    source_to_target: Option<T>,
    source_domain: Option<Domain<U>>,
    target_domain: Option<Domain<U>>,
    kernel: Option<V>,
    expansion_order: Option<usize>,
    ncoeffs: Option<usize>,
    eval_type: Option<EvalType>,
}

impl<T> FmmTree for SingleNodeFmmTree<T>
where
    T: Float + Default + Scalar<Real = T>,
{
    type Tree = SingleNodeTreeNew<T>;

    fn get_source_tree(&self) -> &Self::Tree {
        &self.target_tree
    }
    fn get_target_tree(&self) -> &Self::Tree {
        &self.source_tree
    }
}

impl<T, U, V> KiFmmBuilderSingleNode<T, U, V>
where
    T: SourceToTargetData<V, Domain = Domain<U>>,
    U: Float + Scalar<Real = U> + Default,
    V: Kernel + Clone,
{
    // Start building with mandatory parameters
    pub fn new() -> Self {
        KiFmmBuilderSingleNode {
            tree: None,
            source_domain: None,
            target_domain: None,
            source_to_target: None,
            kernel: None,
            expansion_order: None,
            ncoeffs: None,
            eval_type: None,
        }
    }

    pub fn tree(
        mut self,
        sources: &[U],
        targets: &[U],
        n_crit: Option<u64>,
        sparse: Option<bool>,
    ) -> Self {
        let source_tree = SingleNodeTreeNew::new(sources, n_crit, sparse);
        let target_tree = SingleNodeTreeNew::new(targets, n_crit, sparse);
        self.source_domain = Some(source_tree.get_domain().clone());
        self.target_domain = Some(target_tree.get_domain().clone());

        let fmm_tree = SingleNodeFmmTree {
            source_tree,
            target_tree,
        };
        self.tree = Some(fmm_tree);
        self
    }

    pub fn parameters(
        mut self,
        expansion_order: usize,
        kernel: V,
        eval_type: EvalType,
        mut source_to_target: T,
    ) -> Result<Self, String> {
        if self.tree.is_none() {
            Err("Must build tree before specifying FMM parameters".to_string())
        } else {
            self.expansion_order = Some(expansion_order);
            self.ncoeffs = Some(ncoeffs(expansion_order));
            self.kernel = Some(kernel);
            self.eval_type = Some(eval_type);
            // Set the expansion order
            source_to_target.set_expansion_order(self.expansion_order.unwrap());

            // Set the associated kernel
            let kernel = self.kernel.as_ref().unwrap().clone();
            source_to_target.set_kernel(kernel);

            // Compute the field translation operators
            source_to_target.set_operator_data(
                self.expansion_order.unwrap(),
                self.source_domain.unwrap().clone(),
            );

            self.source_to_target = Some(source_to_target);
            Ok(self)
        }
    }

    // Finalize and build the KiFmm
    pub fn build(self) -> Result<NewKiFmm<SingleNodeFmmTree<U>, T, V, U>, String> {
        if self.tree.is_none() || self.source_to_target.is_none() || self.expansion_order.is_none()
        {
            Err("Missing fields for KiFmm".to_string())
        } else {
            let uc2e_inv_1 = rlst_dynamic_array2!(U, [1, 1]);
            let uc2e_inv_2 = rlst_dynamic_array2!(U, [1, 1]);
            let dc2e_inv_1 = rlst_dynamic_array2!(U, [1, 1]);
            let dc2e_inv_2 = rlst_dynamic_array2!(U, [1, 1]);
            let source = rlst_dynamic_array2!(U, [1, 1]);

            Ok(NewKiFmm {
                tree: self.tree.unwrap(),
                field_translation_data: self.source_to_target.unwrap(),
                kernel: self.kernel.unwrap(),
                expansion_order: self.expansion_order.unwrap(),
                ncoeffs: self.ncoeffs.unwrap(),
                uc2e_inv_1,
                uc2e_inv_2,
                dc2e_inv_1,
                dc2e_inv_2,
                source,
                target: Vec::default(),
                multipoles: Vec::default(),
                locals: Vec::default(),
                leaf_multipoles: Vec::default(),
                level_multipoles: Vec::default(),
                leaf_locals: Vec::default(),
                level_locals: Vec::default(),
                level_index_pointer: Vec::default(),
                potentials: Vec::default(),
                potentials_send_pointers: Vec::default(),
                upward_surfaces: Vec::default(),
                downward_surfaces: Vec::default(),
                leaf_upward_surfaces: Vec::default(),
                leaf_downward_surfaces: Vec::default(),
                charges: Vec::default(),
                charge_index_pointer: Vec::default(),
                scales: Vec::default(),
                global_indices: Vec::default(),
            })
        }
    }
}

impl<T, U, V, W> SourceTranslation for NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: Scalar + Default,
{
    fn p2m(&self) {}

    fn m2m(&self, level: u64) {}
}

impl<T, U, V, W> TargetTranslation for NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: Scalar + Default,
{
    fn l2l(&self, level: u64) {}

    fn l2p(&self) {}

    fn m2p(&self) {}

    fn p2p(&self) {}
}

impl<T, U, V> SourceToTarget for NewKiFmm<V, FftFieldTranslationKiFmmNew<U, T>, T, U>
where
    T: ScaleInvariantHomogenousKernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: Scalar<Real = U>
        + Float
        + Default
        + std::marker::Send
        + std::marker::Sync
        + Fft
        + rlst_blis::interface::gemm::Gemm,
    Complex<U>: Scalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree,
{
    fn m2l(&self, level: u64) {}

    fn p2l(&self, level: u64) {}
}

impl<T, U, V> SourceToTargetHomogenousScaleInvariant<U>
    for NewKiFmm<V, FftFieldTranslationKiFmmNew<U, T>, T, U>
where
    T: ScaleInvariantHomogenousKernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: Scalar<Real = U>
        + Float
        + Default
        + std::marker::Send
        + std::marker::Sync
        + Fft
        + rlst_blis::interface::gemm::Gemm,
    Complex<U>: Scalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree,
{
    fn s2t_scale(&self, level: u64) -> U {
        U::from(1.).unwrap()
    }
}

/// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
impl<T, U, V> SourceToTarget for NewKiFmm<V, SvdFieldTranslationKiFmm<U, T>, T, U>
where
    T: ScaleInvariantHomogenousKernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: Scalar<Real = U> + rlst_blis::interface::gemm::Gemm,
    U: Float + Default,
    U: std::marker::Send + std::marker::Sync + Default,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree,
{
    fn m2l(&self, level: u64) {}

    fn p2l(&self, level: u64) {}
}

impl<T, U, V> SourceToTargetHomogenousScaleInvariant<U>
    for NewKiFmm<V, SvdFieldTranslationKiFmm<U, T>, T, U>
where
    T: ScaleInvariantHomogenousKernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: Scalar<Real = U>
        + Float
        + Default
        + std::marker::Send
        + std::marker::Sync
        + Fft
        + rlst_blis::interface::gemm::Gemm,
    Complex<U>: Scalar,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree,
{
    fn s2t_scale(&self, level: u64) -> U {
        U::from(1.).unwrap()
    }
}

impl<T, U, V, W> NewFmm for NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    U: SourceToTargetData<V>,
    V: ScaleInvariantHomogenousKernel,
    W: Scalar<Real = W> + Default + Float,
    Self: SourceToTargetHomogenousScaleInvariant<W>,
{
    type Precision = W;

    fn evaluate_vec(&self, charges_vec: &[Self::Precision], result: &mut [Self::Precision]) {}

    fn evaluate_mat(&self, charges_mat: &[Self::Precision], result: &mut [Self::Precision]) {}

    fn get_expansion_order(&self) -> usize {
        self.expansion_order
    }

    fn get_ncoeffs(&self) -> usize {
        self.ncoeffs
    }
}
mod test {

    use bempp_field::types::FftFieldTranslationKiFmmNew;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::implementations::helpers::points_fixture;
    use rayon::result;
    use rlst_dense::traits::RawAccess;

    use super::*;

    #[test]
    fn test_builder() {
        let npoints = 1000;
        let sources = points_fixture::<f64>(npoints, None, None);
        let targets = points_fixture::<f64>(npoints, None, None);
        let mut result = vec![0.; npoints];
        let charges = vec![1.0; npoints];
        let n_crit = Some(100);
        let expansion_order = 5;
        let sparse = true;

        let fmm = KiFmmBuilderSingleNode::new()
            .tree(&sources.data(), &targets.data(), None, None)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::Value,
                FftFieldTranslationKiFmmNew::default(),
            )
            .unwrap()
            .build()
            .unwrap();

        fmm.evaluate_vec(&charges, &mut result);
    }
}
