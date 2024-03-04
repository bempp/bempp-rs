use std::{char, collections::HashMap, thread::LocalKey};

use bempp_field::{
    fft::Fft,
    types::{FftFieldTranslationKiFmm, FftFieldTranslationKiFmmNew, SvdFieldTranslationKiFmm},
};
use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData, SourceToTargetHomogenousScaleInvariant},
    fmm::{Fmm, NewFmm, SourceTranslation, TargetTranslation},
    kernel::{self, HomogenousKernel, Kernel},
    tree::{FmmTree, MortonKeyInterface, Tree},
    types::EvalType,
};
use bempp_tree::{
    constants::ROOT,
    types::{
        domain::Domain,
        morton::MortonKey,
        single_node::{SingleNodeTree, SingleNodeTreeNew},
    },
};
use cauchy::Scalar;
use num::{traits::real::Real, Complex, Float};
use rlst_blis::interface::gemm::Gemm;
use rlst_dense::{
    array::{empty_array, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array2,
    traits::{MatrixSvd, MultIntoResize, RawAccess, RawAccessMut, Shape},
};

use crate::{
    charge::{Charges, Coordinates},
    constants::{ALPHA_INNER, ALPHA_OUTER},
    field_translation::target,
    pinv::pinv,
    types::{C2EType, SendPtr, SendPtrMut},
};

pub enum FmmInputType {
    Vector,
    Matrix(usize),
}

pub fn ncoeffs(expansion_order: usize) -> usize {
    6 * (expansion_order - 1).pow(2) + 2
}

/// Combines the old datatree + Fmm structs into a single storage of metadata
pub struct NewKiFmm<T: FmmTree, U: SourceToTargetData<V>, V: Kernel, W: Scalar + Default> {
    pub tree: T,
    pub source_to_target_data: U,
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

#[derive(Default)]
pub struct SingleNodeFmmTree<T: Float + Default + Scalar<Real = T>> {
    pub source_tree: SingleNodeTreeNew<T>,
    pub target_tree: SingleNodeTreeNew<T>,
    pub domain: Domain<T>,
}

#[derive(Default)]
pub struct KiFmmBuilderSingleNode<'builder, T, U, V>
where
    T: SourceToTargetData<V>,
    U: Float + Default + Scalar<Real = U>,
    V: Kernel + HomogenousKernel,
{
    tree: Option<SingleNodeFmmTree<U>>,
    charges: Option<&'builder Charges<U>>,
    source_to_target: Option<T>,
    domain: Option<Domain<U>>,
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

    fn get_domain(&self) -> &<Self::Tree as Tree>::Domain {
        &self.domain
    }
}

impl<'builder, T, U, V> KiFmmBuilderSingleNode<'builder, T, U, V>
where
    T: SourceToTargetData<V, Domain = Domain<U>> + Default,
    // U: Float + Scalar<Real = U> + Default,
    U: Scalar<Real = U> + rlst_blis::interface::gemm::Gemm,
    U: Float + Default,
    U: std::marker::Send + std::marker::Sync + Default,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: Kernel<T = U> + HomogenousKernel + Clone + Default,
{
    // Start building with mandatory parameters
    pub fn new() -> Self {
        KiFmmBuilderSingleNode {
            tree: None,
            domain: None,
            source_to_target: None,
            kernel: None,
            expansion_order: None,
            ncoeffs: None,
            eval_type: None,
            charges: None,
        }
    }

    pub fn tree(
        mut self,
        sources: &Coordinates<U>,
        targets: &Coordinates<U>,
        charges: &'builder Charges<U>,
        n_crit: Option<u64>,
        sparse: Option<bool>,
    ) -> Self {
        // Source and target trees calcualted over the same domain
        let source_domain = Domain::from_local_points(sources.data());
        let target_domain = Domain::from_local_points(targets.data());

        // Calculate union of domains for source and target points, needed to define operators
        let domain = source_domain.union(&target_domain);
        self.domain = Some(domain);

        let source_tree = SingleNodeTreeNew::new(sources.data(), n_crit, sparse, self.domain);
        let target_tree = SingleNodeTreeNew::new(targets.data(), n_crit, sparse, self.domain);

        let fmm_tree = SingleNodeFmmTree {
            source_tree,
            target_tree,
            domain,
        };

        self.tree = Some(fmm_tree);
        self.charges = Some(charges);

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

            // Set source to target metadata
            // Set the expansion order
            source_to_target.set_expansion_order(self.expansion_order.unwrap());

            // Set the associated kernel
            let kernel = self.kernel.as_ref().unwrap().clone();
            source_to_target.set_kernel(kernel);

            // Compute the field translation operators
            source_to_target.set_operator_data(self.expansion_order.unwrap(), self.domain.unwrap());

            self.source_to_target = Some(source_to_target);

            Ok(self)
        }
    }

    // Finalize and build the KiFmm
    pub fn build(self) -> Result<NewKiFmm<SingleNodeFmmTree<U>, T, V, U>, String> {
        if self.tree.is_none() || self.source_to_target.is_none() || self.expansion_order.is_none()
        {
            Err("Missing fields for constructing KiFmm".to_string())
        } else {
            // Configure with tree, expansion parameters and source to target field translation operators
            let mut result = NewKiFmm {
                tree: self.tree.unwrap(),
                expansion_order: self.expansion_order.unwrap(),
                ncoeffs: self.ncoeffs.unwrap(),
                kernel: self.kernel.unwrap(),
                source_to_target_data: self.source_to_target.unwrap(),
                ..Default::default()
            };

            // Compute the source to source and target to target field translation operators
            result.set_source_and_target_operator_data();

            result.set_metadata(self.eval_type.unwrap(), self.charges.unwrap());

            Ok(result)
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
    T: HomogenousKernel<T = U> + std::marker::Send + std::marker::Sync + Default,
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

/// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
impl<T, U, V> SourceToTarget for NewKiFmm<V, SvdFieldTranslationKiFmm<U, T>, T, U>
where
    T: HomogenousKernel<T = U> + std::marker::Send + std::marker::Sync + Default,
    U: Scalar<Real = U> + rlst_blis::interface::gemm::Gemm,
    U: Float + Default,
    U: std::marker::Send + std::marker::Sync + Default,
    Array<U, BaseArray<U, VectorContainer<U>, 2>, 2>: MatrixSvd<Item = U>,
    V: FmmTree,
{
    fn m2l(&self, level: u64) {}

    fn p2l(&self, level: u64) {}
}

// impl<T, U, V> SourceToTargetHomogenousScaleInvariant<U>
//     for NewKiFmm<V, SvdFieldTranslationKiFmm<U, T>, T, U>
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

impl<T, U, V, W> NewFmm for NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    U: SourceToTargetData<V>,
    V: HomogenousKernel,
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

impl<T, U, V, W> Default for NewKiFmm<T, U, V, W>
where
    T: FmmTree + Default,
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
        NewKiFmm {
            tree: T::default(),
            source_to_target_data: U::default(),
            kernel: V::default(),
            expansion_order: 0,
            ncoeffs: 0,
            uc2e_inv_1,
            uc2e_inv_2,
            dc2e_inv_1,
            dc2e_inv_2,
            source_data: source,
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

impl<T, U, V, W> NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    T::Tree: Tree<Domain = Domain<W>, Precision = W, NodeIndex = MortonKey>,
    U: SourceToTargetData<V>,
    V: HomogenousKernel<T = W>,
    W: Scalar<Real = W> + Default + Float + rlst_blis::interface::gemm::Gemm,
    Array<W, BaseArray<W, VectorContainer<W>, 2>, 2>: MatrixSvd<Item = W>,
    // Self: SourceToTargetHomogenousScaleInvariant<W>
{
    pub fn set_source_and_target_operator_data(&mut self) {
        // Cast surface parameters
        let alpha_outer = W::from(ALPHA_OUTER).unwrap();
        let alpha_inner = W::from(ALPHA_INNER).unwrap();
        let domain = self.tree.get_domain();

        // Compute required surfaces
        let upward_equivalent_surface =
            ROOT.compute_surface(domain, self.expansion_order, alpha_inner);
        let upward_check_surface = ROOT.compute_surface(domain, self.expansion_order, alpha_outer);
        let downward_equivalent_surface =
            ROOT.compute_surface(domain, self.expansion_order, alpha_outer);
        let downward_check_surface =
            ROOT.compute_surface(domain, self.expansion_order, alpha_inner);

        let nequiv_surface = upward_equivalent_surface.len() / self.kernel.space_dimension();
        let ncheck_surface = upward_check_surface.len() / self.kernel.space_dimension();

        // Assemble matrix of kernel evaluations between upward check to equivalent, and downward check to equivalent matrices
        // As well as estimating their inverses using GESVD
        let mut uc2e_t = rlst_dynamic_array2!(W, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &upward_equivalent_surface[..],
            &upward_check_surface[..],
            uc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut uc2e = rlst_dynamic_array2!(W, [nequiv_surface, ncheck_surface]);
        uc2e.fill_from(uc2e_t.transpose());

        let mut dc2e_t = rlst_dynamic_array2!(W, [ncheck_surface, nequiv_surface]);
        self.kernel.assemble_st(
            EvalType::Value,
            &downward_equivalent_surface[..],
            &downward_check_surface[..],
            dc2e_t.data_mut(),
        );

        // Need to transpose so that rows correspond to targets and columns to sources
        let mut dc2e = rlst_dynamic_array2!(W, [nequiv_surface, ncheck_surface]);
        dc2e.fill_from(dc2e_t.transpose());

        let (s, ut, v) = pinv::<W>(&uc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(W, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = W::from_real(s[i]);
        }
        let uc2e_inv_1 = empty_array::<W, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let uc2e_inv_2 = ut;

        let (s, ut, v) = pinv::<W>(&dc2e, None, None).unwrap();

        let mut mat_s = rlst_dynamic_array2!(W, [s.len(), s.len()]);
        for i in 0..s.len() {
            mat_s[[i, i]] = W::from_real(s[i]);
        }

        let dc2e_inv_1 = empty_array::<W, 2>().simple_mult_into_resize(v.view(), mat_s.view());
        let dc2e_inv_2 = ut;

        // Calculate M2M and L2L operator matrices
        let children = ROOT.children();
        let mut m2m = rlst_dynamic_array2!(W, [nequiv_surface, 8 * nequiv_surface]);
        let mut l2l = Vec::new();

        for (i, child) in children.iter().enumerate() {
            let child_upward_equivalent_surface =
                child.compute_surface(domain, self.expansion_order, alpha_inner);
            let child_downward_check_surface =
                child.compute_surface(domain, self.expansion_order, alpha_inner);

            let mut pc2ce_t = rlst_dynamic_array2!(W, [ncheck_surface, nequiv_surface]);

            self.kernel.assemble_st(
                EvalType::Value,
                &child_upward_equivalent_surface,
                &upward_check_surface,
                pc2ce_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut pc2ce = rlst_dynamic_array2!(W, [nequiv_surface, ncheck_surface]);
            pc2ce.fill_from(pc2ce_t.transpose());

            let tmp = empty_array::<W, 2>().simple_mult_into_resize(
                uc2e_inv_1.view(),
                empty_array::<W, 2>().simple_mult_into_resize(uc2e_inv_2.view(), pc2ce.view()),
            );
            let l = i * nequiv_surface * nequiv_surface;
            let r = l + nequiv_surface * nequiv_surface;

            m2m.data_mut()[l..r].copy_from_slice(tmp.data());

            let mut cc2pe_t = rlst_dynamic_array2!(W, [ncheck_surface, nequiv_surface]);

            self.kernel.assemble_st(
                EvalType::Value,
                &downward_equivalent_surface,
                &child_downward_check_surface,
                cc2pe_t.data_mut(),
            );

            // Need to transpose so that rows correspond to targets, and columns to sources
            let mut cc2pe = rlst_dynamic_array2!(W, [nequiv_surface, ncheck_surface]);
            cc2pe.fill_from(cc2pe_t.transpose());
            let mut tmp = empty_array::<W, 2>().simple_mult_into_resize(
                dc2e_inv_1.view(),
                empty_array::<W, 2>().simple_mult_into_resize(dc2e_inv_2.view(), cc2pe.view()),
            );
            tmp.data_mut()
                .iter_mut()
                .for_each(|d| *d *= self.kernel.scale(child.level()));

            l2l.push(tmp);
        }

        self.source_data = m2m;
        self.target_data = l2l;
        self.dc2e_inv_1 = dc2e_inv_1;
        self.dc2e_inv_2 = dc2e_inv_2;
        self.uc2e_inv_1 = uc2e_inv_1;
        self.uc2e_inv_2 = uc2e_inv_2;
    }

    fn set_metadata(&mut self, eval_type: EvalType, charges: &Charges<W>) {
        let dim = self.kernel.space_dimension();
        let alpha_outer = W::from(ALPHA_OUTER).unwrap();
        let alpha_inner = W::from(ALPHA_INNER).unwrap();

        // Check if computing potentials, or potentials and derivatives
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => dim + 1,
        };

        // Check if we are computing matvec or matmul
        let [_, nevals] = charges.shape();

        let ntarget_points = self
            .tree
            .get_target_tree()
            .get_all_coordinates()
            .unwrap()
            .len();

        let nsource_keys = self.tree.get_source_tree().get_nkeys().unwrap();
        let ntarget_keys = self.tree.get_target_tree().get_nkeys().unwrap();
        let ntarget_leaves = self.tree.get_target_tree().get_nleaves().unwrap();
        let nsource_leaves = self.tree.get_source_tree().get_nleaves().unwrap();

        // Buffers to store all multipole and local data
        let multipoles = vec![W::default(); self.ncoeffs * nsource_keys * nevals];
        let locals = vec![W::default(); self.ncoeffs * ntarget_keys * nevals];

        // Mutable pointers to multipole and local data, indexed by level
        let mut level_multipoles = vec![
            Vec::new();
            (self.tree.get_source_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];
        let mut level_locals = vec![
            Vec::new();
            (self.tree.get_target_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];

        // Index pointers of multipole and local data, indexed by level
        let mut level_index_pointer_multipoles = vec![
            HashMap::new();
            (self.tree.get_source_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];
        let mut level_index_pointer_locals = vec![
            HashMap::new();
            (self.tree.get_target_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];

        // Mutable pointers to multipole and local data only at leaf level
        let mut leaf_multipoles = vec![Vec::new(); nsource_leaves];
        let mut leaf_locals = vec![Vec::new(); ntarget_leaves];

        // Buffer to store evaluated potentials and/or gradients at target points
        let mut potentials = vec![W::default(); ntarget_points * eval_size * nevals];

        // Mutable pointers to potential data at each target leaf
        let mut potentials_send_pointers = vec![SendPtrMut::default(); ntarget_leaves * nevals];

        // Index pointer of charge data at each target leaf
        let mut charge_index_pointer = vec![(0usize, 0usize); ntarget_leaves];

        // Kernel scale at each target and source leaf
        let mut target_leaf_scales = vec![W::default(); ntarget_leaves * self.ncoeffs * nevals];
        let mut source_leaf_scales = vec![W::default(); nsource_leaves * self.ncoeffs * nevals];

        // Pre compute check surfaces
        let mut upward_surfaces = vec![W::default(); self.ncoeffs * nsource_keys * dim];
        let mut downward_surfaces = vec![W::default(); self.ncoeffs * ntarget_keys * dim];
        let mut leaf_upward_surfaces = vec![W::default(); self.ncoeffs * nsource_leaves * dim];
        let mut leaf_downward_surfaces = vec![W::default(); self.ncoeffs * ntarget_leaves * dim];

        // Create mutable pointers to multipole and local data indexed by tree level
        {
            for level in 0..=self.tree.get_source_tree().get_depth() {
                let mut tmp_multipoles = Vec::new();

                let keys = self.tree.get_source_tree().get_keys(level).unwrap();
                for key in keys.into_iter() {
                    let &key_idx = self.tree.get_source_tree().get_index(key).unwrap();
                    let key_displacement = self.ncoeffs * nevals * key_idx;
                    let mut key_multipoles = Vec::new();
                    for eval_idx in 0..nevals {
                        let eval_displacement = self.ncoeffs * eval_idx;
                        let raw = unsafe {
                            multipoles
                                .as_ptr()
                                .add(key_displacement + eval_displacement)
                                as *mut W
                        };
                        key_multipoles.push(SendPtrMut { raw });
                    }
                    tmp_multipoles.push(key_multipoles)
                }
                level_multipoles[level as usize] = tmp_multipoles
            }

            for level in 0..=self.tree.get_target_tree().get_depth() {
                let mut tmp_locals = Vec::new();

                let keys = self.tree.get_target_tree().get_keys(level).unwrap();
                for key in keys.into_iter() {
                    let &key_idx = self.tree.get_target_tree().get_index(key).unwrap();
                    let key_displacement = self.ncoeffs * nevals * key_idx;
                    let mut key_locals = Vec::new();
                    for eval_idx in 0..nevals {
                        let eval_displacement = self.ncoeffs * eval_idx;
                        let raw = unsafe {
                            locals.as_ptr().add(key_displacement + eval_displacement) as *mut W
                        };
                        key_locals.push(SendPtrMut { raw });
                    }
                    tmp_locals.push(key_locals)
                }
                level_locals[level as usize] = tmp_locals
            }

            for level in 0..=self.tree.get_source_tree().get_depth() {
                let keys = self.tree.get_source_tree().get_keys(level).unwrap();
                for (level_idx, key) in keys.into_iter().enumerate() {
                    level_index_pointer_multipoles[level as usize].insert(*key, level_idx);
                }
            }

            for level in 0..=self.tree.get_target_tree().get_depth() {
                let keys = self.tree.get_target_tree().get_keys(level).unwrap();
                for (level_idx, key) in keys.into_iter().enumerate() {
                    level_index_pointer_locals[level as usize].insert(*key, level_idx);
                }
            }
        }

        // Create mutable pointers to multipole and local data at leaf level
        {
            for (leaf_idx, leaf) in self
                .tree
                .get_source_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let key_idx = self.tree.get_source_tree().get_index(leaf).unwrap();
                let key_displacement = self.ncoeffs * nevals * key_idx;
                for eval_idx in 0..nevals {
                    let eval_displacement = self.ncoeffs * eval_idx;
                    let raw = unsafe {
                        multipoles
                            .as_ptr()
                            .add(eval_displacement + key_displacement)
                            as *mut W
                    };

                    leaf_multipoles[leaf_idx].push(SendPtrMut { raw });
                }
            }

            for (leaf_idx, leaf) in self
                .tree
                .get_target_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let key_idx = self.tree.get_target_tree().get_index(leaf).unwrap();
                let key_displacement = self.ncoeffs * nevals * key_idx;
                for eval_idx in 0..nevals {
                    let eval_displacement = self.ncoeffs * eval_idx;
                    let raw = unsafe {
                        locals.as_ptr().add(eval_displacement + key_displacement) as *mut W
                    };
                    leaf_locals[leaf_idx].push(SendPtrMut { raw });
                }
            }
        }

        // Set index pointers for evaluated potentials
        {
            let mut index_pointer = 0;
            let mut potential_raw_pointer = potentials.as_mut_ptr();
            let mut potential_raw_pointers = Vec::new();
            for eval_idx in 0..nevals {
                let ptr = unsafe {
                    potentials
                        .as_mut_ptr()
                        .add(eval_idx * ntarget_points * (dim + 1))
                };
                potential_raw_pointers.push(ptr)
            }

            for (i, leaf) in self
                .tree
                .get_target_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs;
                let r = l + self.ncoeffs;
                target_leaf_scales[l..r].copy_from_slice(
                    vec![self.kernel.scale(leaf.level()); self.ncoeffs].as_slice(),
                );

                let npoints;
                let nevals;

                if let Some(coordinates) = self.tree.get_target_tree().get_coordinates(leaf) {
                    npoints = coordinates.len() / dim;
                    nevals = npoints * eval_size;
                } else {
                    npoints = 0;
                    nevals = 0;
                }

                for j in 0..nevals {
                    potentials_send_pointers[ntarget_leaves * j + i] = SendPtrMut {
                        raw: potential_raw_pointers[j],
                    }
                }

                potentials_send_pointers[i] = SendPtrMut {
                    raw: potential_raw_pointer,
                };

                // Update charge index pointer
                let bounds_points = (index_pointer, index_pointer + npoints);
                charge_index_pointer[i] = bounds_points;
                index_pointer += npoints;

                // Update raw pointer with number of points at this leaf
                for ptr in potential_raw_pointers.iter_mut() {
                    *ptr = unsafe { ptr.add(nevals) }
                }
            }

            for (i, leaf) in self
                .tree
                .get_source_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                // Assign scales
                let l = i * self.ncoeffs;
                let r = l + self.ncoeffs;
                source_leaf_scales[l..r].copy_from_slice(
                    vec![self.kernel.scale(leaf.level()); self.ncoeffs].as_slice(),
                );
            }
        }

        // Compute surfaces
        {
            // All upward and downward surfaces
            for (i, key) in self
                .tree
                .get_source_tree()
                .get_all_keys()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs * dim;
                let r = l + self.ncoeffs * dim;
                let upward_surface =
                    key.compute_surface(self.tree.get_domain(), self.expansion_order, alpha_outer);

                upward_surfaces[l..r].copy_from_slice(&upward_surface);
            }

            for (i, key) in self
                .tree
                .get_target_tree()
                .get_all_keys()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs * dim;
                let r = l + self.ncoeffs * dim;
                let downward_surface =
                    key.compute_surface(self.tree.get_domain(), self.expansion_order, alpha_outer);

                downward_surfaces[l..r].copy_from_slice(&downward_surface);
            }

            // Leaf upward and downward surfaces
            for (i, key) in self
                .tree
                .get_source_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs * dim;
                let r = l + self.ncoeffs * dim;
                let upward_surface =
                    key.compute_surface(self.tree.get_domain(), self.expansion_order, alpha_outer);

                leaf_upward_surfaces[l..r].copy_from_slice(&upward_surface);
            }

            for (i, key) in self
                .tree
                .get_target_tree()
                .get_all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let l = i * self.ncoeffs * dim;
                let r = l + self.ncoeffs * dim;
                let downward_surface =
                    key.compute_surface(self.tree.get_domain(), self.expansion_order, alpha_inner);

                leaf_downward_surfaces[l..r].copy_from_slice(&downward_surface);
            }
        }

        // Set data
        {
            self.multipoles = multipoles;
            self.locals = locals;
            self.leaf_multipoles = leaf_multipoles;
            self.level_multipoles = level_multipoles;
            self.leaf_locals = leaf_locals;
            self.level_locals = level_locals;
            self.level_index_pointer_locals = level_index_pointer_locals;
            self.level_index_pointer_multipoles = level_index_pointer_multipoles;
            self.potentials = potentials;
            self.potentials_send_pointers = potentials_send_pointers;
            self.upward_surfaces = upward_surfaces;
            self.downward_surfaces = downward_surfaces;
            self.leaf_upward_surfaces = leaf_upward_surfaces;
            self.leaf_downward_surfaces = leaf_downward_surfaces;
            self.charge_index_pointer = charge_index_pointer;
            self.target_scales = target_leaf_scales;
            self.source_scales = source_leaf_scales;
        }
    }
}

mod test {

    use bempp_field::types::FftFieldTranslationKiFmmNew;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::implementations::helpers::points_fixture;
    use rlst_dense::{rlst_array_from_slice2, traits::RawAccess};

    use super::*;

    #[test]
    fn test_builder() {
        let npoints = 1000;
        let nvecs = 1;
        let sources = points_fixture::<f64>(npoints, None, None);
        let targets = points_fixture::<f64>(npoints, None, None);
        let mut result = vec![0.; npoints];
        let tmp = vec![1.0; npoints * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [npoints, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let n_crit = Some(100);
        let expansion_order = 5;
        let sparse = true;

        let fmm = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, &charges, None, None)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                EvalType::ValueDeriv,
                FftFieldTranslationKiFmmNew::default(),
            )
            .unwrap()
            .build()
            .unwrap();

        // fmm.evaluate_vec(&charges, &mut result);
    }
}
