//! Data structures for kernel independent FMM
use std::collections::HashMap;

use bempp_traits::{field::SourceToTargetData, kernel::Kernel, tree::FmmTree, types::EvalType};
use bempp_tree::types::{domain::Domain, morton::MortonKey, single_node::SingleNodeTree};
use num::Float;
use rlst_dense::{
    array::Array, base_array::BaseArray, data_container::VectorContainer, types::RlstScalar,
};

use crate::tree::SingleNodeFmmTree;

/// Type alias to store charges corresponding to `nvecs`, in column major order such that the shape is `[ncharges, nvecs]`
pub type Charges<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Type alias for coordinate data stored in column major order such that the shape is `[ncoords, dim]`
pub type Coordinates<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Type alias for approximation of FMM operator matrices.
pub type C2EType<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// A threadsafe mutable raw pointer
#[derive(Clone, Debug, Copy)]
pub struct SendPtrMut<T> {
    /// Raw pointer
    pub raw: *mut T,
}

/// A threadsafe raw pointer
#[derive(Clone, Debug, Copy)]
pub struct SendPtr<T> {
    /// Raw pointer
    pub raw: *const T,
}

/// Holds all required data and metadata for evaluating a kernel independent FMM on a single node.
pub struct KiFmm<
    T: FmmTree<Tree = SingleNodeTree<W>>,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: RlstScalar<Real = W> + Float + Default,
> {
    /// A single node tree
    pub tree: T,

    /// The metadata required for source to target translation
    pub source_to_target_translation_data: U,

    /// The associated kernel function
    pub kernel: V,

    /// The expansion order of the FMM
    pub expansion_order: usize,

    /// The number of coefficients, corresponding to points discretising the equivalent surface
    pub ncoeffs: usize,

    /// The FMM evaluation type, either for a vector or matrix of input charges.
    pub fmm_eval_type: FmmEvalType,

    /// The kernel evaluation type, either for potentials or potentials and derivatives
    pub kernel_eval_type: EvalType,

    /// The metadata required for source to source translation
    pub source_translation_data_vec: Vec<C2EType<W>>,

    /// Set by the kernel evaluation type, either 1 or 4 corresponding to evaluating potentials or potentials and derivatives
    pub kernel_eval_size: usize,

    /// Index pointer for source coordinates
    pub charge_index_pointer_sources: Vec<(usize, usize)>,

    /// Index pointer for target coordinates
    pub charge_index_pointer_targets: Vec<(usize, usize)>,

    /// Dimension of the FMM
    pub dim: usize,

    /// Upward surfaces associated with source leaves
    pub leaf_upward_surfaces_sources: Vec<W>,

    /// Upward surfaces associated with target leaves
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

    /// The evaluated potentials at each target leaf box.
    pub potentials: Vec<W>,

    /// The evaluated potentials at each target leaf box.
    pub potentials_send_pointers: Vec<SendPtrMut<W>>,

    /// Leaf downward surfaces
    pub leaf_downward_surfaces: Vec<W>,

    /// The charge data at each target leaf box.
    pub charges: Vec<W>,

    /// Scales of each source leaf box
    pub leaf_scales_sources: Vec<W>,

    /// Global indices of each charge
    pub global_indices: Vec<usize>,
}

/// Dummy for KiFMM, simply used for implementing the trait that evaluated interactions directly
pub struct KiFmmDummy<T, U, V>
where
    T: FmmTree<Tree = SingleNodeTree<U>>,
    U: RlstScalar<Real = U> + Float + Default,
    V: Kernel<T = U> + Send + Sync,
{
    /// A single node tree
    pub tree: T,

    /// The charge data at each target leaf box.
    pub charges: Vec<U>,

    /// The evaluated potentials at each target leaf box.
    pub potentials: Vec<U>,

    /// The expansion order of the FMM
    pub expansion_order: usize,

    /// The associated kernel function
    pub kernel: V,

    /// The kernel evaluation type, either for potentials or potentials and derivatives
    pub kernel_eval_type: EvalType,

    /// The FMM evaluation type, either for a vector or matrix of input charges.
    pub fmm_eval_type: FmmEvalType,

    /// Set by the kernel evaluation type, either 1 or 4 corresponding to evaluating potentials or potentials and derivatives
    pub kernel_eval_size: usize,

    /// Index pointer for target coordinates
    pub charge_index_pointer_targets: Vec<(usize, usize)>,
}

/// Specifies the nature of the input to the FMM, either a vector or an matrix where each column corresponds to a vector of charges.
#[derive(Clone, Copy)]
pub enum FmmEvalType {
    /// Vector
    Vector,
    /// Matrix
    Matrix(usize),
}

/// Builder for kernel independent FMM object on a single node.
///
/// # Example
/// ```
/// use bempp_field::types::{BlasFieldTranslationKiFmm, FftFieldTranslationKiFmm};
/// use bempp_fmm::types::KiFmmBuilderSingleNode;
/// use bempp_kernel::laplace_3d::Laplace3dKernel;
/// use bempp_traits::fmm::Fmm;
/// use bempp_traits::tree::FmmTree;
/// use bempp_tree::implementations::helpers::points_fixture;
/// use rlst_dense::{rlst_dynamic_array2, traits::RawAccessMut};
///
/// /// Particle data
/// let nsources = 1000;
/// let ntargets = 2000;
/// let sources = points_fixture::<f64>(nsources, None, None, Some(0));
/// let targets = points_fixture::<f64>(ntargets, None, None, Some(3));
///
/// // FMM parameters
/// let n_crit = Some(150);
/// let expansion_order = 10;
/// let sparse = true;
///
/// /// Charge data
/// let nvecs = 1;
/// let tmp = vec![1.0; nsources * nvecs];
/// let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
/// charges.data_mut().copy_from_slice(&tmp);
///
/// /// Create a new builder, and attach a tree
/// let fmm = KiFmmBuilderSingleNode::new()
///     .tree(&sources, &targets, n_crit, sparse)
///     .unwrap();
///
/// /// Specify the FMM parameters, such as the kernel , the kernel evaluation mode, expansion order and charge data
/// let fmm = fmm
///     .parameters(
///         &charges,
///         expansion_order,
///         Laplace3dKernel::new(),
///         bempp_traits::types::EvalType::Value,
///         FftFieldTranslationKiFmm::new(),
///     )
///     .unwrap()
///     .build()
///     .unwrap();
/// ````
#[derive(Default)]
pub struct KiFmmBuilderSingleNode<T, U, V>
where
    T: SourceToTargetData<V>,
    U: RlstScalar<Real = U> + Float + Default,
    V: Kernel,
{
    /// Tree
    pub tree: Option<SingleNodeFmmTree<U>>,
    /// Charges
    pub charges: Option<Charges<U>>,
    /// Source to target
    pub source_to_target: Option<T>,
    /// Domain
    pub domain: Option<Domain<U>>,
    /// Kernel
    pub kernel: Option<V>,
    /// Expansion order
    pub expansion_order: Option<usize>,
    /// Number of coefficients
    pub ncoeffs: Option<usize>,
    /// Kernel eval type
    pub kernel_eval_type: Option<EvalType>,
    /// FMM eval type
    pub fmm_eval_type: Option<FmmEvalType>,
}
