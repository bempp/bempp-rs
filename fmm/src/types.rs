use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bempp_traits::{field::FieldTranslationData, fmm::Fmm, kernel::Kernel, tree::Tree};
use bempp_tree::types::{morton::MortonKey, point::Point};
use cauchy::Scalar;
use num::Float;
use rlst::dense::traits::*;
use rlst::dense::{base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix};
use rlst::{self};

/// Type alias for charge data
pub type Charge<T> = T;

/// Type alias for global index for identifying charge data with a point
pub type GlobalIdx = usize;

/// Type alias for mapping charge data to global indices.
pub type ChargeDict<T> = HashMap<GlobalIdx, Charge<T>>;

/// Type alias for multipole/local expansion containers.
pub type Expansions<T> = Matrix<T, BaseMatrix<T, VectorContainer<T>, Dynamic>, Dynamic>;

/// Type alias for potential containers.
pub type Potentials<T> = Matrix<T, BaseMatrix<T, VectorContainer<T>, Dynamic>, Dynamic>;

/// Type alias for approximation of FMM operator matrices.
pub type C2EType<T> = Matrix<T, BaseMatrix<T, VectorContainer<T>, Dynamic>, Dynamic>;

/// Type to store data associated with an FMM in.
pub struct FmmData<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
{
    /// The associated FMM object, which implements an FMM interface
    pub fmm: Arc<T>,

    /// The multipole expansion data at each box.
    pub multipoles: HashMap<MortonKey, Arc<Mutex<Expansions<U>>>>,

    /// The local expansion data at each box.
    pub locals: HashMap<MortonKey, Arc<Mutex<Expansions<U>>>>,

    /// The evaluated potentials at each leaf box.
    pub potentials: HashMap<MortonKey, Arc<Mutex<Potentials<U>>>>,

    /// The point data at each leaf box.
    pub points: HashMap<MortonKey, Vec<Point<U>>>,

    /// The charge data at each leaf box.
    pub charges: HashMap<MortonKey, Arc<Vec<U>>>,
}

/// Type to store data associated with the kernel independent (KiFMM) in.
pub struct KiFmm<T, U, V, W>
where
    T: Tree,
    U: Kernel<T = W>,
    V: FieldTranslationData<U>,
    W: Scalar + Float + Default,
{
    /// The expansion order
    pub order: usize,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_1: C2EType<W>,
    pub uc2e_inv_2: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_1: C2EType<W>,
    pub dc2e_inv_2: C2EType<W>,

    /// The ratio of the inner check surface diamater in comparison to the surface discretising a box.
    pub alpha_inner: W,

    /// The ratio of the outer check surface diamater in comparison to the surface discretising a box.
    pub alpha_outer: W,

    /// The multipole to multipole operator matrices, each index is associated with a child box (in sequential Morton order),
    pub m2m: Vec<C2EType<W>>,

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub l2l: Vec<C2EType<W>>,

    /// The tree (single or multi node) associated with this FMM
    pub tree: T,

    /// The kernel associated with this FMM.
    pub kernel: U,

    /// The M2L operator matrices, as well as metadata associated with this FMM.
    pub m2l: V,
}
