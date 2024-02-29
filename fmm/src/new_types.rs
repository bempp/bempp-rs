use std::collections::HashMap;

use bempp_traits::{field::{SourceToTarget, SourceToTargetData, SourceToTargetHomogenousScaleInvariant}, fmm::{NewFmm, SourceTranslation, TargetTranslation}, kernel::Kernel, tree::{FmmTree, Tree}};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};
use cauchy::Scalar;
use num::Float;

use crate::types::{C2EType, SendPtrMut};


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


impl <T, U, V, W>SourceTranslation for NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: Scalar + Default

{

    fn m2m(&self, level: u64) {}

    fn p2m(&self) {}
}


impl <T, U, V, W>TargetTranslation for NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: Scalar + Default

{
    fn l2l(&self, level: u64) {

    }

    fn l2p(&self) {

    }

    fn m2p(&self) {

    }

    fn p2p(&self) {

    }
}

impl <T, U, V, W> SourceToTarget for NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: Scalar + Default
{
    fn m2l(&self, level: u64) {}

    fn p2l(&self, level: u64) {}
}

impl <T, U, V, W> SourceToTargetHomogenousScaleInvariant<W> for NewKiFmm<T, U, V, W>
where
    T: FmmTree,
    U: SourceToTargetData<V>,
    V: Kernel,
    W: Scalar<Real = W> + Float + Default
{
    fn s2t_scale(&self, level: u64) -> W {
        W::from(1.).unwrap()
    }
}