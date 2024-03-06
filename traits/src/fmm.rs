//! FMM traits
use core::time;
use std::collections::HashMap;
use std::time::Duration;

use cauchy::Scalar;
use num::Float;

use crate::kernel::Kernel;
use crate::tree::{FmmTree, Tree};
use crate::types::EvalType;

/// Interface for source box translations.
pub trait SourceTranslation {
    /// Particle to multipole translations, applied at leaf level.
    fn p2m(&self);

    /// Multipole to multipole translations, applied during upward pass, level by level.
    ///
    /// # Arguments
    /// * `level` - The child level at which this kernel is being applied.
    fn m2m(&self, level: u64);
}

/// Interface for target box translations.
pub trait TargetTranslation {
    /// Local to local translations, applied during downward pass.
    ///
    /// # Arguments
    /// * `level` - The child level at which this kernel is being applied.
    fn l2l(&self, level: u64);

    /// Multipole to particle translations, applies to leaf boxes when a source box is within
    /// the near field of a target box, but is small enough that a multipole expansion converges
    /// at the target box.
    fn m2p(&self);

    /// Local to particle translations, applies the local expansion accumulated at each leaf box to the
    /// target particles it contains.
    fn l2p(&self);

    /// Near field particle to particle (direct) potential contributions to particles in a given leaf box's
    /// near field where the `p2l` and `m2p` do not apply.
    fn p2p(&self);
}

pub trait Fmm {
    type Precision;
    type NodeIndex;
    type Tree: FmmTree;
    type Kernel: Kernel;

    fn get_multipole(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]>;
    fn get_local(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]>;
    fn get_potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>>;

    fn get_expansion_order(&self) -> usize;
    fn get_tree(&self) -> &Self::Tree;
    fn get_kernel(&self) -> &Self::Kernel;
    fn evaluate(&self);
}

/// Interface to compute interaction lists given a tree.
pub trait InteractionLists {
    type Tree: Tree;

    /// The interaction list defining multipole to local translations, i.e. for well separated boxes.
    ///
    /// # Arguments
    /// * `key` - The target key for which this interaction list is being calculated.
    fn get_v_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;

    /// The interaction list defining particle to local translations, i.e. where the box in the in
    /// the interaction list is too large for the multipole expansion to apply at the target box
    /// specified by `key`.
    ///
    /// # Arguments
    /// * `key` - The target key for which this interaction list is being calculated.
    fn get_x_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;

    /// The interaction list defining multiopole to particle translations, i.e. where the multipole
    /// expansion of the source key applies at the target key, only applies to leaf nodes.
    ///
    /// # Arguments
    /// * `leaf` - The target leaf key for which this interaction list is being calculated.
    fn get_w_list(
        &self,
        leaf: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;

    /// The interaction list defining the near field of each leaf box, i.e. adjacent boxes.
    ///
    /// # Arguments
    /// * `key` - The target key for which this interaction list is being calculated.
    fn get_u_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
}
