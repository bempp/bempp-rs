//! FMM traits
use core::time;
use std::collections::HashMap;
use std::time::Duration;

use cauchy::Scalar;
use num::Float;

use crate::kernel::Kernel;
use crate::tree::Tree;
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

/// Interface for an FMM algorithm, this is uniquely specified by the type of tree
/// (single/multi node), the order of expansions being used, as well as the kernel function
/// being evaluated.
pub trait Fmm {
    type Kernel: Kernel;
    type Tree: Tree;

    /// Expansion order.
    fn order(&self) -> usize;

    /// Kernel function
    fn kernel(&self) -> &Self::Kernel;

    /// Associated tree
    fn tree(&self) -> &Self::Tree;
}

pub trait NewFmm {
    type Precision: Scalar + Default + Float;
    fn evaluate(&self, result: &mut [Self::Precision], profile: bool) -> Option<HashMap<String, Duration>>;
    fn get_expansion_order(&self) -> usize;
    fn get_ncoeffs(&self) -> usize;
}

/// Dictionary containing timings
pub type TimeDict = HashMap<String, u128>;

/// Interface for running the FMM loop.
pub trait FmmLoop {
    /// Compute the upward pass, optionally collect timing for each operator.
    ///
    /// # Arguments
    /// `time` - If true, method returns a dictionary of times for the downward pass operators.
    fn upward_pass(&self, time: bool) -> Option<TimeDict>;

    /// Compute the downward pass, optionally collect timing for each operator.
    ///
    /// # Arguments
    /// `time` - If true, method returns a dictionary of times for the upward pass operators.
    fn downward_pass(&self, time: bool) -> Option<TimeDict>;

    /// Compute the upward and downward pass, optionally collect timing for each operator.
    ///
    /// # Arguments
    /// `time` - If true, method returns a dictionary of times for all operators.
    fn run(&self, time: bool) -> Option<TimeDict>;
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
