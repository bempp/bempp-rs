//! FMM traits
use std::collections::HashMap;

use crate::kernel::Kernel;
use crate::tree::Tree;

/// Interface for source box translations.
pub trait SourceTranslation {
    /// Point to multipole translations, applied at leaf level.
    fn p2m(&self);

    /// Multipole to multipole translations, applied during upward pass, level by level.
    fn m2m(&self, level: u64);
}

/// Interface for target box translations.
pub trait TargetTranslation {
    /// Local to local translations, applied during downward pass.
    fn l2l(&self, level: u64);

    /// Multipole to particle translations, applies to leaf boxes when a source box is within
    /// the near field of a target box, but is small enough that a multipole expansion converges
    /// at the target box.
    fn m2p(&self);

    /// Particle to local translations, applies to leaf boxes when a source box is within
    /// the near field of a target box, but is too large for the multipole expansion to converge
    /// at the target, so instead its contribution is computed directly.
    fn p2l(&self);

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

/// Dictionary containing timings
pub type TimeDict = HashMap<String, u128>;

/// Interface for running the FMM loop.
pub trait FmmLoop {
    /// Compute the upward pass, optionally collect timing for each operator.
    ///
    /// # Arguments
    /// `time` - If true, method returns a dictionary of times for the downward pass operators.
    fn upward_pass(&self, time: Option<bool>) -> Option<TimeDict>;

    /// Compute the downward pass, optionally collect timing for each operator.
    ///
    /// # Arguments
    /// `time` - If true, method returns a dictionary of times for the upward pass operators.
    fn downward_pass(&self, time: Option<bool>) -> Option<TimeDict>;

    /// Compute the upward and downward pass, optionally collect timing for each operator.
    ///
    /// # Arguments
    /// `time` - If true, method returns a dictionary of times for all operators.
    fn run(&self, time: Option<bool>) -> Option<TimeDict>;
}

/// Interface to compute interaction lists given a tree.
pub trait InteractionLists {
    type Tree: Tree;

    /// The interaction list defining multipole to local translations.
    fn get_v_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;

    /// The interaction list defining particle to local translations.
    fn get_x_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;

    /// The interaction list defining multiopole to particle translations.
    fn get_w_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;

    /// The interaction list defining the near field of each leaf box.
    fn get_u_list(
        &self,
        key: &<Self::Tree as Tree>::NodeIndex,
    ) -> Option<<Self::Tree as Tree>::NodeIndices>;
}
