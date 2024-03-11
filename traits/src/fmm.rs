//! FMM traits
use crate::kernel::Kernel;
use crate::tree::FmmTree;

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

    fn multipole(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]>;
    fn local(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]>;
    fn potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>>;
    fn expansion_order(&self) -> usize;
    fn tree(&self) -> &Self::Tree;
    fn kernel(&self) -> &Self::Kernel;
    fn dim(&self) -> usize;
    fn evaluate(&self);
}
