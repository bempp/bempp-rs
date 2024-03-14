//! FMM traits
use rlst_dense::array::Array;
use rlst_dense::base_array::BaseArray;
use rlst_dense::data_container::VectorContainer;
use rlst_dense::types::RlstScalar;

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

/// Interface for FMM
pub trait Fmm {
    /// Precision of data associated with FMM
    type Precision: RlstScalar;

    /// Type of node index
    type NodeIndex;

    /// Type of tree, must implement FmmTree, allowing for separate source and target trees
    type Tree: FmmTree;

    /// Type of kernel associated with this FMM
    type Kernel: Kernel;

    /// Get the multipole expansion data associated with a node index as a slice
    /// # Arguments
    /// * `key` - The source node index.
    fn multipole(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]>;

    /// Get the local expansion data associated with a node index as a slice
    /// # Arguments
    /// * `key` - The target node index.
    fn local(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]>;

    /// Get the potential data associated with the particles contained at a given node
    /// # Arguments
    /// * `key` - The target leaf node index.
    fn potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>>;

    /// Get the expansion order associated with this FMM
    fn expansion_order(&self) -> usize;

    /// Get the tree associated with this FMM
    fn tree(&self) -> &Self::Tree;

    /// Get the kernel associated with this FMM
    fn kernel(&self) -> &Self::Kernel;

    /// Get the dimension of the data in this FMM
    fn dim(&self) -> usize;

    /// Evaluate the potentials, or potential gradients, for this FMM
    fn evaluate(&self);

    /// Clear the data buffers and add new charge data
    fn clear(&mut self, charges: &Array<
                Self::Precision,
                BaseArray<Self::Precision, VectorContainer<Self::Precision>, 2>,
                2,
            >);
}
