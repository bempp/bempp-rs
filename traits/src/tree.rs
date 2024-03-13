//! Traits
use std::collections::HashSet;
use std::hash::Hash;

use num::Float;
use rlst_dense::types::RlstScalar;

/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
/// This trait makes no assumptions about the downstream usage of a struct implementing Tree,
/// it simply provides methods for accessing tree nodes, and associated data, and is generically
/// defined for both single and multi-node settings.
pub trait Tree {
    /// The computational domain defined by the tree.
    type Domain;

    type Precision: RlstScalar<Real = Self::Precision> + Float + Default;

    /// A tree node.
    type Node: TreeNode<Self::Precision, Domain = Self::Domain> + Clone + Copy;

    /// Slice of nodes.
    type NodeSlice<'a>: IntoIterator<Item = &'a Self::Node>
    where
        Self: 'a;

    /// Copy of nodes
    type Nodes: IntoIterator<Item = Self::Node>;

    fn nleaves(&self) -> Option<usize>;
    fn nkeys_tot(&self) -> Option<usize>;
    fn nkeys(&self, level: u64) -> Option<usize>;

    /// Get depth of tree.
    fn depth(&self) -> u64;

    /// Get a reference to all leaves, gets local keys in multi-node setting.
    fn all_leaves(&self) -> Option<Self::NodeSlice<'_>>;

    /// Get a reference to keys at a given level, gets local keys in a multi-node setting.
    fn keys(&self, level: u64) -> Option<Self::NodeSlice<'_>>;

    /// Get a reference to all keys, gets local keys in a multi-node setting.
    fn all_keys(&self) -> Option<Self::NodeSlice<'_>>;

    /// Get a reference to all keys as a set, gets local keys in a multi-node setting.
    fn all_keys_set(&self) -> Option<&'_ HashSet<Self::Node>>;

    /// Get a reference to all leaves as a set, gets local keys in a multi-node setting.
    fn all_leaves_set(&self) -> Option<&'_ HashSet<Self::Node>>;

    /// Gets a reference to the coordinates contained with a leaf node.
    fn coordinates<'a>(&'a self, key: &Self::Node) -> Option<&'a [Self::Precision]>;

    /// Gets a reference to the coordinates contained in across tree (local in multinode setting)
    fn all_coordinates(&self) -> Option<&[Self::Precision]>;

    /// Gets global indices at a leaf (local in multinode setting)
    fn global_indices<'a>(&'a self, key: &Self::Node) -> Option<&'a [usize]>;

    /// Gets all global indices (local in multinode setting)
    fn all_global_indices(&self) -> Option<&[usize]>;

    /// Get domain defined by the points, gets global domain in multi-node setting.
    fn domain(&self) -> &'_ Self::Domain;

    /// Get a map from the key to index position in sorted keys
    fn index(&self, key: &Self::Node) -> Option<&usize>;

    fn node(&self, idx: usize) -> Option<&Self::Node>;

    /// Get a map from the key to leaf index position in sorted leaves
    fn leaf_index(&self, key: &Self::Node) -> Option<&usize>;
}

pub trait FmmTree {
    type Precision;
    type Node;

    type Tree: Tree<Precision = Self::Precision, Node = Self::Node>;

    fn source_tree(&self) -> &Self::Tree;

    fn target_tree(&self) -> &Self::Tree;

    fn domain(&self) -> &<Self::Tree as Tree>::Domain;

    fn near_field(&self, leaf: &Self::Node) -> Option<Vec<Self::Node>>;
}

pub trait TreeNode<T>
where
    Self: Hash + Eq,
    T: RlstScalar,
{
    type Domain;

    // Copy of nodes
    type Nodes: IntoIterator<Item = Self>;

    /// The parent of a key.
    fn parent(&self) -> Self;

    fn level(&self) -> u64;

    fn compute_surface(
        &self,
        domain: &Self::Domain,
        expansion_order: usize,
        alpha: T,
    ) -> Vec<<T as RlstScalar>::Real>;

    /// Neighbours defined by keys sharing a vertex, edge, or face.
    fn neighbors(&self) -> Self::Nodes;

    /// Childen of a key.
    fn children(&self) -> Self::Nodes;

    /// Checks adjacency, defined by sharing a vertex, edge, or face, between two keys.
    fn is_adjacent(&self, other: &Self) -> bool;
}
