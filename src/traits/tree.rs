//! Traits
use std::{collections::HashSet, hash::Hash};

use num::Float;
use rlst::RlstScalar;

/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
/// This trait makes no assumptions about the downstream usage of a struct implementing Tree,
/// it simply provides methods for accessing tree nodes, and associated data, and is generically
/// defined for both single and multi-node settings.
pub trait Tree {
    /// The computational domain defined by the tree.
    type Domain;

    /// Precision
    type Precision: RlstScalar<Real = Self::Precision> + Float + Default;

    /// A tree node.
    type Node: TreeNode<Self::Precision, Domain = Self::Domain> + Clone + Copy;

    /// Slice of nodes.
    type NodeSlice<'a>: IntoIterator<Item = &'a Self::Node>
    where
        Self: 'a;

    /// Copy of nodes
    type Nodes: IntoIterator<Item = Self::Node>;

    /// Number of leaves
    fn nleaves(&self) -> Option<usize>;

    /// Total number of keys
    fn nkeys_tot(&self) -> Option<usize>;

    /// Number of keys at a given tree level
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
    fn coordinates(&self, key: &Self::Node) -> Option<&[Self::Precision]>;

    /// Number of coordinates
    fn ncoordinates(&self, key: &Self::Node) -> Option<usize>;

    /// Gets a reference to the coordinates contained in across tree (local in multinode setting)
    fn all_coordinates(&self) -> Option<&[Self::Precision]>;

    /// Total number of coordinates
    fn ncoordinates_tot(&self) -> Option<usize>;

    /// Gets global indices at a leaf (local in multinode setting)
    fn global_indices<'a>(&'a self, key: &Self::Node) -> Option<&'a [usize]>;

    /// Gets all global indices (local in multinode setting)
    fn all_global_indices(&self) -> Option<&[usize]>;

    /// Get domain defined by the points, gets global domain in multi-node setting.
    fn domain(&self) -> &'_ Self::Domain;

    /// Get a map from the key to index position in sorted keys
    fn index(&self, key: &Self::Node) -> Option<&usize>;

    /// Get a node
    fn node(&self, idx: usize) -> Option<&Self::Node>;

    /// Get a map from the key to leaf index position in sorted leaves
    fn leaf_index(&self, key: &Self::Node) -> Option<&usize>;
}

/// An FMM tree
pub trait FmmTree {
    /// Precision
    type Precision;
    /// Node type
    type Node;
    /// Tree type
    type Tree: Tree<Precision = Self::Precision, Node = Self::Node>;

    /// Get the source tree
    fn source_tree(&self) -> &Self::Tree;

    /// Get the target tree
    fn target_tree(&self) -> &Self::Tree;

    /// Get the domain
    fn domain(&self) -> &<Self::Tree as Tree>::Domain;

    /// Get the near field of a leaf node
    fn near_field(&self, leaf: &Self::Node) -> Option<Vec<Self::Node>>;
}

/// A tree node
pub trait TreeNode<T>
where
    Self: Hash + Eq,
    T: RlstScalar,
{
    /// Domain
    type Domain;

    /// Copy of nodes
    type Nodes: IntoIterator<Item = Self>;

    /// The parent of this node
    fn parent(&self) -> Self;

    /// The level of this node
    fn level(&self) -> u64;

    /// Neighbours of this node defined by nodes sharing a vertex, edge, or face
    fn neighbors(&self) -> Self::Nodes;

    /// Children of this node
    fn children(&self) -> Self::Nodes;

    /// Checks adjacency, defined by sharing a vertex, edge, or face, between this node and another
    fn is_adjacent(&self, other: &Self) -> bool;
}
