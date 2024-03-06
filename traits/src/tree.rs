//! Traits
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::Index;

use cauchy::Scalar;
use num::Float;

pub trait LenAndIntoIterator<'a> {
    type Item;
    type IntoIter: IntoIterator<Item = Self::Item>;

    fn len(&self) -> usize;
    fn into_iter(&'a self) -> Self::IntoIter;
}

/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
/// This trait makes no assumptions about the downstream usage of a struct implementing Tree,
/// it simply provides methods for accessing tree nodes, and associated data, and is generically
/// defined for both single and multi-node settings.
pub trait Tree {
    /// The computational domain defined by the tree.
    type Domain;

    /// The precision of the point data
    type Precision: Scalar<Real = Self::Precision> + Float + Default;

    /// A tree node.
    type NodeIndex: MortonKeyInterface<Self::Precision, Domain = Self::Domain> + Clone + Copy;

    /// Slice of nodes.
    type NodeIndexSlice<'a>: IntoIterator<Item = &'a Self::NodeIndex>
    where
        Self: 'a;

    /// Copy of nodes
    type NodeIndices: IntoIterator<Item = Self::NodeIndex>;

    fn get_nleaves(&self) -> Option<usize>;
    fn get_nall_keys(&self) -> Option<usize>;
    fn get_nkeys(&self, level: u64) -> Option<usize>;

    /// Get depth of tree.
    fn get_depth(&self) -> u64;

    /// Get a reference to all leaves, gets local keys in multi-node setting.
    fn get_all_leaves(&self) -> Option<Self::NodeIndexSlice<'_>>;

    /// Get a reference to keys at a given level, gets local keys in a multi-node setting.
    fn get_keys(&self, level: u64) -> Option<Self::NodeIndexSlice<'_>>;

    /// Get a reference to all keys, gets local keys in a multi-node setting.
    fn get_all_keys(&self) -> Option<Self::NodeIndexSlice<'_>>;

    /// Get a reference to all keys as a set, gets local keys in a multi-node setting.
    fn get_all_keys_set(&self) -> Option<&'_ HashSet<Self::NodeIndex>>;

    /// Get a reference to all leaves as a set, gets local keys in a multi-node setting.
    fn get_all_leaves_set(&self) -> Option<&'_ HashSet<Self::NodeIndex>>;

    /// Gets a reference to the coordinates contained with a leaf node.
    fn get_coordinates<'a>(&'a self, key: &Self::NodeIndex) -> Option<&'a [Self::Precision]>;

    /// Gets a reference to the coordinates contained in across tree (local in multinode setting)
    fn get_all_coordinates(&self) -> Option<&[Self::Precision]>;

    /// Gets global indices at a leaf (local in multinode setting)
    fn get_global_indices<'a>(&'a self, key: &Self::NodeIndex) -> Option<&'a [usize]>;

    /// Gets all global indices (local in multinode setting)
    fn get_all_global_indices(&self) -> Option<&[usize]>;

    /// Get domain defined by the points, gets global domain in multi-node setting.
    fn get_domain(&self) -> &'_ Self::Domain;

    /// Get a map from the key to index position in sorted keys
    fn get_index(&self, key: &Self::NodeIndex) -> Option<&usize>;

    fn get_node_index(&self, idx: usize) -> Option<&Self::NodeIndex>;

    /// Get a map from the key to leaf index position in sorted leaves
    fn get_leaf_index(&self, key: &Self::NodeIndex) -> Option<&usize>;
}

pub trait FmmTree {
    type Precision;
    type NodeIndex;

    type Tree: Tree<Precision = Self::Precision, NodeIndex = Self::NodeIndex>;

    fn get_source_tree(&self) -> &Self::Tree;

    fn get_target_tree(&self) -> &Self::Tree;

    fn get_domain(&self) -> &<Self::Tree as Tree>::Domain;
}

/// A minimal interface for Morton Key like nodes.
pub trait MortonKeyInterface<T>
where
    Self: Hash + Eq,
    T: Scalar,
{
    type Domain;

    // Copy of nodes
    type NodeIndices: IntoIterator<Item = Self>;

    /// The parent of a key.
    fn parent(&self) -> Self;

    fn level(&self) -> u64;

    fn compute_surface(
        &self,
        domain: &Self::Domain,
        expansion_order: usize,
        alpha: T,
    ) -> Vec<<T as Scalar>::Real>;

    /// Neighbours defined by keys sharing a vertex, edge, or face.
    fn neighbors(&self) -> Self::NodeIndices;

    /// Childen of a key.
    fn children(&self) -> Self::NodeIndices;

    /// Checks adjacency, defined by sharing a vertex, edge, or face, between two keys.
    fn is_adjacent(&self, other: &Self) -> bool;
}
