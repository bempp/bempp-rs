//! Traits
use std::collections::HashSet;
use std::hash::Hash;

/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
/// This trait makes no assumptions about the downstream usage of a struct implementing Tree,
/// it simply provides methods for accessing tree nodes, and associated data, and is generically
/// defined for both single and multi-node settings.
pub trait Tree {
    // The computational domain defined by the tree.
    type Domain;

    // The type of points that define a tree.
    type Point;

    // Slice of points.
    type PointSlice<'a>: IntoIterator<Item = &'a Self::Point>
    where
        Self: 'a;

    // Data to be attached to each point.
    type PointData;

    type PointDataSlice<'a>: IntoIterator<Item = &'a Self::PointData>
    where
        Self: 'a;

    // A tree node.
    type NodeIndex: MortonKeyInterface;

    // Slice of nodes.
    type NodeIndexSlice<'a>: IntoIterator<Item = &'a Self::NodeIndex>
    where
        Self: 'a;

    // Copy of nodes
    type NodeIndices: IntoIterator<Item = Self::NodeIndex>;

    // Global indices of points
    type GlobalIndex;

    // Slice of global indices
    type GlobalIndexSlice<'a>: IntoIterator<Item = &'a Self::GlobalIndex>
    where
        Self: 'a;

    // Get depth of tree.
    fn get_depth(&self) -> u64;

    // Get a reference to all leaves, gets local keys in multi-node setting.
    fn get_leaves(&self) -> Option<Self::NodeIndexSlice<'_>>;

    // Get a reference to keys at a given level, gets local keys in a multi-node setting.
    fn get_keys(&self, level: u64) -> Option<Self::NodeIndexSlice<'_>>;

    // Get a reference to all keys, gets local keys in a multi-node setting.
    fn get_all_keys(&self) -> Option<Self::NodeIndexSlice<'_>>;

    // Get a reference to all keys as a set, gets local keys in a multi-node setting.
    fn get_all_keys_set(&self) -> &'_ HashSet<Self::NodeIndex>;

    // Get a reference to all leaves as a set, gets local keys in a multi-node setting.
    fn get_all_leaves_set(&self) -> &'_ HashSet<Self::NodeIndex>;

    // Gets a reference to the points contained with a leaf node.
    fn get_points<'a>(&'a self, key: &Self::NodeIndex) -> Option<Self::PointSlice<'a>>;

    // Get domain defined by the points, gets global domain in multi-node setting.
    fn get_domain(&self) -> &'_ Self::Domain;

    // Checks whether a a given node corresponds to a leaf
    fn is_leaf(&self, key: &Self::NodeIndex) -> bool;

    // Checks whether a a given node is contained in the tree
    fn is_node(&self, key: &Self::NodeIndex) -> bool;
}

/// A minimal interface for Morton Key like nodes.
pub trait MortonKeyInterface
where
    Self: Hash + Eq,
{
    // Copy of nodes
    type NodeIndices: IntoIterator<Item = Self>;

    fn parent(&self) -> Self;

    fn neighbors(&self) -> Self::NodeIndices;

    fn children(&self) -> Self::NodeIndices;

    fn is_adjacent(&self, other: &Self) -> bool;
}
