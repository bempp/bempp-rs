//! Traits
use std::collections::HashSet;

/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
/// This trait makes no assumptions about the downstream usage of a struct implementing Tree,
/// it simply provides methods for accessing tree nodes, and associated data, and is generically
/// defined for both single and multi-node settings.

// TODO: Why does the tree have a lifetime? It should not be necessary.
pub trait Tree {
    // The computational domain defined by the tree.
    type Domain;

    // TODO Clean-Up
    // The type of points that define a tree.
    type Point;
    type PointIterator: std::iter::Iterator<Item = Self::Point>;

    // TODO Clean-Up
    // A tree node, containing node data.
    type NodeIndex;
    type NodeIndexIterator: std::iter::Iterator<Item = Self::NodeIndex>;

    // Get depth of tree.
    fn get_depth(&self) -> usize;

    // TODO Should this return an iterator?
    // Get a reference all leaves, gets local keys in multi-node setting.
    fn get_leaves(&self) -> &Self::NodeIndexIterator;

    // TODO Should this return an iterator?
    // Get a reference to all keys, gets local keys in a multi-node setting.
    fn get_keys(&self) -> &Self::NodeIndexIterator;

    // Get domain defined by the points, gets global domain in multi-node setting.
    fn get_domain(&self) -> &Self::Domain;

    // Get a reference to a leaf node if it exists, checks locally in a multi-node setting.
    fn has_leaf(key: &Self::NodeIndex) -> bool;

    // Get a reference to a node if it exists, checks locally in a multi-node setting.
    fn has_node(&self, key: &Self::NodeIndex) -> bool;
}

// TODO We discussed attaching data directly to the tree. Ideally, we would have something like the following trait:

pub trait AttachedData {
    type Data;
    type Tree: Tree;

    fn get_data(&self, key: &<Self::Tree as Tree>::NodeIndex) -> Option<&Self::Data>;

    // Use interior mutability to avoid mut ref to self
    fn get_data_mut(&self, key: &<Self::Tree as Tree>::NodeIndex) -> Option<&mut Self::Data>;

    fn get_tree(&self) -> &Self::Tree;
}
