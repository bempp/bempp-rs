//! Traits
use std::collections::HashSet;

/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
/// This trait makes no assumptions about the downstream usage of a struct implementing Tree,
/// it simply provides methods for accessing tree nodes, and associated data, and is generically
/// defined for both single and multi-node settings.
pub trait Tree<'a> {
    // The computational domain defined by the tree.
    type Domain;

    // The type of points that define a tree.
    type Point;

    // Container for multiple Points.
    type Points;

    // A tree leaf nodes, containing leaf data.
    type LeafNodeIndex;

    // Container for multiple tree leaf nodes.
    type LeafNodeIndices;

    // A tree node, containing node data.
    type NodeIndex;

    // Container for multiple tree nodes.
    type NodeIndices;

    // A raw index for a node, for example a Morton Index from a space filling curve.
    type RawNodeIndex;

    // Get depth of tree.
    fn get_depth(&self) -> usize;

    // Get a reference all leaves, gets local keys in multi-node setting.
    fn get_leaves(&self) -> &Self::LeafNodeIndices;

    // Get a mutable reference all leaves, gets local keys in multi-node setting.
    fn get_leaves_mut(&mut self) -> &mut Self::LeafNodeIndices;

    // Get a reference to all keys, gets local keys in a multi-node setting.
    fn get_keys(&self) -> &Self::NodeIndices;

    // Get a mutable reference to all keys, gets local keys in a multi-node setting.
    fn get_keys_mut(&mut self) -> &mut Self::NodeIndices;

    // Get domain defined by the points, gets global domain in multi-node setting.
    fn get_domain(&self) -> &Self::Domain;

    // Get a reference to a hashset of all keys, gets matching local keys in a multi-node setting.
    fn get_keys_set(&self) -> &HashSet<Self::RawNodeIndex>;

    // Get a reference to a leaf node if it exists, checks locally in a multi-node setting.
    fn get_leaf_node(&self, key: &Self::RawNodeIndex) -> Option<&Self::LeafNodeIndex>;

    // Get a mutable reference to a leaf node if it exists, checks locally in a multi-node setting.
    fn get_leaf_node_mut(&mut self, key: &Self::RawNodeIndex) -> Option<&mut Self::LeafNodeIndex>;

    // Get a reference to a node if it exists, checks locally in a multi-node setting.
    fn get_node(&self, key: &Self::RawNodeIndex) -> Option<&Self::NodeIndex>;

    // Get a mutable reference to a node if it exists, checks locally in a multi-node setting.
    fn get_node_mut(&mut self, key: &Self::RawNodeIndex) -> Option<&mut Self::NodeIndex>;
}
