use std::collections::HashSet;

/// Tree is the trait interface for distributed octrees implemented by Rusty Fast Solvers.
pub trait Tree<'a> {
    // The computational domain defined by the tree
    type Domain;

    // The type of points that define a tree
    type Point;

    // A container for multiple Points
    type Points;

    // Unique index for tree nodes
    type LeafNodeIndex;

    // Container for multiple tree nodes
    type LeafNodeIndices;

    // Unique index for tree nodes
    type NodeIndex;

    // Container for multiple tree nodes
    type NodeIndices;

    type RawNodeIndex;

    // Get depth of tree
    fn get_depth(&self) -> usize;

    // Get all leaves, gets local keys in multi-node setting
    fn get_leaves(&self) -> &Self::LeafNodeIndices;

    fn get_leaves_mut(&mut self) -> &mut Self::LeafNodeIndices;

    // Get all keys at a given level, gets matching local keys in a multi-node setting
    fn get_keys(&self) -> &Self::NodeIndices;

    fn get_keys_mut(&mut self) -> &mut Self::NodeIndices;

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Self::Domain;

    fn get_keys_set(&self) -> &HashSet<Self::RawNodeIndex>;

    fn key_to_index(&self, key: &Self::RawNodeIndex) -> usize;
    
    fn leaf_to_index(&self, key: &Self::RawNodeIndex) -> usize;
}
