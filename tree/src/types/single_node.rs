//! Data Structures and methods to create octrees on a single node.
use std::collections::{HashSet, HashMap};

use crate::types::{
    domain::Domain,
    node::{LeafNodes, Nodes},
    point::Points,
};

use super::morton::MortonKey;

/// Concrete local (non-distributed) Tree.
#[derive(Debug)]
pub struct SingleNodeTree {
    // Depth of a tree
    pub depth: usize,

    /// Domain spanned by the points in the SingleNodeTree.
    pub domain: Domain,

    ///  All Points.
    pub points: Points,

    /// The leaves that span the tree, defined by its leaf nodes.
    pub leaves: LeafNodes,

    /// All ancestors of leaves in tree
    pub keys: Nodes,

    pub keys_set: HashSet<MortonKey>,

    pub leaf_to_index: HashMap<MortonKey, usize>,
    
    pub key_to_index: HashMap<MortonKey, usize>
}
