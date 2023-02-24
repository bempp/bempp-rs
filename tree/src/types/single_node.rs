//! Data Structures and methods to create octrees on a single node.
use std::collections::{HashMap, HashSet};

use crate::types::{
    domain::Domain,
    node::{LeafNodes, Nodes},
    point::Points,
};

use super::morton::MortonKey;

/// Local Trees (non-distributed).
#[derive(Debug)]
pub struct SingleNodeTree {
    /// Depth of a tree.
    pub depth: usize,

    /// Domain spanned by the points.
    pub domain: Domain,

    ///  All Points.
    pub points: Points,

    /// The leaves that span the tree, and associated Point data.
    pub leaves: LeafNodes,

    /// All nodes in tree, and associated Node data.
    pub keys: Nodes,

    /// A convenent wrapper for a set of all the the raw MortonKeys associated with nodes.
    pub keys_set: HashSet<MortonKey>,

    /// Index pointer mapping Mortonkeys of leaves to their index in the container of associated data.
    pub leaf_to_index: HashMap<MortonKey, usize>,

    /// Index pointer mapping MortonKeys of nodes to their index in the container of associated data.
    pub key_to_index: HashMap<MortonKey, usize>,
}
