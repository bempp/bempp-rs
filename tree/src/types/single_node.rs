//! Data Structures and methods to create octrees on a single node.
use std::collections::{HashMap, HashSet};

use crate::types::{
    data::NodeData,
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, Points},
};

/// Concrete local (non-distributed) Tree.
#[derive(Debug)]
pub struct SingleNodeTree {

    // Depth of a tree
    pub depth: usize,

    /// Adaptivity is optional.
    pub adaptive: bool,

    ///  A vector of Cartesian points.
    pub points: Points,

    /// All ancestors of leaves in tree, as a set.
    pub keys_set: HashSet<MortonKey>,

    /// The leaves that span the tree, defined by its leaf nodes.
    pub leaves: MortonKeys,

    /// The nodes that span the tree, defined by its leaf nodes, as a set.
    pub leaves_set: HashSet<MortonKey>,

    /// Domain spanned by the points in the SingleNodeTree.
    pub domain: Domain,

    /// Map between the points and the nodes in the SingleNodeTree.
    pub points_to_leaves: HashMap<Point, MortonKey>,

    // Map between keys and data
    pub keys_to_data: HashMap<MortonKey, NodeData>,

    /// Map between the nodes in the SingleNodetree and the points they contain.
    pub leaves_to_points: HashMap<MortonKey, Points>,
}
