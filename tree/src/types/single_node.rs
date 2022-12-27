//! Data Structures and methods to create octrees on a single node.
use std::collections::{HashSet, HashMap};

use crate::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, Points},
};

/// Concrete local (non-distributed) Tree.
#[derive(Debug)]
pub struct SingleNodeTree {

    /// Adaptivity is optional.
    pub adaptive: bool,

    ///  A vector of Cartesian points.
    pub points: Points,

    /// The nodes that span the tree, defined by its leaf nodes.
    pub keys: MortonKeys,
    
    /// The nodes that span the tree, defined by its leaf nodes, as a set.
    pub keys_set: HashSet<MortonKey>,

    /// Domain spanned by the points in the SingleNodeTree.
    pub domain: Domain,

    /// Map between the points and the nodes in the SingleNodeTree.
    pub points_to_keys: HashMap<Point, MortonKey>,

    /// Map between the nodes in the SingleNodetree and the points they contain.
    pub keys_to_points: HashMap<MortonKey, Points>,
}
