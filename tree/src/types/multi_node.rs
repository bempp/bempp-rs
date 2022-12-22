//! Data structures and methods to create distributed octrees with MPI.

use std::collections::HashMap;

use crate::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, Points},
};

/// Concrete distributed multi-node tree.
pub struct MultiNodeTree {
    /// Adaptivity is optional.
    pub adaptive: bool,

    ///  A vector of Cartesian points.
    pub points: Points,

    /// The nodes that span the tree, defined by its leaf nodes.
    pub keys: MortonKeys,
    
    /// The nodes that span the tree, defined by its leaf nodes, as a set.
    pub keys_set: HashSet<MortonKey>,

    /// Domain spanned by the points in the tree.
    pub domain: Domain,

    /// Map between the points and the nodes in the tree.
    pub points_to_keys: HashMap<Point, MortonKey>,

    /// Map between the nodes in the tree and the points they contain.
    pub keys_to_points: HashMap<MortonKey, Points>,
}
