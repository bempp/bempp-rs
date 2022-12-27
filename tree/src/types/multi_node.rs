//! Data structures and methods to create distributed octrees with MPI.
use mpi::topology::UserCommunicator;

use std::collections::{HashSet, HashMap};

use crate::types::{
    domain::Domain,
    morton::{KeyType, MortonKey, MortonKeys},
    point::{Point, Points},
};

/// Concrete distributed multi-node tree.
pub struct MultiNodeTree {

    /// Global communicator for this Tree
    pub world: UserCommunicator,

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

    /// Range of Morton keys at this processor, and their current rank [rank, min, max]
    pub range: [KeyType; 3],
}
