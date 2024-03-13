//! Data structures to create distributed octrees with MPI.
use bempp_traits::types::RlstScalar;
use mpi::topology::UserCommunicator;

use std::collections::{HashMap, HashSet};

use num::traits::Float;

use crate::types::{
    domain::Domain,
    morton::{KeyType, MortonKey, MortonKeys},
    point::Points,
};

/// Concrete distributed multi-node tree.
pub struct MultiNodeTree<T>
where
    T: Default + Float + RlstScalar<Real = T>,
{
    /// Global communicator for this Tree
    pub world: UserCommunicator,

    // Depth of the tree
    pub depth: u64,

    /// Domain spanned by the points.
    pub domain: Domain<T>,

    ///  A vector of Cartesian points.
    pub points: Points<T>,

    /// All coordinates
    pub coordinates: Vec<T>,

    /// All global indices
    pub global_indices: Vec<usize>,

    /// The leaves that span the tree.
    pub leaves: MortonKeys,

    /// All nodes in tree.
    pub keys: MortonKeys,

    /// Associate leaves with point indices.
    pub leaves_to_coordinates: HashMap<MortonKey, (usize, usize)>,

    /// Associate levels with key indices.
    pub levels_to_keys: HashMap<u64, (usize, usize)>,

    /// All leaves, returned as a set.
    pub leaves_set: HashSet<MortonKey>,

    /// All keys, returned as a set.
    pub keys_set: HashSet<MortonKey>,

    /// Map between a key and its index
    pub key_to_index: HashMap<MortonKey, usize>,

    /// Map between a leaf and its index
    pub leaf_to_index: HashMap<MortonKey, usize>,

    /// Range of Morton keys at this processor, and their current rank [rank, min, max]
    pub range: [KeyType; 3],
}
