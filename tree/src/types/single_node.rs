//! Data Structures to create octrees on a single node.
use std::collections::{HashMap, HashSet};

use bempp_traits::types::RlstScalar;
use num::Float;

use crate::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
};

/// Local Trees (non-distributed).
#[derive(Default)]
pub struct SingleNodeTree<T>
where
    T: Float + Default + RlstScalar<Real = T>,
{
    /// Depth of a tree.
    pub depth: u64,

    /// Whether the tree is adaptive.
    pub adaptive: bool,

    /// Domain spanned by the points.
    pub domain: Domain<T>,

    /// All coordinates
    pub coordinates: Vec<T>,

    /// All global indices
    pub global_indices: Vec<usize>,

    /// The leaves that span the tree, and associated Point data.
    pub leaves: MortonKeys,

    /// All nodes in tree, and associated Node data.
    pub keys: MortonKeys,

    /// Associate leaves with coordinate indices.
    pub leaves_to_coordinates: HashMap<MortonKey, (usize, usize)>,

    /// Associate levels with key indices.
    pub levels_to_keys: HashMap<u64, (usize, usize)>,

    /// Map between a key and its index
    pub key_to_index: HashMap<MortonKey, usize>,

    /// Map between a leaf and its index
    pub leaf_to_index: HashMap<MortonKey, usize>,

    /// All leaves, returned as a set.
    pub leaves_set: HashSet<MortonKey>,

    /// All keys, returned as a set.
    pub keys_set: HashSet<MortonKey>,
}
