//! Data Structures to create octrees on a single node.
use std::collections::{HashMap, HashSet};

use num::Float;

use crate::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::Points,
};

/// Local Trees (non-distributed).
#[derive(Debug)]
pub struct SingleNodeTree<T>
where
    T: Float + Default,
{
    /// Depth of a tree.
    pub depth: u64,

    /// Domain spanned by the points.
    pub domain: Domain<T>,

    ///  All Points.
    pub points: Points<T>,

    /// The leaves that span the tree, and associated Point data.
    pub leaves: MortonKeys,

    /// All nodes in tree, and associated Node data.
    pub keys: MortonKeys,

    /// Associate leaves with point indices.
    pub leaves_to_points: HashMap<MortonKey, (usize, usize)>,

    /// Associate levels with key indices.
    pub levels_to_keys: HashMap<u64, (usize, usize)>,

    /// All leaves, returned as a set.
    pub leaves_set: HashSet<MortonKey>,

    /// All keys, returned as a set.
    pub keys_set: HashSet<MortonKey>,
}
