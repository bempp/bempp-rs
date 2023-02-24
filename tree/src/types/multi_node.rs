// //! Data structures and methods to create distributed octrees with MPI.
// use mpi::topology::UserCommunicator;

// use std::collections::{HashMap, HashSet};

// use crate::types::{
//     data::NodeData,
//     domain::Domain,
//     morton::{KeyType, MortonKey, MortonKeys},
//     point::{Point, Points},
// };

// /// Concrete distributed multi-node tree.
// pub struct MultiNodeTree {
//     /// Global communicator for this Tree
//     pub world: UserCommunicator,

//     /// Adaptivity is optional.
//     pub adaptive: bool,

//     ///  A vector of Cartesian points.
//     pub points: Points,

//     /// All ancestors of leaves in tree, as a set.
//     pub keys_set: HashSet<MortonKey>,

//     /// The leaf nodes that span the tree, defined by its leaf nodes.
//     pub leaves: MortonKeys,

//     /// The leaf nodes that span the tree, defined by its leaf nodes, as a set.
//     pub leaves_set: HashSet<MortonKey>,

//     /// Domain spanned by the points in the tree.
//     pub domain: Domain,

//     /// Map between the points and the leaves in the tree.
//     pub points_to_leaves: HashMap<Point, MortonKey>,

//     /// Map between the nodes in the tree and the points they contain.
//     pub leaves_to_points: HashMap<MortonKey, Points>,

//     // Map between keys and data
//     pub keys_to_data: HashMap<MortonKey, NodeData>,

//     /// Range of Morton keys at this processor, and their current rank [rank, min, max]
//     pub range: [KeyType; 3],
// }
