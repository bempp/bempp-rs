//! Data structures for Cartesian Points in 3D.
use bempp_traits::types::RlstScalar;

use crate::types::morton::MortonKey;

<<<<<<< HEAD
=======
/// Point type
pub type PointType<T> = T;

>>>>>>> main
/// A 3D cartesian point, described by coordinate, a unique global index, and the Morton Key for
/// the octree node in which it lies. Each Point as an associated 'base key', which is its matching
/// Morton encoding at the lowest possible level of discretization (DEEPEST_LEVEL), and an 'encoded key'
/// specifiying its encoding at a given level of discretization. Points also have associated data
#[repr(C)]
#[derive(Clone, Debug, Default, Copy)]
pub struct Point<T>
where
    T: RlstScalar<Real = T>,
{
    /// Physical coordinate in Cartesian space.
    pub coordinate: [T; 3],

    /// Global unique index.
    pub global_idx: usize,

    /// Key at finest level of encoding.
    pub base_key: MortonKey,

    /// Key at a given level of encoding, strictly an ancestor of 'base_key'.
    pub encoded_key: MortonKey,
}

/// Vector of **Points**.
/// Container of **Points**.
#[derive(Clone, Debug, Default)]
pub struct Points<T>
where
    T: RlstScalar<Real = T>,
{
    /// A vector of Points
    pub points: Vec<Point<T>>,

    /// index for implementing the Iterator trait.
    pub index: usize,
}
