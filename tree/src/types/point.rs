//! Data structures and methods for Cartesian Points in 3D.
use crate::types::morton::MortonKey;

pub type PointType = f64;

/// A 3D cartesian point, described by coordinate, a unique global index, and the Morton Key for
/// the octree node in which it lies. The ordering of Points is determined by their Morton Key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub coordinate: [PointType; 3],
    pub global_idx: usize,
    pub key: MortonKey,
}

/// Vector of **Points**.
pub type Points = Vec<Point>;
