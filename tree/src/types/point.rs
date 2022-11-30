//! Data structures and methods for Cartesian Points in 3D.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::{
    types::morton::MortonKey,
};

pub type PointType = f64;

/// A 3D cartesian point, described by coordinate, a unique global index, and the Morton Key for
/// the octree node in which it lies. The ordering of Points is determined by their Morton Key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Point {
    pub coordinate: [PointType; 3],
    pub global_idx: usize,
    pub key: MortonKey,
}

/// Vector of **Points**.
pub type Points = Vec<Point>;
