//! Data structures and methods for defining the computational domain.

use crate::{
    types::{
        point::PointType,
    }
};

/// A domain is defined by an origin coordinate, and its diameter along all three Cartesian axes.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct Domain {
    pub origin: [PointType; 3],
    pub diameter: [PointType; 3],
}
