//! Data structures for defining the computational domain.

use crate::types::point::PointType;

/// A domain is a box defined aby an origin coordinate and its diameter along all three Cartesian axes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Domain {
    /// The lower left corner of the domain, defined by the point distribution.
    pub origin: [PointType; 3],

    /// The diameter of the domain along the [x, y, z] axes respectively, defined
    /// by the maximum width of the point distribution along a given axis.
    pub diameter: [PointType; 3],
}
