//! Data structures for defining the computational domain.

use crate::types::point::PointType;
use num::traits::Float;

/// A domain is a box defined aby an origin coordinate and its diameter along all three Cartesian axes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Domain<T>
where
    T: Float + Default,
{
    /// The lower left corner of the domain, defined by the point distribution.
    pub origin: [PointType<T>; 3],

    /// The diameter of the domain along the [x, y, z] axes respectively, defined
    /// by the maximum width of the point distribution along a given axis.
    pub diameter: [PointType<T>; 3],
}
