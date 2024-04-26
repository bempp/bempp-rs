//! Definition of a vertex

use crate::traits::types::Ownership;
use rlst::RlstScalar;

pub trait PointType {
    //! A point

    /// The floating point type used for coordinates
    type T: RlstScalar;

    /// Get the coordinates of the point
    fn coords(&self, data: &mut [<Self::T as RlstScalar>::Real]);

    /// Get the point's index
    fn index(&self) -> usize;

    /// Get the point's id
    fn id(&self) -> usize;

    /// Get the point's ownership
    fn ownership(&self) -> Ownership;
}
