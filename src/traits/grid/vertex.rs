//! Definition of a vertex

use rlst::RlstScalar;

pub trait Vertex {
    //! A point

    /// The floating point type used for coordinates
    type T: RlstScalar;

    /// Get the coordinates of the point
    fn coords(&self) -> [Self::T; 3];

    /// Get the point's index
    fn index(&self) -> usize;
}
