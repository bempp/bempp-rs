//! Map from reference to physical space.

use crate::grid::GridType;
use rlst::RlstScalar;

pub trait ReferenceMapType {
    //! Reference to physical map

    /// The type of the grid that this map maps
    type Grid: GridType;

    /// The topoloical/domain dimension
    fn domain_dimension(&self) -> usize;

    /// The geometric/physical dimension
    fn physical_dimension(&self) -> usize;

    /// The number of reference points that this map uses
    fn number_of_reference_points(&self) -> usize;

    /// Get an iterator that returns a slice with the value of the
    /// physical point for each reference point
    fn reference_to_physical(
        &self,
        cell_index: usize,
        point_index: usize,
        value: &mut [<<Self::Grid as GridType>::T as RlstScalar>::Real],
    );

    /// Get an iterator that returns a slice with the value of the
    /// Jacobian at the physical point for each reference point
    fn jacobian(
        &self,
        cell_index: usize,
        point_index: usize,
        value: &mut [<<Self::Grid as GridType>::T as RlstScalar>::Real],
    );

    /// Get an iterator that returns a slice with the normal direction
    /// at each point
    fn normal(
        &self,
        cell_index: usize,
        point_index: usize,
        value: &mut [<<Self::Grid as GridType>::T as RlstScalar>::Real],
    );
}
