//! Map from reference to physical space.

use crate::traits::grid::Grid;
use rlst::RlstScalar;

pub trait ReferenceMap {
    //! Reference to physical map

    /// The type of the grid that this map maps
    type Grid: Grid;

    /// The topoloical/domain dimension
    fn domain_dimension(&self) -> usize;

    /// The geometric/physical dimension
    fn physical_dimension(&self) -> usize;

    /// The number of reference points that this map uses
    fn number_of_reference_points(&self) -> usize;

    /// Write the physical points for the cell with index `cell_index` into `value`
    ///
    /// `value` should have shape [npts, physical_dimension] and use column-major ordering
    fn reference_to_physical(
        &self,
        cell_index: usize,
        value: &mut [<<Self::Grid as Grid>::T as RlstScalar>::Real],
    );

    /// Write the jacobians at the physical points for the cell with index `cell_index` into `value`
    ///
    /// `value` should have shape [npts, physical_dimension*domain_dimension] and use column-major ordering
    fn jacobian(
        &self,
        cell_index: usize,
        value: &mut [<<Self::Grid as Grid>::T as RlstScalar>::Real],
    );

    /// Write the normals at the physical points for the cell with index `cell_index` into `value`
    ///
    /// `value` should have shape [physical_dimension, npts] and use column-major ordering
    fn normal(
        &self,
        cell_index: usize,
        value: &mut [<<Self::Grid as Grid>::T as RlstScalar>::Real],
    );
}
