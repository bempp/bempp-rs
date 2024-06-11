//! Grid builder
use super::GridType;
use rlst::RlstScalar;

pub trait Builder<const GDIM: usize> {
    //! Object that can be used to build a mesh

    /// The geometric/physical dimension
    const GDIM: usize = GDIM;
    /// The type of the grid that the builder creates
    type GridType: GridType;
    /// The floating point type used for coordinates
    type T: RlstScalar;
    /// The type of the data that is input to add a cell
    type CellData;

    /// Add a point to the grid
    fn add_point(&mut self, id: usize, data: [<Self::T as RlstScalar>::Real; GDIM]);

    /// Add a cell to the grid
    fn add_cell(&mut self, id: usize, cell_data: Self::CellData);

    /// Create the grid
    fn create_grid(self) -> Self::GridType;
}
