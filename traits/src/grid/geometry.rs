//! Geometry of a physical cell
use super::GridType;
use rlst_dense::types::RlstScalar;

pub trait GeometryType {
    //! Cell geometry

    /// The type of the grid that the cell is part of
    type Grid: GridType;
    /// Type of iterator over vertices
    type VertexIterator<'iter>: std::iter::Iterator<Item = <Self::Grid as GridType>::Point<'iter>>
    where
        Self: 'iter;
    /// Type of iterator over points
    type PointIterator<'iter>: std::iter::Iterator<Item = <Self::Grid as GridType>::Point<'iter>>
    where
        Self: 'iter;

    /// The physical/geometric dimension of the cell
    fn physical_dimension(&self) -> usize;

    /// The midpoint of the cell
    fn midpoint(&self, point: &mut [<<Self::Grid as GridType>::T as RlstScalar>::Real]);

    /// The diameter of the cell
    fn diameter(&self) -> <<Self::Grid as GridType>::T as RlstScalar>::Real;

    /// The volume of the cell
    fn volume(&self) -> <<Self::Grid as GridType>::T as RlstScalar>::Real;

    /// The vertices of the cell
    ///
    /// The vertices are the points at the corners of the cell
    fn vertices(&self) -> Self::VertexIterator<'_>;

    /// The points of the cell
    ///
    /// The points are all points used to define the cell. For curved cells, this includes points on the edges and interior
    fn points(&self) -> Self::PointIterator<'_>;
}
