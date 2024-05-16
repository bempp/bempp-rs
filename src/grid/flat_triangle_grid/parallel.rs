//! Parallel grid builder

use crate::grid::flat_triangle_grid::{FlatTriangleGrid, FlatTriangleGridBuilder};
use crate::grid::parallel_grid::ParallelGridBuilder;
use crate::traits::types::ReferenceCellType;
use mpi::traits::{Buffer, Equivalence};
use num::Float;
use rlst::{
    dense::array::views::ArrayViewMut, Array, BaseArray, MatrixInverse, RlstScalar, VectorContainer,
};
use std::collections::HashMap;

impl<T: Float + RlstScalar<Real = T> + Equivalence> ParallelGridBuilder
    for FlatTriangleGridBuilder<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
    [T]: Buffer,
{
    type G = FlatTriangleGrid<T>;
    type ExtraCellInfo = ();

    fn new_extra_cell_info(&self) {}

    fn point_indices_to_ids(&self) -> &[usize] {
        &self.point_indices_to_ids
    }
    fn points(&self) -> &[T] {
        &self.points
    }
    fn cell_indices_to_ids(&self) -> &[usize] {
        &self.cell_indices_to_ids
    }
    fn cell_points(&self, index: usize) -> &[usize] {
        &self.cells[3 * index..3 * (index + 1)]
    }
    fn cell_vertices(&self, index: usize) -> &[usize] {
        self.cell_points(index)
    }
    fn cell_type(&self, _index: usize) -> ReferenceCellType {
        ReferenceCellType::Triangle
    }
    fn create_serial_grid(
        &self,
        points: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        cells: &[usize],
        point_indices_to_ids: Vec<usize>,
        point_ids_to_indices: HashMap<usize, usize>,
        cell_indices_to_ids: Vec<usize>,
        cell_ids_to_indices: HashMap<usize, usize>,
        edge_ids: HashMap<[usize; 2], usize>,
        _extra_cell_info: &(),
    ) -> Self::G {
        FlatTriangleGrid::new(
            points,
            cells,
            point_indices_to_ids,
            point_ids_to_indices,
            cell_indices_to_ids,
            cell_ids_to_indices,
            Some(edge_ids),
        )
    }
}
