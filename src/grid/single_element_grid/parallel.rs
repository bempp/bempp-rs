//! Parallel grid builder

use crate::grid::parallel_grid::ParallelGridBuilder;
use crate::grid::single_element_grid::{SingleElementGrid, SingleElementGridBuilder};
use mpi::traits::{Buffer, Equivalence};
use ndelement::types::ReferenceCellType;
use num::Float;
use rlst::{
    dense::array::views::ArrayViewMut, Array, BaseArray, MatrixInverse, RlstScalar, VectorContainer,
};
use std::collections::HashMap;

impl<T: Float + RlstScalar<Real = T> + Equivalence> ParallelGridBuilder
    for SingleElementGridBuilder<3, T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
    [T]: Buffer,
{
    type G = SingleElementGrid<T>;
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
        &self.cells[self.points_per_cell * index..self.points_per_cell * (index + 1)]
    }
    fn cell_vertices(&self, index: usize) -> &[usize] {
        &self.cells
            [self.points_per_cell * index..self.points_per_cell * index + self.vertices_per_cell]
    }
    fn cell_type(&self, _index: usize) -> ReferenceCellType {
        self.element_data.0
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
        SingleElementGrid::new(
            points,
            cells,
            self.element_data.0,
            self.element_data.1,
            point_indices_to_ids,
            point_ids_to_indices,
            cell_indices_to_ids,
            cell_ids_to_indices,
            Some(edge_ids),
        )
    }
}
