//! Grid builder

use crate::flat_triangle_grid::grid::SerialFlatTriangleGrid;
use crate::traits_impl::WrappedGrid;
use bempp_traits::grid::Builder;
use num::Float;
use rlst_dense::types::RlstScalar;
use rlst_dense::{
    array::{views::ArrayViewMut, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_array_from_slice2, rlst_dynamic_array2,
    traits::MatrixInverse,
};
use std::collections::HashMap;

/// Grid builder for a flat triangle grid
pub struct SerialFlatTriangleGridBuilder<T: Float + RlstScalar<Real = T>> {
    points: Vec<T>,
    cells: Vec<usize>,
    point_indices_to_ids: Vec<usize>,
    point_ids_to_indices: HashMap<usize, usize>,
    cell_indices_to_ids: Vec<usize>,
    cell_ids_to_indices: HashMap<usize, usize>,
}

impl<T: Float + RlstScalar<Real = T>> Builder<3> for SerialFlatTriangleGridBuilder<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    type GridType = WrappedGrid<SerialFlatTriangleGrid<T>>;
    type T = T;
    type CellData = [usize; 3];
    type GridMetadata = ();

    fn new(_data: ()) -> Self {
        Self {
            points: vec![],
            cells: vec![],
            point_indices_to_ids: vec![],
            point_ids_to_indices: HashMap::new(),
            cell_indices_to_ids: vec![],
            cell_ids_to_indices: HashMap::new(),
        }
    }

    fn new_with_capacity(npoints: usize, ncells: usize, _data: ()) -> Self {
        Self {
            points: Vec::with_capacity(npoints * Self::GDIM),
            cells: Vec::with_capacity(ncells * 3),
            point_indices_to_ids: Vec::with_capacity(npoints),
            point_ids_to_indices: HashMap::new(),
            cell_indices_to_ids: Vec::with_capacity(ncells),
            cell_ids_to_indices: HashMap::new(),
        }
    }

    fn add_point(&mut self, id: usize, data: [T; 3]) {
        self.point_ids_to_indices
            .insert(id, self.point_indices_to_ids.len());
        self.point_indices_to_ids.push(id);
        self.points.extend_from_slice(&data);
    }

    fn add_cell(&mut self, id: usize, cell_data: [usize; 3]) {
        self.cell_ids_to_indices
            .insert(id, self.cell_indices_to_ids.len());
        self.cell_indices_to_ids.push(id);
        for id in &cell_data {
            self.cells.push(self.point_ids_to_indices[id]);
        }
    }

    fn create_grid(self) -> Self::GridType {
        // TODO: remove this transposing
        let npts = self.point_indices_to_ids.len();
        let mut points = rlst_dynamic_array2!(T, [npts, 3]);
        points.fill_from(rlst_array_from_slice2!(
            T,
            &self.points,
            [npts, 3],
            [1, npts]
        ));
        WrappedGrid {
            grid: SerialFlatTriangleGrid::new(
                points,
                &self.cells,
                self.point_indices_to_ids,
                self.point_ids_to_indices,
                self.cell_indices_to_ids,
                self.cell_ids_to_indices,
            ),
        }
    }
}
