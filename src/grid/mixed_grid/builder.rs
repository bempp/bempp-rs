//! Grid builder

use crate::element::ciarlet::lagrange;
use crate::grid::mixed_grid::grid::MixedGrid;
use crate::traits::element::{Continuity, FiniteElement};
use crate::traits::grid::Builder;
use crate::traits::types::ReferenceCellType;
use num::Float;
use rlst::RlstScalar;
use rlst::{
    dense::array::{views::ArrayViewMut, Array},
    rlst_array_from_slice2, rlst_dynamic_array2, BaseArray, MatrixInverse, VectorContainer,
};
use std::collections::HashMap;

/// Grid builder for a mixed grid
pub struct MixedGridBuilder<const GDIM: usize, T: Float + RlstScalar<Real = T>> {
    elements_to_npoints: HashMap<(ReferenceCellType, usize), usize>,
    points: Vec<T>,
    cells: Vec<usize>,
    cell_types: Vec<ReferenceCellType>,
    cell_degrees: Vec<usize>,
    point_indices_to_ids: Vec<usize>,
    point_ids_to_indices: HashMap<usize, usize>,
    cell_indices_to_ids: Vec<usize>,
    cell_ids_to_indices: HashMap<usize, usize>,
}

impl<const GDIM: usize, T: Float + RlstScalar<Real = T>> Builder<GDIM> for MixedGridBuilder<GDIM, T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    type GridType = MixedGrid<T>;
    type T = T;
    type CellData = (Vec<usize>, ReferenceCellType, usize);
    type GridMetadata = ();

    fn new(_data: ()) -> Self {
        Self {
            elements_to_npoints: HashMap::new(),
            points: vec![],
            cells: vec![],
            cell_types: vec![],
            cell_degrees: vec![],
            point_indices_to_ids: vec![],
            point_ids_to_indices: HashMap::new(),
            cell_indices_to_ids: vec![],
            cell_ids_to_indices: HashMap::new(),
        }
    }

    fn new_with_capacity(npoints: usize, ncells: usize, _data: ()) -> Self {
        Self {
            elements_to_npoints: HashMap::new(),
            points: Vec::with_capacity(npoints * Self::GDIM),
            cells: vec![],
            cell_types: vec![],
            cell_degrees: vec![],
            point_indices_to_ids: Vec::with_capacity(npoints),
            point_ids_to_indices: HashMap::new(),
            cell_indices_to_ids: Vec::with_capacity(ncells),
            cell_ids_to_indices: HashMap::new(),
        }
    }

    fn add_point(&mut self, id: usize, data: [T; GDIM]) {
        self.point_ids_to_indices
            .insert(id, self.point_indices_to_ids.len());
        self.point_indices_to_ids.push(id);
        self.points.extend_from_slice(&data);
    }

    fn add_cell(&mut self, id: usize, cell_data: (Vec<usize>, ReferenceCellType, usize)) {
        let points_per_cell =
            if let Some(npts) = self.elements_to_npoints.get(&(cell_data.1, cell_data.2)) {
                *npts
            } else {
                let npts =
                    lagrange::create::<T>(cell_data.1, cell_data.2, Continuity::Continuous).dim();
                self.elements_to_npoints
                    .insert((cell_data.1, cell_data.2), npts);
                npts
            };
        assert_eq!(cell_data.0.len(), points_per_cell);
        self.cell_ids_to_indices
            .insert(id, self.cell_indices_to_ids.len());
        self.cell_indices_to_ids.push(id);
        for id in &cell_data.0 {
            self.cells.push(self.point_ids_to_indices[id]);
        }
        self.cell_types.push(cell_data.1);
        self.cell_degrees.push(cell_data.2);
    }

    fn create_grid(self) -> Self::GridType {
        // TODO: remove this transposing
        let npts = self.point_indices_to_ids.len();
        let mut points = rlst_dynamic_array2!(T, [npts, 3]);
        points.fill_from(rlst_array_from_slice2!(&self.points, [npts, 3], [3, 1]));
        MixedGrid::new(
            points,
            &self.cells,
            &self.cell_types,
            &self.cell_degrees,
            self.point_indices_to_ids,
            self.point_ids_to_indices,
            self.cell_indices_to_ids,
        )
    }
}
