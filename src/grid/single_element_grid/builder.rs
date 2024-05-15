//! Grid builder

use crate::element::ciarlet::lagrange;
use crate::grid::single_element_grid::grid::SingleElementGrid;
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

/// Grid builder for a single element grid
pub struct SingleElementGridBuilder<const GDIM: usize, T: Float + RlstScalar<Real = T>> {
    pub(crate) element_data: (ReferenceCellType, usize),
    pub(crate) points_per_cell: usize,
    pub(crate) points: Vec<T>,
    pub(crate) cells: Vec<usize>,
    pub(crate) point_indices_to_ids: Vec<usize>,
    pub(crate) point_ids_to_indices: HashMap<usize, usize>,
    pub(crate) cell_indices_to_ids: Vec<usize>,
    pub(crate) cell_ids_to_indices: HashMap<usize, usize>,
}

impl<const GDIM: usize, T: Float + RlstScalar<Real = T>> Builder<GDIM>
    for SingleElementGridBuilder<GDIM, T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    type GridType = SingleElementGrid<T>;
    type T = T;
    type CellData = Vec<usize>;
    type GridMetadata = (ReferenceCellType, usize);

    fn new(data: (ReferenceCellType, usize)) -> Self {
        let points_per_cell = lagrange::create::<T>(data.0, data.1, Continuity::Continuous).dim();
        Self {
            element_data: data,
            points_per_cell,
            points: vec![],
            cells: vec![],
            point_indices_to_ids: vec![],
            point_ids_to_indices: HashMap::new(),
            cell_indices_to_ids: vec![],
            cell_ids_to_indices: HashMap::new(),
        }
    }

    fn new_with_capacity(npoints: usize, ncells: usize, data: (ReferenceCellType, usize)) -> Self {
        let points_per_cell = lagrange::create::<T>(data.0, data.1, Continuity::Continuous).dim();
        Self {
            element_data: data,
            points_per_cell,
            points: Vec::with_capacity(npoints * Self::GDIM),
            cells: Vec::with_capacity(ncells * points_per_cell),
            point_indices_to_ids: Vec::with_capacity(npoints),
            point_ids_to_indices: HashMap::new(),
            cell_indices_to_ids: Vec::with_capacity(ncells),
            cell_ids_to_indices: HashMap::new(),
        }
    }

    fn add_point(&mut self, id: usize, data: [T; GDIM]) {
        if self.point_indices_to_ids.contains(&id) {
            panic!("Cannot add point with duplicate id.");
        }
        self.point_ids_to_indices
            .insert(id, self.point_indices_to_ids.len());
        self.point_indices_to_ids.push(id);
        self.points.extend_from_slice(&data);
    }

    fn add_cell(&mut self, id: usize, cell_data: Vec<usize>) {
        if self.cell_indices_to_ids.contains(&id) {
            panic!("Cannot add cell with duplicate id.");
        }
        assert_eq!(cell_data.len(), self.points_per_cell);
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
        points.fill_from(rlst_array_from_slice2!(&self.points, [npts, 3], [3, 1]));
        SingleElementGrid::new(
            points,
            &self.cells,
            self.element_data.0,
            self.element_data.1,
            self.point_indices_to_ids,
            self.point_ids_to_indices,
            self.cell_indices_to_ids,
            self.cell_ids_to_indices,
            None,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic]
    fn test_duplicate_point_id() {
        let mut b = SingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Triangle, 1));

        b.add_point(2, [0.0, 0.0, 0.0]);
        b.add_point(0, [1.0, 0.0, 0.0]);
        b.add_point(1, [0.0, 1.0, 0.0]);
        b.add_point(2, [1.0, 1.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn test_duplicate_cell_id() {
        let mut b = SingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Triangle, 1));

        b.add_point(0, [0.0, 0.0, 0.0]);
        b.add_point(1, [1.0, 0.0, 0.0]);
        b.add_point(2, [0.0, 1.0, 0.0]);
        b.add_point(3, [1.0, 1.0, 0.0]);

        b.add_cell(0, vec![0, 1, 2]);
        b.add_cell(0, vec![1, 2, 3]);
    }
}
