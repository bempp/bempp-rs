//! Grid builder

use crate::element::ciarlet::lagrange;
use crate::grid::mixed_grid::grid::MixedGrid;
use crate::traits::element::{Continuity, FiniteElement};
use crate::traits::grid::Builder;
use crate::traits::types::ReferenceCellType;
use num::Float;
use rlst::{rlst_array_from_slice2, rlst_dynamic_array2};
use rlst::{LinAlg, RlstScalar};
use std::collections::HashMap;

/// Grid builder for a mixed grid
pub struct MixedGridBuilder<const GDIM: usize, T: LinAlg + Float + RlstScalar<Real = T>> {
    pub(crate) elements_to_npoints: HashMap<(ReferenceCellType, usize), usize>,
    pub(crate) points: Vec<T>,
    pub(crate) cells: Vec<usize>,
    pub(crate) cell_types: Vec<ReferenceCellType>,
    pub(crate) cell_degrees: Vec<usize>,
    pub(crate) point_indices_to_ids: Vec<usize>,
    pub(crate) point_ids_to_indices: HashMap<usize, usize>,
    pub(crate) cell_indices_to_ids: Vec<usize>,
    pub(crate) cell_ids_to_indices: HashMap<usize, usize>,
}

impl<const GDIM: usize, T: LinAlg + Float + RlstScalar<Real = T>> Builder<GDIM>
    for MixedGridBuilder<GDIM, T>
// for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
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
        if self.point_indices_to_ids.contains(&id) {
            panic!("Cannot add point with duplicate id.");
        }
        self.point_ids_to_indices
            .insert(id, self.point_indices_to_ids.len());
        self.point_indices_to_ids.push(id);
        self.points.extend_from_slice(&data);
    }

    fn add_cell(&mut self, id: usize, cell_data: (Vec<usize>, ReferenceCellType, usize)) {
        if self.cell_indices_to_ids.contains(&id) {
            panic!("Cannot add cell with duplicate id.");
        }
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
        let mut b = MixedGridBuilder::<3, f64>::new(());

        b.add_point(2, [0.0, 0.0, 0.0]);
        b.add_point(0, [1.0, 0.0, 0.0]);
        b.add_point(1, [0.0, 1.0, 0.0]);
        b.add_point(2, [1.0, 1.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn test_duplicate_cell_id() {
        let mut b = MixedGridBuilder::<3, f64>::new(());

        b.add_point(0, [0.0, 0.0, 0.0]);
        b.add_point(1, [1.0, 0.0, 0.0]);
        b.add_point(2, [0.0, 1.0, 0.0]);
        b.add_point(3, [1.0, 1.0, 0.0]);

        b.add_cell(0, (vec![0, 1, 2], ReferenceCellType::Triangle, 1));
        b.add_cell(0, (vec![1, 2, 3], ReferenceCellType::Triangle, 1));
    }
}
