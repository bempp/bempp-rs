// //! Grid builder

// use crate::grid::flat_triangle_grid::grid::FlatTriangleGrid;
// use num::Float;
// use rlst::prelude::*;
// use std::collections::HashMap;

// /// Grid builder for a flat triangle grid
// pub struct FlatTriangleGridBuilder<T: LinAlg + Float + RlstScalar<Real = T>> {
//     pub(crate) points: Vec<[T; 3]>,
//     pub(crate) cells: Vec<usize>,
//     pub(crate) point_indices_to_ids: Vec<usize>,
//     point_ids_to_indices: HashMap<usize, usize>,
//     pub(crate) cell_indices_to_ids: Vec<usize>,
//     cell_ids_to_indices: HashMap<usize, usize>,
// }

// impl<T: LinAlg + Float + RlstScalar<Real = T>> FlatTriangleGridBuilder<T> {
//     pub fn new() -> Self {
//         Self {
//             points: Vec::default(),
//             cells: Vec::default(),
//             point_indices_to_ids: vec![],
//             point_ids_to_indices: HashMap::new(),
//             cell_indices_to_ids: vec![],
//             cell_ids_to_indices: HashMap::new(),
//         }
//     }
//     pub fn new_with_capacity(npoints: usize, ncells: usize, _data: ()) -> Self {
//         Self {
//             points: Vec::with_capacity(npoints),
//             cells: Vec::with_capacity(3 * ncells),
//             point_indices_to_ids: Vec::with_capacity(npoints),
//             point_ids_to_indices: HashMap::new(),
//             cell_indices_to_ids: Vec::with_capacity(ncells),
//             cell_ids_to_indices: HashMap::new(),
//         }
//     }
//     pub fn add_point(&mut self, id: usize, data: [T; 3]) {
//         if self.point_indices_to_ids.contains(&id) {
//             panic!("Cannot add point with duplicate id.");
//         }
//         self.point_ids_to_indices
//             .insert(id, self.point_indices_to_ids.len());
//         self.point_indices_to_ids.push(id);
//         self.points.push(data);
//     }

//     pub fn add_cell(&mut self, id: usize, cell_data: [usize; 3]) {
//         if self.cell_indices_to_ids.contains(&id) {
//             panic!("Cannot add cell with duplicate id.");
//         }
//         self.cell_ids_to_indices
//             .insert(id, self.cell_indices_to_ids.len());
//         self.cell_indices_to_ids.push(id);
//         for id in &cell_data {
//             self.cells.push(self.point_ids_to_indices[id]);
//         }
//     }

//     pub fn create_grid(self) -> FlatTriangleGrid<T> {
//         // TODO: remove this transposing
//         let npts = self.point_indices_to_ids.len();
//         let mut points_arr = rlst_dynamic_array2!(T, [3, npts]);
//         for (mut col, point) in itertools::izip!(points_arr.col_iter_mut(), self.points) {
//             col.data_mut().copy_from_slice(&point);
//         }
//         FlatTriangleGrid::new(
//             points_arr,
//             &self.cells,
//             self.point_indices_to_ids,
//             self.point_ids_to_indices,
//             self.cell_indices_to_ids,
//             self.cell_ids_to_indices,
//             None,
//         )
//     }
// }

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     #[should_panic]
//     fn test_duplicate_point_id() {
//         let mut b = FlatTriangleGridBuilder::<f64>::new();

//         b.add_point(2, [0.0, 0.0, 0.0]);
//         b.add_point(0, [1.0, 0.0, 0.0]);
//         b.add_point(1, [0.0, 1.0, 0.0]);
//         b.add_point(2, [1.0, 1.0, 0.0]);
//     }

//     #[test]
//     #[should_panic]
//     fn test_duplicate_cell_id() {
//         let mut b = FlatTriangleGridBuilder::<f64>::new();

//         b.add_point(0, [0.0, 0.0, 0.0]);
//         b.add_point(1, [1.0, 0.0, 0.0]);
//         b.add_point(2, [0.0, 1.0, 0.0]);
//         b.add_point(3, [1.0, 1.0, 0.0]);

//         b.add_cell(0, [0, 1, 2]);
//         b.add_cell(0, [1, 2, 3]);
//     }
// }
