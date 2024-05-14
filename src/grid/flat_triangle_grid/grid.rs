//! Flat triangle grid

use crate::element::ciarlet::{lagrange, CiarletElement};
use crate::element::reference_cell;
use crate::grid::common::compute_diameter_triangle;
use crate::grid::traits::{Geometry, GeometryEvaluator, Grid, Topology};
use crate::traits::element::{Continuity, FiniteElement};
use crate::traits::types::{CellLocalIndexPair, Ownership, ReferenceCellType};
use num::Float;
use rlst::rlst_static_array;
use rlst::rlst_static_type;
use rlst::RlstScalar;
use rlst::{
    dense::array::{views::ArrayViewMut, Array, SliceArray},
    rlst_array_from_slice2, BaseArray, DefaultIterator, DefaultIteratorMut, MatrixInverse,
    RandomAccessByRef, RawAccess, Shape, UnsafeRandomAccessByRef, VectorContainer,
};
use std::collections::HashMap;

/// A flat triangle grid
pub struct FlatTriangleGrid<T: Float + RlstScalar<Real = T>> {
    index_map: Vec<usize>,

    // Geometry information
    pub(crate) coordinates: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
    pub(crate) element: CiarletElement<T>,
    midpoints: Vec<rlst_static_type!(T, 3)>,
    diameters: Vec<T>,
    volumes: Vec<T>,
    pub(crate) normals: Vec<rlst_static_type!(T, 3)>,
    pub(crate) jacobians: Vec<rlst_static_type!(T, 3, 2)>,
    cell_indices: Vec<usize>,

    // Topology information
    entities_to_vertices: Vec<Vec<Vec<usize>>>,
    pub(crate) cells_to_entities: Vec<Vec<Vec<usize>>>,
    entities_to_cells: Vec<Vec<Vec<CellLocalIndexPair<usize>>>>,
    entity_types: Vec<ReferenceCellType>,

    // Point and cell ids
    point_indices_to_ids: Vec<usize>,
    point_ids_to_indices: HashMap<usize, usize>,
    cell_indices_to_ids: Vec<usize>,
    cell_ids_to_indices: HashMap<usize, usize>,
}

impl<T: Float + RlstScalar<Real = T>> FlatTriangleGrid<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    /// Create a flat triangle grid
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        coordinates: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        cells: &[usize],
        point_indices_to_ids: Vec<usize>,
        point_ids_to_indices: HashMap<usize, usize>,
        cell_indices_to_ids: Vec<usize>,
        cell_ids_to_indices: HashMap<usize, usize>,
        edge_ids: Option<HashMap<[usize; 2], usize>>,
    ) -> Self {
        assert_eq!(coordinates.shape()[1], 3);
        let ncells = cells.len() / 3;
        let nvertices = coordinates.shape()[0];

        // Compute geometry
        let mut index_map = vec![0; ncells];
        let mut midpoints = Vec::with_capacity(ncells);
        let mut diameters = Vec::with_capacity(ncells);
        let mut volumes = Vec::with_capacity(ncells);
        let mut normals = Vec::with_capacity(ncells);
        let mut jacobians = Vec::with_capacity(ncells);

        let mut a = rlst_static_array!(T, 3);
        let mut b = rlst_static_array!(T, 3);
        let mut c = rlst_static_array!(T, 3);

        let mut v0 = rlst_static_array!(T, 3);
        let mut v1 = rlst_static_array!(T, 3);
        let mut v2 = rlst_static_array!(T, 3);

        for (cell_i, index) in index_map.iter_mut().enumerate() {
            *index = cell_i;

            midpoints.push(rlst_static_array!(T, 3));
            normals.push(rlst_static_array!(T, 3));
            jacobians.push(rlst_static_array!(T, 3, 2));

            for (i, c) in v0.iter_mut().enumerate() {
                *c = unsafe { *coordinates.get_unchecked([cells[3 * cell_i], i]) };
            }
            for (i, c) in v1.iter_mut().enumerate() {
                *c = unsafe { *coordinates.get_unchecked([cells[3 * cell_i + 1], i]) };
            }
            for (i, c) in v2.iter_mut().enumerate() {
                *c = unsafe { *coordinates.get_unchecked([cells[3 * cell_i + 2], i]) };
            }

            midpoints[cell_i].fill_from(
                (v0.view() + v1.view() + v2.view()).scalar_mul(T::from(1.0 / 3.0).unwrap()),
            );

            a.fill_from(v1.view() - v0.view());
            b.fill_from(v2.view() - v0.view());
            c.fill_from(v2.view() - v1.view());
            jacobians[cell_i].view_mut().slice(1, 0).fill_from(a.view());
            jacobians[cell_i].view_mut().slice(1, 1).fill_from(b.view());

            a.cross(b.view(), normals[cell_i].view_mut());

            let normal_length = normals[cell_i].view().norm_2();
            normals[cell_i].scale_inplace(T::one() / normal_length);

            volumes.push(normal_length / T::from(2.0).unwrap());
            diameters.push(compute_diameter_triangle(v0.view(), v1.view(), v2.view()));
        }

        let element = lagrange::create(ReferenceCellType::Triangle, 1, Continuity::Continuous);
        let cell_indices = (0..ncells).collect::<Vec<_>>();

        // Compute topology
        let entity_types = vec![
            ReferenceCellType::Point,
            ReferenceCellType::Interval,
            ReferenceCellType::Triangle,
        ];

        let mut cells_to_entities = vec![vec![vec![]; ncells]; 3];
        let mut entities_to_cells = vec![vec![]; 3];
        let mut entities_to_vertices = vec![vec![]; 2];

        let mut edge_indices = HashMap::new();

        entities_to_cells[2] = vec![vec![]; ncells];
        entities_to_vertices[0] = (0..nvertices).map(|i| vec![i]).collect::<Vec<_>>();
        entities_to_cells[0] = vec![vec![]; nvertices];

        for (cell_i, i) in index_map.iter_mut().enumerate() {
            let cell = &cells[3 * cell_i..3 * (cell_i + 1)];
            *i = cell_i;
            for (local_index, v) in cell.iter().enumerate() {
                entities_to_cells[0][*v].push(CellLocalIndexPair::new(cell_i, local_index));
            }
            entities_to_cells[2][cell_i] = vec![CellLocalIndexPair::new(cell_i, 0)];
            cells_to_entities[0][cell_i].extend_from_slice(cell);
            cells_to_entities[2][cell_i] = vec![cell_i];
        }
        if let Some(e) = edge_ids {
            for (i, j) in e.into_iter() {
                edge_indices.insert((i[0], i[1]), j);
                entities_to_vertices[1].push(vec![i[0], i[1]]);
                entities_to_cells[1].push(vec![]);
            }
        }
        let ref_conn = &reference_cell::connectivity(ReferenceCellType::Triangle)[1];
        for cell_i in 0..ncells {
            let cell = &cells[3 * cell_i..3 * (cell_i + 1)];
            for (local_index, rc) in ref_conn.iter().enumerate() {
                let mut first = cell[rc[0][0]];
                let mut second = cell[rc[0][1]];
                if first > second {
                    std::mem::swap(&mut first, &mut second);
                }
                if let Some(edge_index) = edge_indices.get(&(first, second)) {
                    cells_to_entities[1][cell_i].push(*edge_index);
                    entities_to_cells[1][*edge_index]
                        .push(CellLocalIndexPair::new(cell_i, local_index));
                } else {
                    edge_indices.insert((first, second), entities_to_vertices[1].len());
                    cells_to_entities[1][cell_i].push(entities_to_vertices[1].len());
                    entities_to_cells[1].push(vec![CellLocalIndexPair::new(cell_i, local_index)]);
                    entities_to_vertices[1].push(vec![first, second]);
                }
            }
        }
        Self {
            index_map,
            coordinates,
            element,
            midpoints,
            diameters,
            volumes,
            normals,
            jacobians,
            cell_indices,
            entities_to_vertices,
            cells_to_entities,
            entities_to_cells,
            entity_types,
            point_indices_to_ids,
            point_ids_to_indices,
            cell_indices_to_ids,
            cell_ids_to_indices,
        }
    }
}

impl<T: Float + RlstScalar<Real = T>> Grid for FlatTriangleGrid<T> {
    type T = T;
    type Topology = Self;
    type Geometry = Self;

    fn topology(&self) -> &Self::Topology {
        self
    }

    fn geometry(&self) -> &Self::Geometry {
        self
    }

    fn is_serial(&self) -> bool {
        true
    }
}

impl<T: Float + RlstScalar<Real = T>> Geometry for FlatTriangleGrid<T> {
    type IndexType = usize;
    type T = T;
    type Element = CiarletElement<T>;
    type Evaluator<'a> = GeometryEvaluatorFlatTriangle<'a, T> where Self: 'a;

    fn dim(&self) -> usize {
        3
    }

    fn index_map(&self) -> &[usize] {
        &self.index_map
    }

    fn coordinate(&self, point_index: usize, coord_index: usize) -> Option<&Self::T> {
        self.coordinates.get([point_index, coord_index])
    }

    fn point_count(&self) -> usize {
        self.coordinates.shape()[0]
    }

    fn cell_points(&self, index: usize) -> Option<&[usize]> {
        if index < self.cells_to_entities[0].len() {
            Some(&self.cells_to_entities[0][index])
        } else {
            None
        }
    }

    fn cell_count(&self) -> usize {
        self.cells_to_entities[0].len()
    }

    fn cell_element(&self, index: usize) -> Option<&Self::Element> {
        if index < self.cells_to_entities[0].len() {
            Some(&self.element)
        } else {
            None
        }
    }

    fn element_count(&self) -> usize {
        1
    }
    fn element(&self, i: usize) -> Option<&Self::Element> {
        if i == 0 {
            Some(&self.element)
        } else {
            None
        }
    }
    fn cell_indices(&self, i: usize) -> Option<&[usize]> {
        if i == 0 {
            Some(&self.cell_indices)
        } else {
            None
        }
    }

    fn midpoint(&self, index: usize, point: &mut [Self::T]) {
        point.copy_from_slice(self.midpoints[index].data());
    }

    fn diameter(&self, index: usize) -> Self::T {
        self.diameters[index]
    }
    fn volume(&self, index: usize) -> Self::T {
        self.volumes[index]
    }

    fn get_evaluator<'a>(&'a self, points: &'a [Self::T]) -> GeometryEvaluatorFlatTriangle<'a, T> {
        GeometryEvaluatorFlatTriangle::<T>::new(self, points)
    }

    fn point_index_to_id(&self, index: usize) -> usize {
        self.point_indices_to_ids[index]
    }
    fn cell_index_to_id(&self, index: usize) -> usize {
        self.cell_indices_to_ids[index]
    }
    fn point_id_to_index(&self, id: usize) -> usize {
        self.point_ids_to_indices[&id]
    }
    fn cell_id_to_index(&self, id: usize) -> usize {
        self.cell_ids_to_indices[&id]
    }
}

/// Geometry evaluator for a flat triangle grid
pub struct GeometryEvaluatorFlatTriangle<'a, T: Float + RlstScalar<Real = T>> {
    grid: &'a FlatTriangleGrid<T>,
    points: SliceArray<'a, T, 2>,
}

impl<'a, T: Float + RlstScalar<Real = T>> GeometryEvaluatorFlatTriangle<'a, T> {
    /// Create a geometry evaluator
    fn new(grid: &'a FlatTriangleGrid<T>, points: &'a [T]) -> Self {
        let tdim = reference_cell::dim(grid.element.cell_type());
        assert_eq!(points.len() % tdim, 0);
        let npoints = points.len() / tdim;
        Self {
            grid,
            points: rlst_array_from_slice2!(points, [npoints, tdim]),
        }
    }
}

impl<'a, T: Float + RlstScalar<Real = T>> GeometryEvaluator
    for GeometryEvaluatorFlatTriangle<'a, T>
{
    type T = T;

    fn point_count(&self) -> usize {
        self.points.shape()[0]
    }

    fn compute_points(&self, cell_index: usize, points: &mut [T]) {
        let jacobian = &self.grid.jacobians[cell_index];
        let npts = self.points.shape()[0];
        for d in 0..3 {
            for point_index in 0..npts {
                points[d * npts + point_index] = self.grid.coordinates
                    [[self.grid.cells_to_entities[0][cell_index][0], d]]
                    + jacobian[[d, 0]] * self.points[[point_index, 0]]
                    + jacobian[[d, 1]] * self.points[[point_index, 1]];
            }
        }
    }

    fn compute_jacobians(&self, cell_index: usize, jacobians: &mut [T]) {
        let npts = self.points.shape()[0];
        for (i, j) in self.grid.jacobians[cell_index].iter().enumerate() {
            for point_index in 0..npts {
                jacobians[i * npts + point_index] = j;
            }
        }
    }

    fn compute_normals(&self, cell_index: usize, normals: &mut [T]) {
        let npts = self.points.shape()[0];
        for (i, j) in self.grid.normals[cell_index].iter().enumerate() {
            for point_index in 0..npts {
                normals[i * npts + point_index] = j;
            }
        }
    }
}

impl<T: Float + RlstScalar<Real = T>> Topology for FlatTriangleGrid<T> {
    type IndexType = usize;

    fn dim(&self) -> usize {
        2
    }
    fn index_map(&self) -> &[usize] {
        &self.index_map
    }
    fn entity_count(&self, etype: ReferenceCellType) -> usize {
        if self.entity_types.contains(&etype) {
            self.entities_to_cells[reference_cell::dim(etype)].len()
        } else {
            0
        }
    }
    fn entity_count_by_dim(&self, dim: usize) -> usize {
        self.entity_count(self.entity_types[dim])
    }
    fn cell(&self, index: usize) -> Option<&[usize]> {
        if index < self.cells_to_entities[2].len() {
            Some(&self.cells_to_entities[2][index])
        } else {
            None
        }
    }
    fn cell_type(&self, index: usize) -> Option<ReferenceCellType> {
        if index < self.cells_to_entities[2].len() {
            Some(self.entity_types[2])
        } else {
            None
        }
    }

    fn entity_types(&self, dim: usize) -> &[ReferenceCellType] {
        &self.entity_types[dim..dim + 1]
    }

    fn cell_ownership(&self, _index: usize) -> Ownership {
        Ownership::Owned
    }
    fn vertex_ownership(&self, _index: usize) -> Ownership {
        Ownership::Owned
    }
    fn edge_ownership(&self, _index: usize) -> Ownership {
        Ownership::Owned
    }
    fn cell_to_entities(&self, index: usize, dim: usize) -> Option<&[usize]> {
        if dim <= 2 && index < self.cells_to_entities[dim].len() {
            Some(&self.cells_to_entities[dim][index])
        } else {
            None
        }
    }
    fn entity_to_cells(&self, dim: usize, index: usize) -> Option<&[CellLocalIndexPair<usize>]> {
        if dim <= 2 && index < self.entities_to_cells[dim].len() {
            Some(&self.entities_to_cells[dim][index])
        } else {
            None
        }
    }

    fn entity_to_flat_cells(
        &self,
        dim: usize,
        index: Self::IndexType,
    ) -> Option<&[CellLocalIndexPair<usize>]> {
        self.entity_to_cells(dim, index)
    }

    fn entity_vertices(&self, dim: usize, index: usize) -> Option<&[usize]> {
        if dim == 2 {
            self.cell_to_entities(index, 0)
        } else if dim < 2 && index < self.entities_to_vertices[dim].len() {
            Some(&self.entities_to_vertices[dim][index])
        } else {
            None
        }
    }

    fn vertex_index_to_id(&self, index: usize) -> usize {
        self.point_indices_to_ids[index]
    }
    fn cell_index_to_id(&self, index: usize) -> usize {
        self.cell_indices_to_ids[index]
    }
    fn vertex_id_to_index(&self, id: usize) -> usize {
        self.point_ids_to_indices[&id]
    }
    fn edge_id_to_index(&self, id: usize) -> usize {
        id
    }
    fn edge_index_to_id(&self, id: usize) -> usize {
        id
    }
    fn cell_id_to_index(&self, id: usize) -> usize {
        self.cell_ids_to_indices[&id]
    }
    fn face_index_to_flat_index(&self, index: usize) -> usize {
        index
    }
    fn face_flat_index_to_index(&self, index: usize) -> usize {
        index
    }
    fn cell_types(&self) -> &[ReferenceCellType] {
        &[ReferenceCellType::Triangle]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use rlst::{rlst_dynamic_array2, rlst_dynamic_array3, RandomAccessMut, RawAccessMut};

    fn example_grid_flat() -> FlatTriangleGrid<f64> {
        //! Create a flat test grid
        let mut points = rlst_dynamic_array2!(f64, [4, 3]);
        points[[0, 0]] = 0.0;
        points[[0, 1]] = 0.0;
        points[[0, 2]] = 0.0;
        points[[1, 0]] = 1.0;
        points[[1, 1]] = 0.0;
        points[[1, 2]] = 0.0;
        points[[2, 0]] = 1.0;
        points[[2, 1]] = 1.0;
        points[[2, 2]] = 0.0;
        points[[3, 0]] = 0.0;
        points[[3, 1]] = 1.0;
        points[[3, 2]] = 0.0;
        let cells = vec![0, 1, 2, 0, 2, 3];
        FlatTriangleGrid::new(
            points,
            &cells,
            vec![0, 1, 2, 3],
            HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            vec![0, 1],
            HashMap::from([(0, 0), (1, 1)]),
            None,
        )
    }

    fn example_grid_3d() -> FlatTriangleGrid<f64> {
        //! Create a non-flat test grid
        let mut points = rlst_dynamic_array2!(f64, [4, 3]);
        points[[0, 0]] = 0.0;
        points[[0, 1]] = 0.0;
        points[[0, 2]] = 0.0;
        points[[1, 0]] = 1.0;
        points[[1, 1]] = 0.0;
        points[[1, 2]] = 1.0;
        points[[2, 0]] = 1.0;
        points[[2, 1]] = 1.0;
        points[[2, 2]] = 0.0;
        points[[3, 0]] = 0.0;
        points[[3, 1]] = 1.0;
        points[[3, 2]] = 0.0;
        let cells = vec![0, 1, 2, 0, 2, 3];
        FlatTriangleGrid::new(
            points,
            &cells,
            vec![0, 1, 2, 3],
            HashMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]),
            vec![0, 1],
            HashMap::from([(0, 0), (1, 1)]),
            None,
        )
    }

    fn triangle_points() -> Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> {
        //! Create a set of points inside the reference triangle
        let mut points = rlst_dynamic_array2!(f64, [2, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.2;
        *points.get_mut([0, 1]).unwrap() = 0.5;
        *points.get_mut([1, 0]).unwrap() = 0.6;
        *points.get_mut([1, 1]).unwrap() = 0.1;
        points
    }

    #[test]
    fn test_cell_points() {
        //! Test that the cell points are correct
        let g = example_grid_flat();
        for (cell_i, points) in [
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![1.0, 1.0, 0.0],
            ],
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0],
            ],
        ]
        .iter()
        .enumerate()
        {
            let vs = g.cell_points(cell_i).unwrap();
            for (p_i, point) in points.iter().enumerate() {
                for (c_i, coord) in point.iter().enumerate() {
                    assert_relative_eq!(
                        *coord,
                        *g.coordinate(vs[p_i], c_i).unwrap(),
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_point_flat() {
        //! Test the compute_point function of an evaluator
        let g = example_grid_flat();
        let points = triangle_points();

        let evaluator = g.get_evaluator(points.data());
        let mut mapped_points = rlst_dynamic_array2!(f64, [points.shape()[0], 3]);
        for (cell_i, pts) in [
            vec![vec![0.7, 0.5, 0.0], vec![0.7, 0.1, 0.0]],
            vec![vec![0.2, 0.7, 0.0], vec![0.6, 0.7, 0.0]],
        ]
        .iter()
        .enumerate()
        {
            evaluator.compute_points(cell_i, mapped_points.data_mut());
            for (point_i, point) in pts.iter().enumerate() {
                for (i, j) in point.iter().enumerate() {
                    assert_relative_eq!(mapped_points[[point_i, i]], *j, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_compute_point_3d() {
        //! Test the compute_point function of an evaluator
        let g = example_grid_3d();
        let points = triangle_points();
        let evaluator = g.get_evaluator(points.data());

        let mut mapped_points = rlst_dynamic_array2!(f64, [points.shape()[0], 3]);
        for (cell_i, pts) in [
            vec![vec![0.7, 0.5, 0.2], vec![0.7, 0.1, 0.6]],
            vec![vec![0.2, 0.7, 0.0], vec![0.6, 0.7, 0.0]],
        ]
        .iter()
        .enumerate()
        {
            evaluator.compute_points(cell_i, mapped_points.data_mut());
            for (point_i, point) in pts.iter().enumerate() {
                for (i, j) in point.iter().enumerate() {
                    assert_relative_eq!(mapped_points[[point_i, i]], *j, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_compute_jacobian_3d() {
        //! Test the compute_jacobian function of an evaluator
        let g = example_grid_3d();
        let points = triangle_points();
        let evaluator = g.get_evaluator(points.data());

        let mut computed_jacobians = rlst_dynamic_array3!(f64, [points.shape()[0], 3, 2]);
        for (cell_i, jacobian) in [
            vec![vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]],
            vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 0.0]],
        ]
        .iter()
        .enumerate()
        {
            evaluator.compute_jacobians(cell_i, computed_jacobians.data_mut());
            for point_i in 0..points.shape()[0] {
                for (i, row) in jacobian.iter().enumerate() {
                    for (j, entry) in row.iter().enumerate() {
                        assert_relative_eq!(
                            *entry,
                            *computed_jacobians.get([point_i, i, j]).unwrap(),
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn test_compute_normal_3d() {
        //! Test the compute_normal function of an evaluator
        let g = example_grid_3d();
        let points = triangle_points();
        let evaluator = g.get_evaluator(points.data());

        let mut computed_normals = rlst_dynamic_array2!(f64, [points.shape()[0], 3]);
        for (cell_i, normal) in [
            vec![
                -1.0 / f64::sqrt(3.0),
                1.0 / f64::sqrt(3.0),
                1.0 / f64::sqrt(3.0),
            ],
            vec![0.0, 0.0, 1.0],
        ]
        .iter()
        .enumerate()
        {
            evaluator.compute_normals(cell_i, computed_normals.data_mut());
            for point_i in 0..points.shape()[0] {
                assert_relative_eq!(
                    computed_normals[[point_i, 0]] * computed_normals[[point_i, 0]]
                        + computed_normals[[point_i, 1]] * computed_normals[[point_i, 1]]
                        + computed_normals[[point_i, 2]] * computed_normals[[point_i, 2]],
                    1.0,
                    epsilon = 1e-12
                );
                for (i, j) in normal.iter().enumerate() {
                    assert_relative_eq!(computed_normals[[point_i, i]], *j, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_midpoint_flat() {
        //! Test midpoints
        let g = example_grid_flat();

        let mut midpoint = vec![0.0; 3];
        for (cell_i, point) in [
            vec![2.0 / 3.0, 1.0 / 3.0, 0.0],
            vec![1.0 / 3.0, 2.0 / 3.0, 0.0],
        ]
        .iter()
        .enumerate()
        {
            g.midpoint(cell_i, &mut midpoint);
            for (i, j) in midpoint.iter().zip(point) {
                assert_relative_eq!(*i, *j, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_midpoint_3d() {
        //! Test midpoints
        let g = example_grid_3d();

        let mut midpoint = vec![0.0; 3];
        for (cell_i, point) in [
            vec![2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![1.0 / 3.0, 2.0 / 3.0, 0.0],
        ]
        .iter()
        .enumerate()
        {
            g.midpoint(cell_i, &mut midpoint);
            for (i, j) in midpoint.iter().zip(point) {
                assert_relative_eq!(*i, *j, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_counts() {
        //! Test the entity counts
        let g = example_grid_flat();
        assert_eq!(Topology::dim(&g), 2);
        assert_eq!(Geometry::dim(&g), 3);
        assert_eq!(g.entity_count(ReferenceCellType::Point), 4);
        assert_eq!(g.entity_count(ReferenceCellType::Interval), 5);
        assert_eq!(g.entity_count(ReferenceCellType::Triangle), 2);

        assert_eq!(g.point_count(), 4);
        assert_eq!(g.cell_count(), 2);
    }

    #[test]
    fn test_cell_entities_vertices() {
        //! Test the cell vertices
        let g = example_grid_3d();
        for (i, vertices) in [[0, 1, 2], [0, 2, 3]].iter().enumerate() {
            let c = g.cell_to_entities(i, 0).unwrap();
            assert_eq!(c.len(), 3);
            assert_eq!(c, vertices);
        }
    }

    #[test]
    fn test_cell_entities_edges() {
        //! Test the cell edges
        let g = example_grid_3d();
        for (i, edges) in [[0, 1, 2], [3, 4, 1]].iter().enumerate() {
            let c = g.cell_to_entities(i, 1).unwrap();
            assert_eq!(c.len(), 3);
            assert_eq!(c, edges);
        }
    }

    #[test]
    fn test_cell_entities_cells() {
        //! Test the cells
        let g = example_grid_3d();
        for i in 0..2 {
            let c = g.cell_to_entities(i, 2).unwrap();
            assert_eq!(c.len(), 1);
            assert_eq!(c[0], i);
        }
    }

    #[test]
    fn test_entities_to_cells_vertices() {
        //! Test the cell-to-vertex connectivity
        let g = example_grid_3d();
        let c_to_e = (0..g.entity_count(ReferenceCellType::Triangle))
            .map(|i| g.cell_to_entities(i, 0).unwrap())
            .collect::<Vec<_>>();
        let e_to_c = (0..g.entity_count(ReferenceCellType::Point))
            .map(|i| {
                g.entity_to_cells(0, i)
                    .unwrap()
                    .iter()
                    .map(|x| x.cell)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for (i, cell) in c_to_e.iter().enumerate() {
            for v in *cell {
                assert!(e_to_c[*v].contains(&i));
            }
        }
        for (i, cells) in e_to_c.iter().enumerate() {
            for c in cells {
                assert!(c_to_e[*c].contains(&i));
            }
        }
    }

    #[test]
    fn test_entities_to_cells_edges() {
        //! Test the cell-to-edge connectivity
        let g = example_grid_3d();
        let c_to_e = (0..g.entity_count(ReferenceCellType::Triangle))
            .map(|i| g.cell_to_entities(i, 1).unwrap())
            .collect::<Vec<_>>();
        let e_to_c = (0..g.entity_count(ReferenceCellType::Interval))
            .map(|i| {
                g.entity_to_cells(1, i)
                    .unwrap()
                    .iter()
                    .map(|x| x.cell)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for (i, cell) in c_to_e.iter().enumerate() {
            for v in *cell {
                assert!(e_to_c[*v].contains(&i));
            }
        }
        for (i, cells) in e_to_c.iter().enumerate() {
            for c in cells {
                assert!(c_to_e[*c].contains(&i));
            }
        }
    }

    #[test]
    fn test_diameter() {
        //! Test cell diameters
        let g = example_grid_flat();

        for cell_i in 0..2 {
            assert_relative_eq!(
                g.diameter(cell_i),
                2.0 * f64::sqrt(1.5 - f64::sqrt(2.0)),
                epsilon = 1e-12
            );
        }

        let g = example_grid_3d();

        for (cell_i, d) in [2.0 / f64::sqrt(6.0), 2.0 * f64::sqrt(1.5 - f64::sqrt(2.0))]
            .iter()
            .enumerate()
        {
            assert_relative_eq!(g.diameter(cell_i), d, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_volume() {
        //! Test cell volumes
        let g = example_grid_flat();

        for cell_i in 0..2 {
            assert_relative_eq!(g.volume(cell_i), 0.5, epsilon = 1e-12);
        }

        let g = example_grid_3d();

        for (cell_i, d) in [f64::sqrt(0.75), 0.5].iter().enumerate() {
            assert_relative_eq!(g.volume(cell_i), d, epsilon = 1e-12);
        }
    }
}
