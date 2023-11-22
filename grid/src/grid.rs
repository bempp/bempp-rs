//! A serial implementation of a grid
use bempp_element::cell;
use bempp_element::element::{create_element, CiarletElement};
use bempp_tools::arrays::{zero_matrix, AdjacencyList, Array4D, Mat};
use bempp_traits::arrays::{AdjacencyListAccess, Array4DAccess};
use bempp_traits::cell::{ReferenceCell, ReferenceCellType};
use bempp_traits::element::{Continuity, ElementFamily, FiniteElement};
use bempp_traits::grid::{Geometry, GeometryEvaluator, Grid, Ownership, Topology};
use itertools::izip;
use rlst_common::traits::{
    RandomAccessByRef, RandomAccessMut, Shape, UnsafeRandomAccessByRef, UnsafeRandomAccessMut,
};
use rlst_proc_macro::rlst_static_array;
use std::cell::RefCell;
use std::ptr;

pub struct EvaluatorTdim2Gdim3<'a> {
    geometry: &'a SerialGeometry,
    points: &'a Mat<f64>,
    table: Array4D<f64>,
    npts: usize,
    axes: RefCell<Mat<f64>>,
    js: RefCell<Mat<f64>>,
}

impl<'a> EvaluatorTdim2Gdim3<'a> {
    pub fn new(
        geometry: &'a SerialGeometry,
        element: &impl FiniteElement,
        points: &'a Mat<f64>,
    ) -> Self {
        let npts = points.shape()[0];
        assert_eq!(points.shape()[1], 2);
        assert_eq!(geometry.dim(), 3);
        let mut table = Array4D::<f64>::new(element.tabulate_array_shape(1, npts));
        element.tabulate(points, 1, &mut table);
        let axes = RefCell::new(zero_matrix([2, 3]));
        let js = RefCell::new(zero_matrix([npts, 6]));

        Self {
            geometry,
            points,
            table,
            npts,
            axes,
            js,
        }
    }
}

impl<'a> GeometryEvaluator<Mat<f64>, Mat<f64>> for EvaluatorTdim2Gdim3<'a> {
    fn points(&self) -> &Mat<f64> {
        self.points
    }

    fn compute_points(&self, cell_index: usize, points: &mut Mat<f64>) {
        for i in 0..3 {
            for p in 0..self.npts {
                unsafe {
                    *points.get_unchecked_mut([p, i]) = 0.0;
                }
            }
        }
        for i in 0..self.table.shape().2 {
            let v = unsafe { *self.geometry.cells.get_unchecked(cell_index, i) };
            for j in 0..3 {
                for p in 0..self.npts {
                    unsafe {
                        *points.get_unchecked_mut([p, j]) +=
                            *self.geometry.coordinate_unchecked(v, j)
                                * *self.table.get(0, p, i, 0).unwrap();
                    }
                }
            }
        }
    }

    fn compute_normals(&self, cell_index: usize, normals: &mut Mat<f64>) {
        let mut axes = self.axes.borrow_mut();
        for p in 0..self.npts {
            for i in 0..2 {
                for j in 0..3 {
                    unsafe {
                        *axes.get_unchecked_mut([i, j]) = 0.0;
                    }
                }
            }
            for i in 0..self.table.shape().2 {
                let v = unsafe { *self.geometry.cells.get_unchecked(cell_index, i) };
                for j in 0..3 {
                    unsafe {
                        *axes.get_unchecked_mut([0, j]) +=
                            *self.geometry.coordinate_unchecked(v, j)
                                * self.table.get(1, p, i, 0).unwrap();
                        *axes.get_unchecked_mut([1, j]) +=
                            *self.geometry.coordinate_unchecked(v, j)
                                * self.table.get(2, p, i, 0).unwrap();
                    }
                }
            }
            unsafe {
                *normals.get_unchecked_mut([p, 0]) = *axes.get_unchecked([0, 1])
                    * *axes.get_unchecked([1, 2])
                    - *axes.get_unchecked([0, 2]) * *axes.get_unchecked([1, 1]);
                *normals.get_unchecked_mut([p, 1]) = *axes.get_unchecked([0, 2])
                    * *axes.get_unchecked([1, 0])
                    - *axes.get_unchecked([0, 0]) * *axes.get_unchecked([1, 2]);
                *normals.get_unchecked_mut([p, 2]) = *axes.get_unchecked([0, 0])
                    * *axes.get_unchecked([1, 1])
                    - *axes.get_unchecked([0, 1]) * *axes.get_unchecked([1, 0]);
                let size = (*normals.get_unchecked([p, 0]) * *normals.get_unchecked([p, 0])
                    + *normals.get_unchecked([p, 1]) * *normals.get_unchecked([p, 1])
                    + *normals.get_unchecked([p, 2]) * *normals.get_unchecked([p, 2]))
                .sqrt();
                *normals.get_unchecked_mut([p, 0]) /= size;
                *normals.get_unchecked_mut([p, 1]) /= size;
                *normals.get_unchecked_mut([p, 2]) /= size;
            }
        }
    }

    fn compute_jacobians(&self, cell_index: usize, jacobians: &mut Mat<f64>) {
        for i in 0..6 {
            for p in 0..self.npts {
                unsafe {
                    *jacobians.get_unchecked_mut([p, i]) = 0.0;
                }
            }
        }
        for i in 0..self.table.shape().2 {
            let v = unsafe { *self.geometry.cells.get_unchecked(cell_index, i) };
            for j in 0..3 {
                for k in 0..2 {
                    for p in 0..self.npts {
                        unsafe {
                            *jacobians.get_unchecked_mut([p, k + 2 * j]) +=
                                *self.geometry.coordinate_unchecked(v, j)
                                    * self.table.get(k + 1, p, i, 0).unwrap();
                        }
                    }
                }
            }
        }
    }

    fn compute_jacobian_determinants(&self, cell_index: usize, jdets: &mut [f64]) {
        let mut js = self.js.borrow_mut();
        self.compute_jacobians(cell_index, &mut js);
        for (p, jdet) in jdets.iter_mut().enumerate() {
            unsafe {
                *jdet = ((js.get_unchecked([p, 0]).powi(2)
                    + js.get_unchecked([p, 2]).powi(2)
                    + js.get_unchecked([p, 4]).powi(2))
                    * (js.get_unchecked([p, 1]).powi(2)
                        + js.get_unchecked([p, 3]).powi(2)
                        + js.get_unchecked([p, 5]).powi(2))
                    - (js.get_unchecked([p, 0]) * js.get_unchecked([p, 1])
                        + js.get_unchecked([p, 2]) * js.get_unchecked([p, 3])
                        + js.get_unchecked([p, 4]) * js.get_unchecked([p, 5]))
                    .powi(2))
                .sqrt();
            }
        }
    }
    fn compute_jacobian_inverses(&self, _cell_index: usize, _jinvs: &mut Mat<f64>) {
        panic!("Not implemented yet");
    }
}

pub struct LinearSimplexEvaluatorTdim2Gdim3<'a> {
    geometry: &'a SerialGeometry,
    points: &'a Mat<f64>,
    table: Array4D<f64>,
    npts: usize,
    axes: RefCell<Mat<f64>>,
    js: RefCell<Vec<f64>>,
}

impl<'a> LinearSimplexEvaluatorTdim2Gdim3<'a> {
    pub fn new(
        geometry: &'a SerialGeometry,
        element: &impl FiniteElement,
        points: &'a Mat<f64>,
    ) -> Self {
        let npts = points.shape()[0];
        assert_eq!(points.shape()[1], 2);
        assert_eq!(geometry.dim(), 3);
        let mut table = Array4D::<f64>::new(element.tabulate_array_shape(1, npts));
        element.tabulate(points, 1, &mut table);
        let axes = RefCell::new(zero_matrix([2, 3]));
        let js = RefCell::new(vec![0.0; 6]);

        Self {
            geometry,
            points,
            table,
            npts,
            axes,
            js,
        }
    }
}

impl<'a> LinearSimplexEvaluatorTdim2Gdim3<'a> {
    fn single_jacobian(&self, cell_index: usize) {
        let mut js = self.js.borrow_mut();
        for i in 0..6 {
            unsafe {
                *js.get_unchecked_mut(i) = 0.0;
            }
        }
        for i in 0..self.table.shape().2 {
            let v = unsafe { *self.geometry.cells.get_unchecked(cell_index, i) };
            for j in 0..3 {
                for k in 0..2 {
                    unsafe {
                        *js.get_unchecked_mut(k + 2 * j) +=
                            *self.geometry.coordinate_unchecked(v, j)
                                * self.table.get(k + 1, 0, i, 0).unwrap();
                    }
                }
            }
        }
    }
}

impl<'a> GeometryEvaluator<Mat<f64>, Mat<f64>> for LinearSimplexEvaluatorTdim2Gdim3<'a> {
    fn points(&self) -> &Mat<f64> {
        self.points
    }

    fn compute_points(&self, cell_index: usize, points: &mut Mat<f64>) {
        for i in 0..3 {
            for p in 0..self.npts {
                unsafe {
                    *points.get_unchecked_mut([p, i]) = 0.0;
                }
            }
        }
        for i in 0..self.table.shape().2 {
            let v = unsafe { *self.geometry.cells.get_unchecked(cell_index, i) };
            for j in 0..3 {
                for p in 0..self.npts {
                    unsafe {
                        *points.get_unchecked_mut([p, j]) +=
                            *self.geometry.coordinate_unchecked(v, j)
                                * *self.table.get(0, p, i, 0).unwrap();
                    }
                }
            }
        }
    }

    fn compute_normals(&self, cell_index: usize, normals: &mut Mat<f64>) {
        let mut axes = self.axes.borrow_mut();
        for j in 0..3 {
            for i in 0..2 {
                unsafe {
                    *axes.get_unchecked_mut([i, j]) = 0.0;
                }
            }
        }
        for i in 0..self.table.shape().2 {
            let v = unsafe { *self.geometry.cells.get_unchecked(cell_index, i) };
            for j in 0..3 {
                for k in 0..2 {
                    unsafe {
                        *axes.get_unchecked_mut([k, j]) +=
                            *self.geometry.coordinate_unchecked(v, j)
                                * self.table.get(k + 1, 0, i, 0).unwrap();
                    }
                }
            }
        }
        unsafe {
            *normals.get_unchecked_mut([0, 0]) = *axes.get_unchecked([0, 1])
                * *axes.get_unchecked([1, 2])
                - *axes.get_unchecked([0, 2]) * *axes.get_unchecked([1, 1]);
            *normals.get_unchecked_mut([0, 1]) = *axes.get_unchecked([0, 2])
                * *axes.get_unchecked([1, 0])
                - *axes.get_unchecked([0, 0]) * *axes.get_unchecked([1, 2]);
            *normals.get_unchecked_mut([0, 2]) = *axes.get_unchecked([0, 0])
                * *axes.get_unchecked([1, 1])
                - *axes.get_unchecked([0, 1]) * *axes.get_unchecked([1, 0]);
            let size = ((*normals.get_unchecked([0, 0])).powi(2)
                + (*normals.get_unchecked([0, 1])).powi(2)
                + (*normals.get_unchecked([0, 2])).powi(2))
            .sqrt();
            *normals.get_unchecked_mut([0, 0]) /= size;
            *normals.get_unchecked_mut([0, 1]) /= size;
            *normals.get_unchecked_mut([0, 2]) /= size;
        }
        for i in 0..3 {
            for p in 1..self.npts {
                unsafe {
                    *normals.get_unchecked_mut([p, i]) = *normals.get_unchecked([0, i]);
                }
            }
        }
    }

    fn compute_jacobians(&self, cell_index: usize, jacobians: &mut Mat<f64>) {
        self.single_jacobian(cell_index);
        let js = self.js.borrow();
        for i in 0..6 {
            for p in 0..self.npts {
                unsafe {
                    *jacobians.get_unchecked_mut([p, i]) = *js.get_unchecked(i);
                }
            }
        }
    }

    fn compute_jacobian_determinants(&self, cell_index: usize, jdets: &mut [f64]) {
        self.single_jacobian(cell_index);
        let js = self.js.borrow();
        let d = unsafe {
            ((js.get_unchecked(0).powi(2)
                + js.get_unchecked(2).powi(2)
                + js.get_unchecked(4).powi(2))
                * (js.get_unchecked(1).powi(2)
                    + js.get_unchecked(3).powi(2)
                    + js.get_unchecked(5).powi(2))
                - (js.get_unchecked(0) * js.get_unchecked(1)
                    + js.get_unchecked(2) * js.get_unchecked(3)
                    + js.get_unchecked(4) * js.get_unchecked(5))
                .powi(2))
            .sqrt()
        };
        for jdet in jdets.iter_mut() {
            *jdet = d;
        }
    }

    fn compute_jacobian_inverses(&self, _cell_index: usize, _jinvs: &mut Mat<f64>) {
        panic!("Not implemented yet");
    }
}

/// Geometry of a serial grid
pub struct SerialGeometry {
    coordinate_elements: Vec<CiarletElement>,
    coordinates: Mat<f64>,
    cells: AdjacencyList<usize>,
    element_changes: Vec<usize>,
    index_map: Vec<usize>,
}

fn element_from_npts(cell_type: ReferenceCellType, npts: usize) -> CiarletElement {
    create_element(
        ElementFamily::Lagrange,
        cell_type,
        match cell_type {
            ReferenceCellType::Triangle => (((1 + 8 * npts) as f64).sqrt() as usize - 1) / 2 - 1,
            ReferenceCellType::Quadrilateral => (npts as f64).sqrt() as usize - 1,
            _ => {
                panic!("Unsupported cell type (for now)");
            }
        },
        Continuity::Continuous,
    )
}

impl SerialGeometry {
    pub fn new(
        coordinates: Mat<f64>,
        cells: &AdjacencyList<usize>,
        cell_types: &[ReferenceCellType],
    ) -> Self {
        let mut index_map = vec![];
        let mut element_changes = vec![];
        let mut coordinate_elements = vec![];
        let mut new_cells = AdjacencyList::<usize>::new();
        for (i, cell) in cells.iter_rows().enumerate() {
            if !index_map.contains(&i) {
                let cell_type = cell_types[i];
                let npts = cell.len();

                element_changes.push(index_map.len());
                coordinate_elements.push(element_from_npts(cell_type, npts));
                for (j, cell_j) in cells.iter_rows().enumerate() {
                    if cell_type == cell_types[j] && npts == cell_j.len() {
                        new_cells.add_row(cells.row(j).unwrap());
                        index_map.push(j);
                    }
                }
            }
        }

        Self {
            coordinate_elements,
            coordinates,
            cells: new_cells,
            element_changes,
            index_map,
        }
    }

    /// TODO: document
    pub fn coordinate_elements(&self) -> &Vec<CiarletElement> {
        &self.coordinate_elements
    }

    /// TODO: document
    pub fn element_changes(&self) -> &Vec<usize> {
        &self.element_changes
    }

    /// Get the coordinate element associated with the given cell
    pub fn element(&self, cell: usize) -> &CiarletElement {
        for i in 0..self.element_changes.len() - 1 {
            if cell < self.element_changes[i + 1] {
                return &self.coordinate_elements[i - 1];
            }
        }
        &self.coordinate_elements[self.element_changes.len() - 1]
    }
}

impl SerialGeometry {
    unsafe fn coordinate_unchecked(&self, point_index: usize, coord_index: usize) -> &f64 {
        self.coordinates.get_unchecked([point_index, coord_index])
    }
}
impl Geometry for SerialGeometry {
    type T = Mat<f64>;
    type TMut = Mat<f64>;
    fn dim(&self) -> usize {
        self.coordinates.shape()[1]
    }

    fn coordinate(&self, point_index: usize, coord_index: usize) -> Option<&f64> {
        if point_index >= self.point_count() || coord_index >= self.dim() {
            None
        } else {
            Some(unsafe { self.coordinate_unchecked(point_index, coord_index) })
        }
    }

    fn point_count(&self) -> usize {
        self.coordinates.shape()[0]
    }

    fn cell_vertices(&self, index: usize) -> Option<&[usize]> {
        self.cells.row(index)
    }
    fn cell_count(&self) -> usize {
        self.index_map.len()
    }
    fn index_map(&self) -> &[usize] {
        &self.index_map
    }

    fn get_evaluator<'a>(
        &'a self,
        element: &impl FiniteElement,
        points: &'a Self::T,
    ) -> Box<dyn GeometryEvaluator<Self::T, Self::TMut> + 'a> {
        if element.degree() == 1 && cell::create_cell(element.cell_type()).is_simplex() {
            Box::new(LinearSimplexEvaluatorTdim2Gdim3::new(self, element, points))
        } else {
            Box::new(EvaluatorTdim2Gdim3::new(self, element, points))
        }
    }

    fn compute_points<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
    >(
        &self,
        points: &T,
        cell: usize,
        physical_points: &mut TMut,
    ) {
        let npts = points.shape()[0];
        let gdim = self.dim();
        if physical_points.shape()[0] != npts {
            panic!("physical_points has wrong number of rows.");
        }
        if physical_points.shape()[1] != gdim {
            panic!("physical_points has wrong number of columns.");
        }
        let element = self.element(cell);
        let mut data = Array4D::<f64>::new(element.tabulate_array_shape(0, npts));
        element.tabulate(points, 0, &mut data);
        for p in 0..npts {
            for i in 0..gdim {
                *physical_points.get_mut([p, i]).unwrap() = 0.0;
            }
        }
        for i in 0..data.shape().2 {
            let v = *self.cells.get(cell, i).unwrap();
            for j in 0..gdim {
                for p in 0..npts {
                    *physical_points.get_mut([p, j]).unwrap() +=
                        *self.coordinate(v, j).unwrap() * data.get(0, p, i, 0).unwrap();
                }
            }
        }
    }
    fn compute_normals<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
    >(
        &self,
        points: &T,
        cell: usize,
        normals: &mut TMut,
    ) {
        let npts = points.shape()[0];
        let tdim = points.shape()[1];
        let gdim = self.dim();
        if gdim != 3 {
            unimplemented!("normals currently only implemented for 2D cells embedded in 3D.");
        }
        if normals.shape()[0] != npts {
            panic!("normals has wrong number of columns.");
        }
        if normals.shape()[1] != gdim {
            panic!("normals has wrong number of rows.");
        }
        let element = self.element(cell);
        let mut data = Array4D::<f64>::new(element.tabulate_array_shape(1, npts));
        let mut axes = rlst_static_array![f64, 2, 3];
        element.tabulate(points, 1, &mut data);
        for p in 0..npts {
            for i in 0..tdim {
                for j in 0..gdim {
                    *axes.get_mut([i, j]).unwrap() = 0.0;
                }
            }
            for i in 0..data.shape().2 {
                let v = *self.cells.get(cell, i).unwrap();
                for j in 0..gdim {
                    *axes.get_mut([0, j]).unwrap() +=
                        *self.coordinate(v, j).unwrap() * data.get(1, p, i, 0).unwrap();
                    *axes.get_mut([1, j]).unwrap() +=
                        *self.coordinate(v, j).unwrap() * data.get(2, p, i, 0).unwrap();
                }
            }
            *normals.get_mut([p, 0]).unwrap() = *axes.get([0, 1]).unwrap()
                * *axes.get([1, 2]).unwrap()
                - *axes.get([0, 2]).unwrap() * *axes.get([1, 1]).unwrap();
            *normals.get_mut([p, 1]).unwrap() = *axes.get([0, 2]).unwrap()
                * *axes.get([1, 0]).unwrap()
                - *axes.get([0, 0]).unwrap() * *axes.get([1, 2]).unwrap();
            *normals.get_mut([p, 2]).unwrap() = *axes.get([0, 0]).unwrap()
                * *axes.get([1, 1]).unwrap()
                - *axes.get([0, 1]).unwrap() * *axes.get([1, 0]).unwrap();
            let size = (*normals.get([p, 0]).unwrap() * *normals.get([p, 0]).unwrap()
                + *normals.get([p, 1]).unwrap() * *normals.get([p, 1]).unwrap()
                + *normals.get([p, 2]).unwrap() * *normals.get([p, 2]).unwrap())
            .sqrt();
            *normals.get_mut([p, 0]).unwrap() /= size;
            *normals.get_mut([p, 1]).unwrap() /= size;
            *normals.get_mut([p, 2]).unwrap() /= size;
        }
    }
    fn compute_jacobians<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
    >(
        &self,
        points: &T,
        cell: usize,
        jacobians: &mut TMut,
    ) {
        let npts = points.shape()[0];
        let tdim = points.shape()[1];
        let gdim = self.dim();
        if jacobians.shape()[0] != npts {
            panic!("jacobians has wrong number of rows.");
        }
        if jacobians.shape()[1] != gdim * tdim {
            panic!("jacobians has wrong number of columns.");
        }
        let element = self.element(cell);
        let mut data = Array4D::<f64>::new(element.tabulate_array_shape(1, npts));
        let tdim = data.shape().0 - 1;
        element.tabulate(points, 1, &mut data);
        for p in 0..npts {
            for i in 0..jacobians.shape()[1] {
                *jacobians.get_mut([p, i]).unwrap() = 0.0;
            }
        }
        for i in 0..data.shape().2 {
            let v = *self.cells.get(cell, i).unwrap();
            for p in 0..npts {
                for j in 0..gdim {
                    for k in 0..tdim {
                        *jacobians.get_mut([p, k + tdim * j]).unwrap() +=
                            *self.coordinate(v, j).unwrap() * data.get(k + 1, p, i, 0).unwrap();
                    }
                }
            }
        }
    }
    fn compute_jacobian_determinants<T: RandomAccessByRef<2, Item = f64> + Shape<2>>(
        &self,
        points: &T,
        cell: usize,
        jacobian_determinants: &mut [f64],
    ) {
        let npts = points.shape()[0];
        let tdim = points.shape()[1];
        let gdim = self.dim();
        if points.shape()[0] != jacobian_determinants.len() {
            panic!("jacobian_determinants has wrong length.");
        }
        let mut js = zero_matrix([npts, gdim * tdim]);
        self.compute_jacobians(points, cell, &mut js);

        for (p, jdet) in jacobian_determinants.iter_mut().enumerate() {
            *jdet = match tdim {
                1 => match gdim {
                    1 => *js.get([p, 0]).unwrap(),
                    2 => ((*js.get([p, 0]).unwrap()).powi(2) + (*js.get([p, 1]).unwrap()).powi(2))
                        .sqrt(),
                    3 => ((*js.get([p, 0]).unwrap()).powi(2)
                        + (*js.get([p, 1]).unwrap()).powi(2)
                        + (*js.get([p, 2]).unwrap()).powi(2))
                    .sqrt(),
                    _ => {
                        panic!("Unsupported dimensions.");
                    }
                },
                2 => match gdim {
                    2 => {
                        *js.get([p, 0]).unwrap() * *js.get([p, 3]).unwrap()
                            - *js.get([p, 1]).unwrap() * *js.get([p, 2]).unwrap()
                    }
                    3 => (((*js.get([p, 0]).unwrap()).powi(2)
                        + (*js.get([p, 2]).unwrap()).powi(2)
                        + (*js.get([p, 4]).unwrap()).powi(2))
                        * ((*js.get([p, 1]).unwrap()).powi(2)
                            + (*js.get([p, 3]).unwrap()).powi(2)
                            + (*js.get([p, 5]).unwrap()).powi(2))
                        - (*js.get([p, 0]).unwrap() * *js.get([p, 1]).unwrap()
                            + *js.get([p, 2]).unwrap() * *js.get([p, 3]).unwrap()
                            + *js.get([p, 4]).unwrap() * *js.get([p, 5]).unwrap())
                        .powi(2))
                    .sqrt(),
                    _ => {
                        panic!("Unsupported dimensions.");
                    }
                },
                3 => match gdim {
                    3 => {
                        *js.get([p, 0]).unwrap()
                            * (*js.get([p, 4]).unwrap() * *js.get([p, 8]).unwrap()
                                - *js.get([p, 5]).unwrap() * *js.get([p, 7]).unwrap())
                            - *js.get([p, 1]).unwrap()
                                * (*js.get([p, 3]).unwrap() * *js.get([p, 8]).unwrap()
                                    - *js.get([p, 5]).unwrap() * *js.get([p, 6]).unwrap())
                            + *js.get([p, 2]).unwrap()
                                * (*js.get([p, 3]).unwrap() * *js.get([p, 7]).unwrap()
                                    - *js.get([p, 4]).unwrap() * *js.get([p, 6]).unwrap())
                    }
                    _ => {
                        panic!("Unsupported dimensions.");
                    }
                },
                _ => {
                    panic!("Unsupported dimensions.");
                }
            }
        }
    }
    fn compute_jacobian_inverses<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
    >(
        &self,
        points: &T,
        cell: usize,
        jacobian_inverses: &mut TMut,
    ) {
        let npts = points.shape()[0];
        let gdim = self.dim();
        let tdim = points.shape()[1];
        if jacobian_inverses.shape()[0] != npts {
            panic!("jacobian_inverses has wrong number of rows.");
        }
        if jacobian_inverses.shape()[1] != gdim * tdim {
            panic!("jacobian_inverses has wrong number of columns.");
        }
        let element = self.element(cell);
        if element.cell_type() == ReferenceCellType::Triangle
            && element.family() == ElementFamily::Lagrange
            && element.degree() == 1
        {
            // Map is affine
            let mut js = zero_matrix([npts, gdim * tdim]);
            self.compute_jacobians(points, cell, &mut js);

            for p in 0..npts {
                if tdim == 1 {
                    if gdim == 1 {
                        *jacobian_inverses.get_mut([p, 0]).unwrap() =
                            1.0 / *js.get([p, 0]).unwrap();
                    } else if gdim == 2 {
                        unimplemented!("Inverse jacobian for this dimension not implemented yet.");
                    } else if gdim == 3 {
                        unimplemented!("Inverse jacobian for this dimension not implemented yet.");
                    } else {
                        panic!("Unsupported dimensions.");
                    }
                } else if tdim == 2 {
                    if gdim == 2 {
                        let det = *js.get([p, 0]).unwrap() * *js.get([p, 3]).unwrap()
                            - *js.get([p, 1]).unwrap() * *js.get([p, 2]).unwrap();
                        *jacobian_inverses.get_mut([p, 0]).unwrap() = js.get([p, 3]).unwrap() / det;
                        *jacobian_inverses.get_mut([p, 1]).unwrap() =
                            -js.get([p, 1]).unwrap() / det;
                        *jacobian_inverses.get_mut([p, 2]).unwrap() =
                            -js.get([p, 2]).unwrap() / det;
                        *jacobian_inverses.get_mut([p, 3]).unwrap() = js.get([p, 0]).unwrap() / det;
                    } else if gdim == 3 {
                        let c = (*js.get([p, 3]).unwrap() * *js.get([p, 4]).unwrap()
                            - *js.get([p, 2]).unwrap() * *js.get([p, 5]).unwrap())
                        .powi(2)
                            + (*js.get([p, 5]).unwrap() * *js.get([p, 0]).unwrap()
                                - *js.get([p, 4]).unwrap() * *js.get([p, 1]).unwrap())
                            .powi(2)
                            + (*js.get([p, 1]).unwrap() * *js.get([p, 2]).unwrap()
                                - *js.get([p, 0]).unwrap() * *js.get([p, 3]).unwrap())
                            .powi(2);
                        *jacobian_inverses.get_mut([p, 0]).unwrap() = (*js.get([p, 0]).unwrap()
                            * ((*js.get([p, 5]).unwrap()).powi(2)
                                + (*js.get([p, 3]).unwrap()).powi(2))
                            - *js.get([p, 1]).unwrap()
                                * (*js.get([p, 2]).unwrap() * *js.get([p, 3]).unwrap()
                                    + *js.get([p, 4]).unwrap() * *js.get([p, 5]).unwrap()))
                            / c;
                        *jacobian_inverses.get_mut([p, 1]).unwrap() = (*js.get([p, 2]).unwrap()
                            * ((*js.get([p, 1]).unwrap()).powi(2)
                                + (*js.get([p, 5]).unwrap()).powi(2))
                            - *js.get([p, 3]).unwrap()
                                * (*js.get([p, 4]).unwrap() * *js.get([p, 5]).unwrap()
                                    + *js.get([p, 0]).unwrap() * *js.get([p, 1]).unwrap()))
                            / c;
                        *jacobian_inverses.get_mut([p, 2]).unwrap() = (*js.get([p, 4]).unwrap()
                            * ((*js.get([p, 3]).unwrap()).powi(2)
                                + (*js.get([p, 1]).unwrap()).powi(2))
                            - *js.get([p, 5]).unwrap()
                                * (*js.get([p, 0]).unwrap() * *js.get([p, 1]).unwrap()
                                    + *js.get([p, 2]).unwrap() * *js.get([p, 3]).unwrap()))
                            / c;
                        *jacobian_inverses.get_mut([p, 3]).unwrap() = (*js.get([p, 1]).unwrap()
                            * ((*js.get([p, 4]).unwrap()).powi(2)
                                + (*js.get([p, 2]).unwrap()).powi(2))
                            - *js.get([p, 0]).unwrap()
                                * (*js.get([p, 2]).unwrap() * *js.get([p, 3]).unwrap()
                                    + *js.get([p, 4]).unwrap() * *js.get([p, 5]).unwrap()))
                            / c;
                        *jacobian_inverses.get_mut([p, 4]).unwrap() = (*js.get([p, 3]).unwrap()
                            * ((*js.get([p, 0]).unwrap()).powi(2)
                                + (*js.get([p, 4]).unwrap()).powi(2))
                            - *js.get([p, 2]).unwrap()
                                * (*js.get([p, 4]).unwrap() * *js.get([p, 5]).unwrap()
                                    + *js.get([p, 0]).unwrap() * *js.get([p, 1]).unwrap()))
                            / c;
                        *jacobian_inverses.get_mut([p, 5]).unwrap() = (*js.get([p, 5]).unwrap()
                            * ((*js.get([p, 2]).unwrap()).powi(2)
                                + (*js.get([p, 0]).unwrap()).powi(2))
                            - *js.get([p, 4]).unwrap()
                                * (*js.get([p, 0]).unwrap() * *js.get([p, 1]).unwrap()
                                    + *js.get([p, 2]).unwrap() * *js.get([p, 3]).unwrap()))
                            / c;
                    } else {
                        panic!("Unsupported dimensions.");
                    }
                } else if tdim == 3 {
                    if gdim == 3 {
                        unimplemented!("Inverse jacobian for this dimension not implemented yet.");
                    } else {
                        panic!("Unsupported dimensions.");
                    }
                }
            }
        } else {
            // The map is not affine, an iterative method will be needed here to approximate the inverse map.
            unimplemented!("Inverse jacobians for this cell not yet implemented.");
        }
    }
}

/// Topology of a serial grid
pub struct SerialTopology {
    dim: usize,
    connectivity: Vec<Vec<AdjacencyList<usize>>>,
    index_map: Vec<usize>,
    starts: Vec<usize>,
    cell_types: Vec<ReferenceCellType>,
}

fn get_reference_cell(cell_type: ReferenceCellType) -> Box<dyn ReferenceCell> {
    match cell_type {
        ReferenceCellType::Interval => Box::new(cell::Interval),
        ReferenceCellType::Triangle => Box::new(cell::Triangle),
        ReferenceCellType::Quadrilateral => Box::new(cell::Quadrilateral),
        _ => {
            panic!("Unsupported cell type (for now)");
        }
    }
}

unsafe impl Sync for SerialTopology {}

impl SerialTopology {
    pub fn new(cells: &AdjacencyList<usize>, cell_types: &[ReferenceCellType]) -> Self {
        let mut index_map = vec![];
        let mut vertices = vec![];
        let mut starts = vec![];
        let mut cell_types_new = vec![];
        let dim = get_reference_cell(cell_types[0]).dim();

        let mut connectivity = vec![];
        for i in 0..dim + 1 {
            connectivity.push(vec![]);
            for _j in 0..dim + 1 {
                connectivity[i].push(AdjacencyList::<usize>::new());
            }
        }

        // dim0 = dim, dim1 = 0
        for c in cell_types {
            if dim != get_reference_cell(*c).dim() {
                panic!("Grids with cells of mixed topological dimension not supported.");
            }
            if !cell_types_new.contains(c) {
                starts.push(connectivity[dim][0].num_rows());
                cell_types_new.push(*c);
                let n = get_reference_cell(*c).vertex_count();
                for (i, cell) in cells.iter_rows().enumerate() {
                    if cell_types[i] == *c {
                        index_map.push(i);
                        // Note: this hard codes that the first n points are at the vertices
                        let mut row = vec![];
                        for v in &cell[..n] {
                            if !vertices.contains(v) {
                                vertices.push(*v);
                            }
                            row.push(vertices.iter().position(|&r| r == *v).unwrap());
                        }
                        connectivity[dim][0].add_row(&row);
                    }
                }
            }
        }

        // dim1 == 0
        for dim0 in 1..dim {
            let mut cty = AdjacencyList::<usize>::new();
            let cells = &connectivity[dim][0];
            for (i, cell_type) in cell_types_new.iter().enumerate() {
                let ref_cell = get_reference_cell(*cell_type);
                let ref_entities = (0..ref_cell.entity_count(dim0))
                    .map(|x| ref_cell.connectivity(dim0, x, 0).unwrap())
                    .collect::<Vec<Vec<usize>>>();

                let cstart = starts[i];
                let cend = if i == starts.len() - 1 {
                    connectivity[2][0].num_rows()
                } else {
                    starts[i + 1]
                };
                for c in cstart..cend {
                    let cell = cells.row(c).unwrap();
                    for e in &ref_entities {
                        let vertices = e.iter().map(|x| cell[*x]).collect::<Vec<usize>>();
                        let mut found = false;
                        for entity in cty.iter_rows() {
                            if all_equal(entity, &vertices) {
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            cty.add_row(&vertices);
                        }
                    }
                }
            }
            connectivity[dim0][0] = cty;
        }

        // dim0 == dim1 == 0
        let mut nvertices = 0;
        let mut cty = AdjacencyList::<usize>::new();
        let cells = &connectivity[dim][0];
        for cell in cells.iter_rows() {
            for j in cell {
                if *j >= nvertices {
                    nvertices = *j + 1;
                }
            }
        }
        for i in 0..nvertices {
            cty.add_row(&[i]);
        }
        connectivity[0][0] = cty;

        // dim0 == dim1
        for (dim0, c) in connectivity.iter_mut().enumerate().skip(1) {
            for i in 0..c[0].num_rows() {
                c[dim0].add_row(&[i]);
            }
        }

        // dim0 == dim
        for dim1 in 1..dim + 1 {
            let mut cty = AdjacencyList::<usize>::new();
            let entities0 = &connectivity[dim][0];
            let entities1 = &connectivity[dim1][0];

            let mut sub_cell_types = vec![ReferenceCellType::Point; entities0.num_rows()];
            for (i, cell_type) in cell_types_new.iter().enumerate() {
                let ref_cell = get_reference_cell(*cell_type);
                let etypes = ref_cell.entity_types(dim);

                let cstart = starts[i];
                let cend = if i == starts.len() - 1 {
                    connectivity[2][0].num_rows()
                } else {
                    starts[i + 1]
                };
                for t in sub_cell_types.iter_mut().skip(cstart).take(cend) {
                    *t = etypes[0];
                }
            }
            for (ei, entity0) in entities0.iter_rows().enumerate() {
                let entity = get_reference_cell(sub_cell_types[ei]);
                let mut row = vec![];
                for i in 0..entity.entity_count(dim1) {
                    let vertices = entity
                        .connectivity(dim1, i, 0)
                        .unwrap()
                        .iter()
                        .map(|x| entity0[*x])
                        .collect::<Vec<usize>>();
                    for (j, entity1) in entities1.iter_rows().enumerate() {
                        if all_equal(&vertices, entity1) {
                            row.push(j);
                            break;
                        }
                    }
                }
                cty.add_row(&row);
            }
            connectivity[dim][dim1] = cty
        }

        // dim1 < dim0
        for dim1 in 1..dim + 1 {
            for dim0 in dim1 + 1..dim {
                let mut cty = AdjacencyList::<usize>::new();
                let entities0 = &connectivity[dim0][0];
                let entities1 = &connectivity[dim1][0];
                let cell_to_entities0 = &connectivity[dim][dim0];

                let mut sub_cell_types = vec![ReferenceCellType::Point; entities0.num_rows()];
                for (i, cell_type) in cell_types_new.iter().enumerate() {
                    let ref_cell = get_reference_cell(*cell_type);
                    let etypes = ref_cell.entity_types(dim0);

                    let cstart = starts[i];
                    let cend = if i == starts.len() - 1 {
                        connectivity[2][0].num_rows()
                    } else {
                        starts[i + 1]
                    };
                    for c in cstart..cend {
                        for (e, t) in izip!(cell_to_entities0.row(c).unwrap(), &etypes) {
                            sub_cell_types[*e] = *t;
                        }
                    }
                }
                for (ei, entity0) in entities0.iter_rows().enumerate() {
                    let entity = get_reference_cell(sub_cell_types[ei]);
                    let mut row = vec![];
                    for i in 0..entity.entity_count(dim1) {
                        let vertices = entity
                            .connectivity(dim1, i, 0)
                            .unwrap()
                            .iter()
                            .map(|x| entity0[*x])
                            .collect::<Vec<usize>>();
                        for (j, entity1) in entities1.iter_rows().enumerate() {
                            if all_equal(&vertices, entity1) {
                                row.push(j);
                                break;
                            }
                        }
                    }
                    cty.add_row(&row);
                }
                connectivity[dim0][dim1] = cty;
            }
        }

        // dim1 > dim0
        for dim1 in 1..dim + 1 {
            for dim0 in 0..dim1 {
                let mut data = vec![vec![]; connectivity[dim0][0].num_rows()];
                for (i, row) in connectivity[dim1][dim0].iter_rows().enumerate() {
                    for v in row {
                        data[*v].push(i);
                    }
                }
                for row in data {
                    connectivity[dim0][dim1].add_row(&row);
                }
            }
        }

        Self {
            dim,
            connectivity,
            index_map,
            starts,
            cell_types: cell_types_new,
        }
    }
}

fn all_equal(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        false
    } else {
        all_in(a, b)
    }
}

fn all_in(a: &[usize], b: &[usize]) -> bool {
    for i in a {
        if !b.contains(i) {
            return false;
        }
    }
    true
}

impl Topology<'_> for SerialTopology {
    type Connectivity = AdjacencyList<usize>;

    fn index_map(&self) -> &[usize] {
        &self.index_map
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn entity_count(&self, dim: usize) -> usize {
        self.connectivity(dim, 0).num_rows()
    }
    fn cell(&self, index: usize) -> Option<&[usize]> {
        if index < self.entity_count(self.dim) {
            Some(self.connectivity(self.dim, 0).row(index).unwrap())
        } else {
            None
        }
    }
    fn cell_type(&self, index: usize) -> Option<ReferenceCellType> {
        for (i, start) in self.starts.iter().enumerate() {
            let end = if i == self.starts.len() - 1 {
                self.connectivity[2][0].num_rows()
            } else {
                self.starts[i + 1]
            };
            if *start <= index && index < end {
                return Some(self.cell_types[i]);
            }
        }
        None
    }

    fn connectivity(&self, dim0: usize, dim1: usize) -> &Self::Connectivity {
        if dim0 > self.dim() || dim1 > self.dim() {
            panic!("Dimension of connectivity should be higher than the topological dimension");
        }
        &self.connectivity[dim0][dim1]
    }

    fn entity_ownership(&self, _dim: usize, _index: usize) -> Ownership {
        Ownership::Owned
    }
}

/// Serial grid
pub struct SerialGrid {
    topology: SerialTopology,
    geometry: SerialGeometry,
}

impl SerialGrid {
    pub fn new(
        coordinates: Mat<f64>,
        cells: AdjacencyList<usize>,
        cell_types: Vec<ReferenceCellType>,
    ) -> Self {
        let topology = SerialTopology::new(&cells, &cell_types);
        Self {
            topology,
            geometry: SerialGeometry::new(coordinates, &cells, &cell_types),
        }
    }
}
impl Grid<'_> for SerialGrid {
    type Topology = SerialTopology;

    type Geometry = SerialGeometry;

    fn topology(&self) -> &Self::Topology {
        &self.topology
    }

    fn geometry(&self) -> &Self::Geometry {
        &self.geometry
    }

    fn is_serial(&self) -> bool {
        true
    }
}

impl PartialEq for SerialGrid {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}
impl Eq for SerialGrid {}

#[cfg(test)]
mod test {
    use crate::grid::*;
    use crate::shapes::regular_sphere;
    use approx::*;
    use bempp_tools::arrays::to_matrix;

    #[test]
    fn test_connectivity() {
        let g = SerialGrid::new(
            to_matrix(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [4, 2]),
            AdjacencyList::from_data(vec![0, 1, 2, 2, 1, 3], vec![0, 3, 6]),
            vec![ReferenceCellType::Triangle; 2],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 2);
        assert_eq!(g.topology().entity_count(0), 4);
        assert_eq!(g.topology().entity_count(1), 5);
        assert_eq!(g.topology().entity_count(2), 2);
        assert_eq!(g.geometry().point_count(), 4);

        assert_eq!(g.topology().connectivity(0, 0).row(0).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(0, 0).row(0).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(0, 0).row(1).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(0, 0).row(1).unwrap()[0], 1);
        assert_eq!(g.topology().connectivity(0, 0).row(2).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(0, 0).row(2).unwrap()[0], 2);
        assert_eq!(g.topology().connectivity(0, 0).row(3).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(0, 0).row(3).unwrap()[0], 3);

        assert_eq!(g.topology().connectivity(1, 0).row(0).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(1, 0).row(0).unwrap()[0], 1);
        assert_eq!(g.topology().connectivity(1, 0).row(0).unwrap()[1], 2);
        assert_eq!(g.topology().connectivity(1, 0).row(1).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(1, 0).row(1).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(1, 0).row(1).unwrap()[1], 2);
        assert_eq!(g.topology().connectivity(1, 0).row(2).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(1, 0).row(2).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(1, 0).row(2).unwrap()[1], 1);
        assert_eq!(g.topology().connectivity(1, 0).row(3).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(1, 0).row(3).unwrap()[0], 1);
        assert_eq!(g.topology().connectivity(1, 0).row(3).unwrap()[1], 3);
        assert_eq!(g.topology().connectivity(1, 0).row(4).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(1, 0).row(4).unwrap()[0], 2);
        assert_eq!(g.topology().connectivity(1, 0).row(4).unwrap()[1], 3);

        assert_eq!(g.topology().connectivity(0, 1).row(0).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(0, 1).row(0).unwrap()[0], 1);
        assert_eq!(g.topology().connectivity(0, 1).row(0).unwrap()[1], 2);
        assert_eq!(g.topology().connectivity(0, 1).row(1).unwrap().len(), 3);
        assert_eq!(g.topology().connectivity(0, 1).row(1).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(0, 1).row(1).unwrap()[1], 2);
        assert_eq!(g.topology().connectivity(0, 1).row(1).unwrap()[2], 3);
        assert_eq!(g.topology().connectivity(0, 1).row(2).unwrap().len(), 3);
        assert_eq!(g.topology().connectivity(0, 1).row(2).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(0, 1).row(2).unwrap()[1], 1);
        assert_eq!(g.topology().connectivity(0, 1).row(2).unwrap()[2], 4);
        assert_eq!(g.topology().connectivity(0, 1).row(3).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(0, 1).row(3).unwrap()[0], 3);
        assert_eq!(g.topology().connectivity(0, 1).row(3).unwrap()[1], 4);

        assert_eq!(g.topology().connectivity(2, 0).row(0).unwrap().len(), 3);
        assert_eq!(g.topology().connectivity(2, 0).row(0).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(2, 0).row(0).unwrap()[1], 1);
        assert_eq!(g.topology().connectivity(2, 0).row(0).unwrap()[2], 2);
        assert_eq!(g.topology().connectivity(2, 0).row(1).unwrap().len(), 3);
        assert_eq!(g.topology().connectivity(2, 0).row(1).unwrap()[0], 2);
        assert_eq!(g.topology().connectivity(2, 0).row(1).unwrap()[1], 1);
        assert_eq!(g.topology().connectivity(2, 0).row(1).unwrap()[2], 3);

        assert_eq!(g.topology().connectivity(0, 2).row(0).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(0, 2).row(0).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(0, 2).row(1).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(0, 2).row(1).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(0, 2).row(1).unwrap()[1], 1);
        assert_eq!(g.topology().connectivity(0, 2).row(2).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(0, 2).row(2).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(0, 2).row(2).unwrap()[1], 1);
        assert_eq!(g.topology().connectivity(0, 2).row(3).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(0, 2).row(3).unwrap()[0], 1);

        assert_eq!(g.topology().connectivity(2, 1).row(0).unwrap().len(), 3);
        assert_eq!(g.topology().connectivity(2, 1).row(0).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(2, 1).row(0).unwrap()[1], 1);
        assert_eq!(g.topology().connectivity(2, 1).row(0).unwrap()[2], 2);
        assert_eq!(g.topology().connectivity(2, 1).row(1).unwrap().len(), 3);
        assert_eq!(g.topology().connectivity(2, 1).row(1).unwrap()[0], 3);
        assert_eq!(g.topology().connectivity(2, 1).row(1).unwrap()[1], 4);
        assert_eq!(g.topology().connectivity(2, 1).row(1).unwrap()[2], 0);

        assert_eq!(g.topology().connectivity(1, 2).row(0).unwrap().len(), 2);
        assert_eq!(g.topology().connectivity(1, 2).row(0).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(1, 2).row(0).unwrap()[1], 1);
        assert_eq!(g.topology().connectivity(1, 2).row(1).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(1, 2).row(1).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(1, 2).row(2).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(1, 2).row(2).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(1, 2).row(3).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(1, 2).row(3).unwrap()[0], 1);
        assert_eq!(g.topology().connectivity(1, 2).row(4).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(1, 2).row(4).unwrap()[0], 1);

        assert_eq!(g.topology().connectivity(2, 2).row(0).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(2, 2).row(0).unwrap()[0], 0);
        assert_eq!(g.topology().connectivity(2, 2).row(1).unwrap().len(), 1);
        assert_eq!(g.topology().connectivity(2, 2).row(1).unwrap()[0], 1);
    }

    #[test]
    fn test_serial_triangle_grid_octahedron() {
        let g = SerialGrid::new(
            to_matrix(
                &[
                    0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, -1.0,
                ],
                [6, 3],
            ),
            AdjacencyList::from_data(
                vec![
                    0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 1,
                ],
                vec![0, 3, 6, 9, 12, 15, 18, 21, 24],
            ),
            vec![ReferenceCellType::Triangle; 8],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 3);
        assert_eq!(g.topology().entity_count(0), 6);
        assert_eq!(g.topology().entity_count(1), 12);
        assert_eq!(g.topology().entity_count(2), 8);
        assert_eq!(g.geometry().point_count(), 6);
    }

    #[test]
    fn test_serial_triangle_grid_screen() {
        let g = SerialGrid::new(
            to_matrix(
                &[
                    0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0,
                    1.0, 1.0,
                ],
                [9, 2],
            ),
            AdjacencyList::from_data(
                vec![
                    0, 1, 4, 1, 2, 5, 0, 4, 3, 1, 5, 4, 3, 4, 7, 4, 5, 8, 3, 7, 6, 4, 8, 7,
                ],
                vec![0, 3, 6, 9, 12, 15, 18, 21, 24],
            ),
            vec![ReferenceCellType::Triangle; 8],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 2);
        assert_eq!(g.topology().entity_count(0), 9);
        assert_eq!(g.topology().entity_count(1), 16);
        assert_eq!(g.topology().entity_count(2), 8);
        assert_eq!(g.geometry().point_count(), 9);
    }

    #[test]
    fn test_serial_mixed_grid_screen() {
        let g = SerialGrid::new(
            to_matrix(
                &[
                    0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0,
                    1.0, 1.0,
                ],
                [9, 2],
            ),
            AdjacencyList::from_data(
                vec![0, 1, 4, 0, 4, 3, 1, 2, 4, 5, 3, 4, 7, 3, 7, 6, 4, 5, 7, 8],
                vec![0, 3, 6, 10, 13, 16, 20],
            ),
            vec![
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Quadrilateral,
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Quadrilateral,
            ],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 2);
        assert_eq!(g.topology().entity_count(0), 9);
        assert_eq!(g.topology().entity_count(1), 14);
        assert_eq!(g.topology().entity_count(2), 6);
        assert_eq!(g.geometry().point_count(), 9);
    }

    #[test]
    fn test_higher_order_grid() {
        let s = 1.0 / (2.0_f64).sqrt();
        let g = SerialGrid::new(
            to_matrix(
                &[
                    0.0, 1.0, s, 0.0, -s, -1.0, -s, 0.0, s, 0.0, 0.0, s, 1.0, s, 0.0, -s, -1.0, -s,
                ],
                [9, 2],
            ),
            AdjacencyList::from_data(vec![4, 8, 2, 1, 3, 0, 4, 6, 8, 7, 0, 5], vec![0, 6, 12]),
            vec![ReferenceCellType::Triangle, ReferenceCellType::Triangle],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 2);
        assert_eq!(g.topology().entity_count(0), 4);
        assert_eq!(g.topology().entity_count(1), 5);
        assert_eq!(g.topology().entity_count(2), 2);
        assert_eq!(g.geometry().point_count(), 9);
    }

    #[test]
    fn test_higher_order_mixed_grid() {
        let g = SerialGrid::new(
            to_matrix(
                &[
                    0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 2.0, 1.5, 0.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0,
                    0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5,
                    0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [13, 3],
            ),
            AdjacencyList::from_data(
                vec![2, 7, 12, 0, 2, 9, 11, 1, 4, 6, 10, 5, 2, 7, 11, 8, 6, 3],
                vec![0, 3, 12, 18],
            ),
            vec![
                ReferenceCellType::Triangle,
                ReferenceCellType::Quadrilateral,
                ReferenceCellType::Triangle,
            ],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 3);
        assert_eq!(g.topology().entity_count(0), 6);
        assert_eq!(g.topology().entity_count(1), 8);
        assert_eq!(g.topology().entity_count(2), 3);
        assert_eq!(g.geometry().point_count(), 13);
    }

    #[test]
    fn test_points_and_jacobians() {
        let g = SerialGrid::new(
            to_matrix(
                &[
                    2.0, 4.0, 5.0, 0.0, -1.0, 0.0, 2.0, 2.0, 3.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0,
                    1.0, 1.0, 1.0,
                ],
                [6, 3],
            ),
            AdjacencyList::from_data(vec![0, 1, 2, 3, 4, 5], vec![0, 3, 6]),
            vec![ReferenceCellType::Triangle, ReferenceCellType::Triangle],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 3);

        let points = to_matrix(
            &[0.2, 0.5, 1.0 / 3.0, 0.15, 0.0, 0.5, 1.0 / 3.0, 0.3],
            [4, 2],
        );

        // Test compute_points
        let mut physical_points = zero_matrix([points.shape()[0], 3]);
        g.geometry()
            .compute_points(&points, 0, &mut physical_points);
        assert_relative_eq!(
            *physical_points.get([0, 0]).unwrap(),
            2.4,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([0, 1]).unwrap(),
            2.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([0, 2]).unwrap(),
            0.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([1, 0]).unwrap(),
            4.5,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([1, 1]).unwrap(),
            2.5,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([1, 2]).unwrap(),
            0.5,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([2, 0]).unwrap(),
            11.0 / 3.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([2, 1]).unwrap(),
            7.0 / 3.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([2, 2]).unwrap(),
            1.0 / 3.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([3, 0]).unwrap(),
            3.2,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([3, 1]).unwrap(),
            2.3,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([3, 2]).unwrap(),
            0.3,
            max_relative = 1e-14
        );
        g.geometry()
            .compute_points(&points, 1, &mut physical_points);
        assert_relative_eq!(
            *physical_points.get([0, 0]).unwrap(),
            -0.2,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([0, 1]).unwrap(),
            0.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([0, 2]).unwrap(),
            1.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([1, 0]).unwrap(),
            -0.5,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([1, 1]).unwrap(),
            -0.5,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([1, 2]).unwrap(),
            1.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([2, 0]).unwrap(),
            -1.0 / 3.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([2, 1]).unwrap(),
            -1.0 / 3.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([2, 2]).unwrap(),
            1.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([3, 0]).unwrap(),
            -0.15,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([3, 1]).unwrap(),
            -0.3,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            *physical_points.get([3, 2]).unwrap(),
            1.0,
            max_relative = 1e-14
        );

        // Test compute_jacobians
        let mut jacobians = zero_matrix([points.shape()[0], 6]);
        g.geometry().compute_jacobians(&points, 0, &mut jacobians);
        for i in 0..3 {
            assert_relative_eq!(*jacobians.get([i, 0]).unwrap(), 2.0, max_relative = 1e-14);
            assert_relative_eq!(*jacobians.get([i, 1]).unwrap(), 3.0, max_relative = 1e-14);
            // assert_relative_eq!(*jacobians.get([i, 2]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jacobians.get([i, 3]).unwrap(), 1.0, max_relative = 1e-14);
            // assert_relative_eq!(*jacobians.get([i, 4]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jacobians.get([i, 5]).unwrap(), 1.0, max_relative = 1e-14);
        }
        g.geometry().compute_jacobians(&points, 1, &mut jacobians);
        for i in 0..3 {
            assert_relative_eq!(*jacobians.get([i, 0]).unwrap(), -1.0, max_relative = 1e-14);
            assert_relative_eq!(*jacobians.get([i, 1]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jacobians.get([i, 2]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jacobians.get([i, 3]).unwrap(), -1.0, max_relative = 1e-14);
            assert_relative_eq!(*jacobians.get([i, 4]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jacobians.get([i, 5]).unwrap(), 0.0, max_relative = 1e-14);
        }

        // test compute_jacobian_determinants
        let mut dets = vec![0.0; points.shape()[0]];
        g.geometry()
            .compute_jacobian_determinants(&points, 0, &mut dets);
        for d in &dets {
            assert_relative_eq!(*d, 2.0 * 2.0_f64.sqrt(), max_relative = 1e-14);
        }
        g.geometry()
            .compute_jacobian_determinants(&points, 1, &mut dets);
        for d in &dets {
            assert_relative_eq!(*d, 1.0, max_relative = 1e-14);
        }

        // Test compute_jacobian_inverses
        let mut jinvs = zero_matrix([points.shape()[0], 6]);
        g.geometry()
            .compute_jacobian_inverses(&points, 0, &mut jinvs);
        for i in 0..3 {
            assert_relative_eq!(*jinvs.get([i, 0]).unwrap(), 0.5, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 1]).unwrap(), -0.75, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 2]).unwrap(), -0.75, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 3]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 4]).unwrap(), 0.5, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 5]).unwrap(), 0.5, max_relative = 1e-14);
        }
        g.geometry()
            .compute_jacobian_inverses(&points, 1, &mut jinvs);
        for i in 0..3 {
            assert_relative_eq!(*jinvs.get([i, 0]).unwrap(), -1.0, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 1]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 2]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 3]).unwrap(), 0.0, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 4]).unwrap(), -1.0, max_relative = 1e-14);
            assert_relative_eq!(*jinvs.get([i, 5]).unwrap(), 0.0, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_normals() {
        let g = SerialGrid::new(
            to_matrix(
                &[
                    0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                ],
                [5, 3],
            ),
            AdjacencyList::from_data(vec![0, 1, 2, 1, 3, 2, 2, 3, 4], vec![0, 3, 6, 9]),
            vec![
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
            ],
        );

        let pt = to_matrix(&[1.0 / 3.0, 1.0 / 3.0], [1, 2]);

        let mut normal = zero_matrix([1, 3]);

        g.geometry().compute_normals(&pt, 0, &mut normal);
        assert_relative_eq!(*normal.get([0, 0]).unwrap(), 0.0);
        assert_relative_eq!(*normal.get([0, 1]).unwrap(), -1.0);
        assert_relative_eq!(*normal.get([0, 2]).unwrap(), 0.0);

        g.geometry().compute_normals(&pt, 1, &mut normal);
        let a = f64::sqrt(1.0 / 3.0);
        assert_relative_eq!(*normal.get([0, 0]).unwrap(), a);
        assert_relative_eq!(*normal.get([0, 1]).unwrap(), a);
        assert_relative_eq!(*normal.get([0, 2]).unwrap(), a);

        g.geometry().compute_normals(&pt, 2, &mut normal);
        assert_relative_eq!(*normal.get([0, 0]).unwrap(), 0.0);
        assert_relative_eq!(*normal.get([0, 1]).unwrap(), 0.0);
        assert_relative_eq!(*normal.get([0, 2]).unwrap(), 1.0);

        // Test a curved quadrilateral cell
        let curved_g = SerialGrid::new(
            to_matrix(
                &[
                    -1.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0, -1.0,
                    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                ],
                [9, 3],
            ),
            AdjacencyList::from_data(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![0, 9]),
            vec![ReferenceCellType::Quadrilateral],
        );

        let points = to_matrix(&[0.0, 0.2, 0.5, 0.7, 1.0, 0.0, 0.3, 0.9, 1.0, 0.3], [5, 2]);
        let mut normals = zero_matrix([5, 3]);

        curved_g
            .geometry()
            .compute_normals(&points, 0, &mut normals);

        assert_relative_eq!(
            *normals.get([0, 0]).unwrap(),
            2.0 * f64::sqrt(1.0 / 5.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(*normals.get([0, 1]).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            *normals.get([0, 2]).unwrap(),
            f64::sqrt(1.0 / 5.0),
            epsilon = 1e-12
        );

        assert_relative_eq!(
            *normals.get([1, 0]).unwrap(),
            1.2 * f64::sqrt(1.0 / 2.44),
            epsilon = 1e-12
        );
        assert_relative_eq!(*normals.get([1, 1]).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            *normals.get([1, 2]).unwrap(),
            f64::sqrt(1.0 / 2.44),
            epsilon = 1e-12
        );

        assert_relative_eq!(*normals.get([2, 0]).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(*normals.get([2, 1]).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(*normals.get([2, 2]).unwrap(), 1.0, epsilon = 1e-12);

        assert_relative_eq!(
            *normals.get([3, 0]).unwrap(),
            -0.8 * f64::sqrt(1.0 / 1.64),
            epsilon = 1e-12
        );
        assert_relative_eq!(*normals.get([3, 1]).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            *normals.get([3, 2]).unwrap(),
            f64::sqrt(1.0 / 1.64),
            epsilon = 1e-12
        );

        assert_relative_eq!(
            *normals.get([4, 0]).unwrap(),
            -2.0 * f64::sqrt(1.0 / 5.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(*normals.get([4, 1]).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            *normals.get([4, 2]).unwrap(),
            f64::sqrt(1.0 / 5.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_compute_points_evaluator() {
        let grid = regular_sphere(2);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let pts = to_matrix(&[0.1, 0.2, 0.6, 0.1, 0.4, 0.2], [3, 2]);
        let e = grid.geometry().get_evaluator(&element, &pts);

        let mut points0 = zero_matrix([3, 3]);
        let mut points1 = zero_matrix([3, 3]);
        for c in 0..grid.geometry().cell_count() {
            grid.geometry().compute_points(&pts, c, &mut points0);
            e.compute_points(c, &mut points1);
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(
                        *points0.get([i, j]).unwrap(),
                        *points1.get([i, j]).unwrap(),
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_normals_evaluator() {
        let grid = regular_sphere(2);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let pts = to_matrix(&[0.1, 0.2, 0.6, 0.1, 0.4, 0.2], [3, 2]);
        let e = grid.geometry().get_evaluator(&element, &pts);

        let mut normals0 = zero_matrix([3, 3]);
        let mut normals1 = zero_matrix([3, 3]);
        for c in 0..grid.geometry().cell_count() {
            grid.geometry().compute_normals(&pts, c, &mut normals0);
            e.compute_normals(c, &mut normals1);
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(
                        *normals0.get([i, j]).unwrap(),
                        *normals1.get([i, j]).unwrap(),
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_jacobians_evaluator() {
        let grid = regular_sphere(2);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let pts = to_matrix(&[0.1, 0.1, 0.2, 0.4, 0.6, 0.2], [3, 2]);
        let e = grid.geometry().get_evaluator(&element, &pts);

        let mut jacobians0 = zero_matrix([3, 6]);
        let mut jacobians1 = zero_matrix([3, 6]);
        for c in 0..grid.geometry().cell_count() {
            grid.geometry().compute_jacobians(&pts, c, &mut jacobians0);
            e.compute_jacobians(c, &mut jacobians1);
            for i in 0..3 {
                for j in 0..6 {
                    assert_relative_eq!(
                        *jacobians0.get([i, j]).unwrap(),
                        *jacobians1.get([i, j]).unwrap(),
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_jacobian_determinants_evaluator() {
        let grid = regular_sphere(2);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let pts = to_matrix(&[0.1, 0.1, 0.2, 0.4, 0.6, 0.2], [3, 2]);
        let e = grid.geometry().get_evaluator(&element, &pts);

        let mut jacobian_determinants0 = vec![0.0; 3];
        let mut jacobian_determinants1 = vec![0.0; 3];
        for c in 0..grid.geometry().cell_count() {
            grid.geometry()
                .compute_jacobian_determinants(&pts, c, &mut jacobian_determinants0);
            e.compute_jacobian_determinants(c, &mut jacobian_determinants1);
            for i in 0..3 {
                assert_relative_eq!(
                    jacobian_determinants0[i],
                    jacobian_determinants1[i],
                    epsilon = 1e-12
                );
            }
        }
    }
}
