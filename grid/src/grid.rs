//! A serial implementation of a grid
use bempp_element::cell;
use bempp_element::element::{create_element, CiarletElement};
use bempp_tools::arrays::{AdjacencyList, Array2D, Array4D};
use bempp_traits::arrays::{AdjacencyListAccess, Array2DAccess, Array4DAccess};
use bempp_traits::cell::{ReferenceCell, ReferenceCellType};
use bempp_traits::element::{ElementFamily, FiniteElement};
use bempp_traits::grid::{Geometry, Grid, Ownership, Topology};
use itertools::izip;
use std::cell::{Ref, RefCell};

/// Geometry of a serial grid
pub struct SerialGeometry {
    coordinate_elements: Vec<CiarletElement>,
    coordinates: Array2D<f64>,
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
        false,
    )
}

impl SerialGeometry {
    pub fn new(
        coordinates: Array2D<f64>,
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

impl Geometry for SerialGeometry {
    fn dim(&self) -> usize {
        self.coordinates.shape().1
    }

    fn point(&self, i: usize) -> Option<&[f64]> {
        self.coordinates.row(i)
    }

    fn point_count(&self) -> usize {
        self.coordinates.shape().0
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
    fn compute_points<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        physical_points: &mut impl Array2DAccess<'a, f64>,
    ) {
        let gdim = self.dim();
        if points.shape().0 != physical_points.shape().0 {
            panic!("physical_points has wrong number of rows.");
        }
        if gdim != physical_points.shape().1 {
            panic!("physical_points has wrong number of columns.");
        }
        let element = self.element(cell);
        let mut data = Array4D::<f64>::new(element.tabulate_array_shape(0, points.shape().0)); // TODO: Memory is assigned here. Can we avoid this?
        element.tabulate(points, 0, &mut data);
        for p in 0..points.shape().0 {
            for i in 0..physical_points.shape().1 {
                unsafe {
                    *physical_points.get_unchecked_mut(p, i) = 0.0;
                }
            }
        }
        for i in 0..data.shape().2 {
            let pt = unsafe {
                self.coordinates
                    .row_unchecked(*self.cells.get_unchecked(cell, i))
            };
            for p in 0..points.shape().0 {
                for (j, pt_j) in pt.iter().enumerate() {
                    unsafe {
                        *physical_points.get_unchecked_mut(p, j) +=
                            *pt_j * data.get_unchecked(0, p, i, 0);
                    }
                }
            }
        }
    }
    fn compute_normals<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        normals: &mut impl Array2DAccess<'a, f64>,
    ) {
        let gdim = self.dim();
        if gdim != 3 {
            unimplemented!("normals currently only implemented for 2D cells embedded in 3D.");
        }
        if points.shape().0 != normals.shape().0 {
            panic!("normals has wrong number of rows.");
        }
        if gdim != normals.shape().1 {
            panic!("normals has wrong number of columns.");
        }
        let element = self.element(cell);
        let mut data = Array4D::<f64>::new(element.tabulate_array_shape(1, points.shape().0)); // TODO: Memory is assigned here. Can we avoid this?
        let mut axes = Array2D::<f64>::new((2, 3));
        element.tabulate(points, 1, &mut data);
        for p in 0..points.shape().0 {
            for i in 0..axes.shape().0 {
                for j in 0..axes.shape().1 {
                    unsafe {
                        *axes.get_unchecked_mut(i, j) = 0.0;
                    }
                }
            }
            for i in 0..data.shape().2 {
                let pt = unsafe {
                    self.coordinates
                        .row_unchecked(*self.cells.get_unchecked(cell, i))
                };
                for (j, pt_j) in pt.iter().enumerate() {
                    unsafe {
                        *axes.get_unchecked_mut(0, j) += *pt_j * data.get_unchecked(1, p, i, 0);
                        *axes.get_unchecked_mut(1, j) += *pt_j * data.get_unchecked(2, p, i, 0);
                    }
                }
            }
            unsafe {
                *normals.get_unchecked_mut(p, 0) = *axes.get_unchecked(0, 1)
                    * *axes.get_unchecked(1, 2)
                    - *axes.get_unchecked(0, 2) * *axes.get_unchecked(1, 1);
                *normals.get_unchecked_mut(p, 1) = *axes.get_unchecked(0, 2)
                    * *axes.get_unchecked(1, 0)
                    - *axes.get_unchecked(0, 0) * *axes.get_unchecked(1, 2);
                *normals.get_unchecked_mut(p, 2) = *axes.get_unchecked(0, 0)
                    * *axes.get_unchecked(1, 1)
                    - *axes.get_unchecked(0, 1) * *axes.get_unchecked(1, 0);
            }
            let size = unsafe {
                (*normals.get_unchecked(p, 0) * *normals.get_unchecked(p, 0)
                    + *normals.get_unchecked(p, 1) * *normals.get_unchecked(p, 1)
                    + *normals.get_unchecked(p, 2) * *normals.get_unchecked(p, 2))
                .sqrt()
            };
            unsafe {
                *normals.get_unchecked_mut(p, 0) /= size;
                *normals.get_unchecked_mut(p, 1) /= size;
                *normals.get_unchecked_mut(p, 2) /= size;
            }
        }
    }
    fn compute_jacobians<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        jacobians: &mut impl Array2DAccess<'a, f64>,
    ) {
        let gdim = self.dim();
        let tdim = points.shape().1;
        if points.shape().0 != jacobians.shape().0 {
            panic!("jacobians has wrong number of rows.");
        }
        if gdim * tdim != jacobians.shape().1 {
            panic!("jacobians has wrong number of columns.");
        }
        let element = self.element(cell);
        let mut data = Array4D::<f64>::new(element.tabulate_array_shape(1, points.shape().0)); // TODO: Memory is assigned here. Can we avoid this?
        let tdim = data.shape().0 - 1;
        element.tabulate(points, 1, &mut data);
        for p in 0..points.shape().0 {
            for i in 0..jacobians.shape().1 {
                unsafe {
                    *jacobians.get_unchecked_mut(p, i) = 0.0;
                }
            }
        }
        for i in 0..data.shape().2 {
            let pt = unsafe {
                self.coordinates
                    .row_unchecked(*self.cells.get_unchecked(cell, i))
            };
            for p in 0..points.shape().0 {
                for (j, pt_j) in pt.iter().enumerate() {
                    for k in 0..tdim {
                        unsafe {
                            *jacobians.get_unchecked_mut(p, k + tdim * j) +=
                                *pt_j * data.get_unchecked(k + 1, p, i, 0);
                        }
                    }
                }
            }
        }
    }
    fn compute_jacobian_determinants<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        jacobian_determinants: &mut [f64],
    ) {
        let gdim = self.dim();
        let tdim = points.shape().1;
        if points.shape().0 != jacobian_determinants.len() {
            panic!("jacobian_determinants has wrong length.");
        }
        let mut js = Array2D::<f64>::new((points.shape().0, gdim * tdim)); // TODO: Memory is assigned here. Can we avoid this?
        self.compute_jacobians(points, cell, &mut js);

        // TODO: is it faster if we move this for inside the match statement?
        for (p, jdet) in jacobian_determinants.iter_mut().enumerate() {
            *jdet = match tdim {
                1 => match gdim {
                    1 => unsafe { *js.get_unchecked(p, 0) },
                    2 => unsafe {
                        ((*js.get_unchecked(p, 0)).powi(2) + (*js.get_unchecked(p, 1)).powi(2))
                            .sqrt()
                    },
                    3 => unsafe {
                        ((*js.get_unchecked(p, 0)).powi(2)
                            + (*js.get_unchecked(p, 1)).powi(2)
                            + (*js.get_unchecked(p, 2)).powi(2))
                        .sqrt()
                    },
                    _ => {
                        panic!("Unsupported dimensions.");
                    }
                },
                2 => match gdim {
                    2 => unsafe {
                        *js.get_unchecked(p, 0) * *js.get_unchecked(p, 3)
                            - *js.get_unchecked(p, 1) * *js.get_unchecked(p, 2)
                    },
                    3 => unsafe {
                        (((*js.get_unchecked(p, 0)).powi(2)
                            + (*js.get_unchecked(p, 2)).powi(2)
                            + (*js.get_unchecked(p, 4)).powi(2))
                            * ((*js.get_unchecked(p, 1)).powi(2)
                                + (*js.get_unchecked(p, 3)).powi(2)
                                + (*js.get_unchecked(p, 5)).powi(2))
                            - (*js.get_unchecked(p, 0) * *js.get_unchecked(p, 1)
                                + *js.get_unchecked(p, 2) * *js.get_unchecked(p, 3)
                                + *js.get_unchecked(p, 4) * *js.get_unchecked(p, 5))
                            .powi(2))
                        .sqrt()
                    },
                    _ => {
                        panic!("Unsupported dimensions.");
                    }
                },
                3 => match gdim {
                    3 => unsafe {
                        *js.get_unchecked(p, 0)
                            * (*js.get_unchecked(p, 4) * *js.get_unchecked(p, 8)
                                - *js.get_unchecked(p, 5) * *js.get_unchecked(p, 7))
                            - *js.get_unchecked(p, 1)
                                * (*js.get_unchecked(p, 3) * *js.get_unchecked(p, 8)
                                    - *js.get_unchecked(p, 5) * *js.get_unchecked(p, 6))
                            + *js.get_unchecked(p, 2)
                                * (*js.get_unchecked(p, 3) * *js.get_unchecked(p, 7)
                                    - *js.get_unchecked(p, 4) * *js.get_unchecked(p, 6))
                    },
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
    fn compute_jacobian_inverses<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        jacobian_inverses: &mut impl Array2DAccess<'a, f64>,
    ) {
        let gdim = self.dim();
        let tdim = points.shape().1;
        if points.shape().0 != jacobian_inverses.shape().0 {
            panic!("jacobian_inverses has wrong number of rows.");
        }
        if gdim * tdim != jacobian_inverses.shape().1 {
            panic!("jacobian_inverses has wrong number of columns.");
        }
        let element = self.element(cell);
        if element.cell_type() == ReferenceCellType::Triangle
            && element.family() == ElementFamily::Lagrange
            && element.degree() == 1
        {
            // Map is affine
            let mut js = Array2D::<f64>::new((points.shape().0, gdim * tdim)); // TODO: Memory is assigned here. Can we avoid this?
            self.compute_jacobians(points, cell, &mut js);

            // TODO: is it faster if we move this for inside the if statement?
            for p in 0..points.shape().0 {
                if tdim == 1 {
                    if gdim == 1 {
                        unsafe {
                            *jacobian_inverses.get_unchecked_mut(p, 0) =
                                1.0 / *js.get_unchecked(p, 0);
                        }
                    } else if gdim == 2 {
                        unimplemented!("Inverse jacobian for this dimension not implemented yet.");
                    } else if gdim == 3 {
                        unimplemented!("Inverse jacobian for this dimension not implemented yet.");
                    } else {
                        panic!("Unsupported dimensions.");
                    }
                } else if tdim == 2 {
                    if gdim == 2 {
                        let det = unsafe {
                            *js.get_unchecked(p, 0) * *js.get_unchecked(p, 3)
                                - *js.get_unchecked(p, 1) * *js.get_unchecked(p, 2)
                        };
                        unsafe {
                            *jacobian_inverses.get_unchecked_mut(p, 0) =
                                js.get_unchecked(p, 3) / det;
                            *jacobian_inverses.get_unchecked_mut(p, 1) =
                                -js.get_unchecked(p, 1) / det;
                            *jacobian_inverses.get_unchecked_mut(p, 2) =
                                -js.get_unchecked(p, 2) / det;
                            *jacobian_inverses.get_unchecked_mut(p, 3) =
                                js.get_unchecked(p, 0) / det;
                        }
                    } else if gdim == 3 {
                        let c = unsafe {
                            (*js.get_unchecked(p, 3) * *js.get_unchecked(p, 4)
                                - *js.get_unchecked(p, 2) * *js.get_unchecked(p, 5))
                            .powi(2)
                                + (*js.get_unchecked(p, 5) * *js.get_unchecked(p, 0)
                                    - *js.get_unchecked(p, 4) * *js.get_unchecked(p, 1))
                                .powi(2)
                                + (*js.get_unchecked(p, 1) * *js.get_unchecked(p, 2)
                                    - *js.get_unchecked(p, 0) * *js.get_unchecked(p, 3))
                                .powi(2)
                        };
                        unsafe {
                            *jacobian_inverses.get_unchecked_mut(p, 0) = (*js.get_unchecked(p, 0)
                                * ((*js.get_unchecked(p, 5)).powi(2)
                                    + (*js.get_unchecked(p, 3)).powi(2))
                                - *js.get_unchecked(p, 1)
                                    * (*js.get_unchecked(p, 2) * *js.get_unchecked(p, 3)
                                        + *js.get_unchecked(p, 4) * *js.get_unchecked(p, 5)))
                                / c;
                            *jacobian_inverses.get_unchecked_mut(p, 1) = (*js.get_unchecked(p, 2)
                                * ((*js.get_unchecked(p, 1)).powi(2)
                                    + (*js.get_unchecked(p, 5)).powi(2))
                                - *js.get_unchecked(p, 3)
                                    * (*js.get_unchecked(p, 4) * *js.get_unchecked(p, 5)
                                        + *js.get_unchecked(p, 0) * *js.get_unchecked(p, 1)))
                                / c;
                            *jacobian_inverses.get_unchecked_mut(p, 2) = (*js.get_unchecked(p, 4)
                                * ((*js.get_unchecked(p, 3)).powi(2)
                                    + (*js.get_unchecked(p, 1)).powi(2))
                                - *js.get_unchecked(p, 5)
                                    * (*js.get_unchecked(p, 0) * *js.get_unchecked(p, 1)
                                        + *js.get_unchecked(p, 2) * *js.get_unchecked(p, 3)))
                                / c;
                            *jacobian_inverses.get_unchecked_mut(p, 3) = (*js.get_unchecked(p, 1)
                                * ((*js.get_unchecked(p, 4)).powi(2)
                                    + (*js.get_unchecked(p, 2)).powi(2))
                                - *js.get_unchecked(p, 0)
                                    * (*js.get_unchecked(p, 2) * *js.get_unchecked(p, 3)
                                        + *js.get_unchecked(p, 4) * *js.get_unchecked(p, 5)))
                                / c;
                            *jacobian_inverses.get_unchecked_mut(p, 4) = (*js.get_unchecked(p, 3)
                                * ((*js.get_unchecked(p, 0)).powi(2)
                                    + (*js.get_unchecked(p, 4)).powi(2))
                                - *js.get_unchecked(p, 2)
                                    * (*js.get_unchecked(p, 4) * *js.get_unchecked(p, 5)
                                        + *js.get_unchecked(p, 0) * *js.get_unchecked(p, 1)))
                                / c;
                            *jacobian_inverses.get_unchecked_mut(p, 5) = (*js.get_unchecked(p, 5)
                                * ((*js.get_unchecked(p, 2)).powi(2)
                                    + (*js.get_unchecked(p, 0)).powi(2))
                                - *js.get_unchecked(p, 4)
                                    * (*js.get_unchecked(p, 0) * *js.get_unchecked(p, 1)
                                        + *js.get_unchecked(p, 2) * *js.get_unchecked(p, 3)))
                                / c;
                        }
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
    connectivity: Vec<Vec<RefCell<AdjacencyList<usize>>>>,
    index_map: Vec<usize>,
    starts: Vec<usize>,
    cell_types: Vec<ReferenceCellType>,
    adjacent_cells: Vec<RefCell<Vec<(usize, usize)>>>,
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
                connectivity[i].push(RefCell::new(AdjacencyList::<usize>::new()));
            }
        }
        for c in cell_types {
            if dim != get_reference_cell(*c).dim() {
                panic!("Grids with cells of mixed topological dimension not supported.");
            }
            if !cell_types_new.contains(c) {
                starts.push(connectivity[dim][0].borrow().num_rows());
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
                        connectivity[dim][0].borrow_mut().add_row(&row);
                    }
                }
            }
        }

        let mut adjacent_cells = vec![];
        for _ in cells.iter_rows() {
            adjacent_cells.push(RefCell::new(vec![]));
        }

        Self {
            dim,
            connectivity,
            index_map,
            starts,
            cell_types: cell_types_new,
            adjacent_cells,
        }
    }

    fn compute_adjacent_cells(&self) {
        // TODO: this could be done quicker using multiplication of sparse matrices
        let tdim = self.dim();
        self.create_connectivity(tdim, 0);
        self.create_connectivity(0, tdim);
        for (n, vertices) in self.connectivity(tdim, 0).iter_rows().enumerate() {
            let mut adj: Vec<(usize, usize)> = vec![];
            for v in vertices {
                for c in self.connectivity(0, tdim).row(*v).unwrap() {
                    let mut found = false;
                    for (i, a) in adj.iter().enumerate() {
                        if a.0 == *c {
                            adj[i] = (*c, a.1 + 1);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        adj.push((*c, 1));
                    }
                }
            }
            *self.adjacent_cells[n].borrow_mut() = adj;
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
    fn cell(&self, index: usize) -> Option<Ref<[usize]>> {
        if index < self.entity_count(self.dim) {
            Some(Ref::map(self.connectivity(self.dim, 0), |x| unsafe {
                x.row_unchecked(index)
            }))
        } else {
            None
        }
    }
    fn cell_type(&self, index: usize) -> Option<ReferenceCellType> {
        for (i, start) in self.starts.iter().enumerate() {
            let end = if i == self.starts.len() - 1 {
                self.connectivity[2][0].borrow().num_rows()
            } else {
                self.starts[i + 1]
            };
            if *start <= index && index < end {
                return Some(self.cell_types[i]);
            }
        }
        None
    }
    fn create_connectivity(&self, dim0: usize, dim1: usize) {
        if dim0 > self.dim() || dim1 > self.dim() {
            panic!("Dimension of connectivity should be higher than the topological dimension");
        }

        if self.connectivity[dim0][dim1].borrow().num_rows() > 0 {
            return;
        }

        if dim0 < dim1 {
            self.create_connectivity(dim0, 0);
            self.create_connectivity(dim1, dim0);
            let mut data = vec![vec![]; self.connectivity[dim0][0].borrow().num_rows()];
            for (i, row) in self.connectivity[dim1][dim0]
                .borrow()
                .iter_rows()
                .enumerate()
            {
                for v in row {
                    data[*v].push(i);
                }
            }
            for row in data {
                self.connectivity[dim0][dim1].borrow_mut().add_row(&row);
            }
        } else if dim0 == dim1 {
            if dim0 == 0 {
                let mut nvertices = 0;
                let cells = &self.connectivity[self.dim()][0].borrow();
                for cell in cells.iter_rows() {
                    for j in cell {
                        if *j >= nvertices {
                            nvertices = *j + 1;
                        }
                    }
                }
                for i in 0..nvertices {
                    self.connectivity[0][0].borrow_mut().add_row(&[i]);
                }
            } else {
                self.create_connectivity(dim0, 0);
                for i in 0..self.connectivity[dim0][0].borrow().num_rows() {
                    self.connectivity[dim0][dim0].borrow_mut().add_row(&[i]);
                }
            }
        } else if dim1 == 0 {
            let cells = &self.connectivity[self.dim()][0].borrow();
            for (i, cell_type) in self.cell_types.iter().enumerate() {
                let ref_cell = get_reference_cell(*cell_type);
                let ref_entities = (0..ref_cell.entity_count(dim0))
                    .map(|x| ref_cell.connectivity(dim0, x, 0).unwrap())
                    .collect::<Vec<Vec<usize>>>();

                let cstart = self.starts[i];
                let cend = if i == self.starts.len() - 1 {
                    self.connectivity[2][0].borrow().num_rows()
                } else {
                    self.starts[i + 1]
                };
                for c in cstart..cend {
                    let cell = unsafe { cells.row_unchecked(c) };
                    for e in &ref_entities {
                        let vertices = e.iter().map(|x| cell[*x]).collect::<Vec<usize>>();
                        let mut found = false;
                        for entity in self.connectivity[dim0][0].borrow().iter_rows() {
                            if all_equal(entity, &vertices) {
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            self.connectivity[dim0][0].borrow_mut().add_row(&vertices);
                        }
                    }
                }
            }
        } else {
            self.create_connectivity(dim0, 0);
            self.create_connectivity(dim1, 0);
            self.create_connectivity(self.dim(), dim0);
            let entities0 = &self.connectivity[dim0][0].borrow();
            let entities1 = &self.connectivity[dim1][0].borrow();
            let cell_to_entities0 = &self.connectivity[self.dim()][dim0].borrow();

            let mut cell_types = vec![ReferenceCellType::Point; entities0.num_rows()];
            for (i, cell_type) in self.cell_types.iter().enumerate() {
                let ref_cell = get_reference_cell(*cell_type);
                let etypes = ref_cell.entity_types(dim0);

                let cstart = self.starts[i];
                let cend = if i == self.starts.len() - 1 {
                    self.connectivity[2][0].borrow().num_rows()
                } else {
                    self.starts[i + 1]
                };
                for c in cstart..cend {
                    for (e, t) in izip!(unsafe { cell_to_entities0.row_unchecked(c) }, &etypes) {
                        cell_types[*e] = *t;
                    }
                }
            }
            for (ei, entity0) in entities0.iter_rows().enumerate() {
                let entity = get_reference_cell(cell_types[ei]);
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
                self.connectivity[dim0][dim1].borrow_mut().add_row(&row);
            }
        }
    }

    fn connectivity(&self, dim0: usize, dim1: usize) -> Ref<Self::Connectivity> {
        self.create_connectivity(dim0, dim1);
        self.connectivity[dim0][dim1].borrow()
    }

    fn entity_ownership(&self, _dim: usize, _index: usize) -> Ownership {
        Ownership::Owned
    }

    fn adjacent_cells(&self, cell: usize) -> Ref<Vec<(usize, usize)>> {
        if self.adjacent_cells[0].borrow().len() == 0 {
            self.compute_adjacent_cells()
        }
        self.adjacent_cells[cell].borrow()
    }
}

/// Serial grid
pub struct SerialGrid {
    topology: SerialTopology,
    geometry: SerialGeometry,
}

impl SerialGrid {
    pub fn new(
        coordinates: Array2D<f64>,
        cells: AdjacencyList<usize>,
        cell_types: Vec<ReferenceCellType>,
    ) -> Self {
        Self {
            topology: SerialTopology::new(&cells, &cell_types),
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

#[cfg(test)]
mod test {
    use crate::grid::*;
    use approx::*;

    #[test]
    fn test_adjacent_cells() {
        let g = SerialGrid::new(
            Array2D::from_data(
                vec![
                    0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0,
                    2.0, 2.0,
                ],
                (9, 2),
            ),
            AdjacencyList::from_data(
                vec![
                    0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4, 3, 4, 7, 3, 7, 6, 4, 5, 8, 4, 8, 7,
                ],
                vec![0, 3, 6, 9, 12, 15, 18, 21, 24],
            ),
            vec![ReferenceCellType::Triangle; 8],
        );
        assert_eq!(g.topology().adjacent_cells(0).len(), 7);
        for i in g.topology().adjacent_cells(0).iter() {
            if i.0 == 0 {
                assert_eq!(i.1, 3);
            } else if i.0 == 1 || i.0 == 3 {
                assert_eq!(i.1, 2);
            } else if i.0 == 2 || i.0 == 4 || i.0 == 6 || i.0 == 7 {
                assert_eq!(i.1, 1);
            } else {
                panic!("Cell is not adjacent.");
            }
        }
        assert_eq!(g.topology().adjacent_cells(2).len(), 4);
        for i in g.topology().adjacent_cells(2).iter() {
            if i.0 == 2 {
                assert_eq!(i.1, 3);
            } else if i.0 == 3 {
                assert_eq!(i.1, 2);
            } else if i.0 == 0 || i.0 == 6 {
                assert_eq!(i.1, 1);
            } else {
                panic!("Cell is not adjacent.");
            }
        }
    }
    #[test]
    fn test_connectivity() {
        let g = SerialGrid::new(
            Array2D::from_data(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], (4, 2)),
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
            Array2D::from_data(
                vec![
                    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
                    0.0, 0.0, -1.0,
                ],
                (6, 3),
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
            Array2D::from_data(
                vec![
                    0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0,
                    1.0, 1.0,
                ],
                (9, 2),
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
            Array2D::from_data(
                vec![
                    0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0,
                    1.0, 1.0,
                ],
                (9, 2),
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
            Array2D::from_data(
                vec![
                    0.0, 0.0, 1.0, 0.0, s, s, 0.0, 1.0, -s, s, -1.0, 0.0, -s, -s, 0.0, -1.0, s, -s,
                ],
                (9, 2),
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
            Array2D::from_data(
                vec![
                    0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.25, 0.0, 0.0, 0.5, 0.5,
                    0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 2.0, 0.5, 0.0, 1.5, 0.75, 0.0, 0.0, 1.0, 0.0,
                    0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, -0.5, 0.0,
                ],
                (13, 3),
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
            Array2D::from_data(
                vec![
                    2.0, 2.0, 0.0, 4.0, 2.0, 0.0, 5.0, 3.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0,
                    0.0, -1.0, 1.0,
                ],
                (6, 3),
            ),
            AdjacencyList::from_data(vec![0, 1, 2, 3, 4, 5], vec![0, 3, 6]),
            vec![ReferenceCellType::Triangle, ReferenceCellType::Triangle],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 3);

        let points = Array2D::from_data(
            vec![0.2, 0.0, 0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0, 0.15, 0.3],
            (4, 2),
        );

        // Test compute_points
        let mut physical_points = Array2D::new((points.shape().0, 3));
        g.geometry()
            .compute_points(&points, 0, &mut physical_points);
        assert_relative_eq!(*physical_points.get(0, 0).unwrap(), 2.4);
        assert_relative_eq!(*physical_points.get(0, 1).unwrap(), 2.0);
        assert_relative_eq!(*physical_points.get(0, 2).unwrap(), 0.0);
        assert_relative_eq!(*physical_points.get(1, 0).unwrap(), 4.5);
        assert_relative_eq!(*physical_points.get(1, 1).unwrap(), 2.5);
        assert_relative_eq!(*physical_points.get(1, 2).unwrap(), 0.5);
        assert_relative_eq!(*physical_points.get(2, 0).unwrap(), 11.0 / 3.0);
        assert_relative_eq!(*physical_points.get(2, 1).unwrap(), 7.0 / 3.0);
        assert_relative_eq!(*physical_points.get(2, 2).unwrap(), 1.0 / 3.0);
        assert_relative_eq!(*physical_points.get(3, 0).unwrap(), 3.2);
        assert_relative_eq!(*physical_points.get(3, 1).unwrap(), 2.3);
        assert_relative_eq!(*physical_points.get(3, 2).unwrap(), 0.3);
        g.geometry()
            .compute_points(&points, 1, &mut physical_points);
        assert_relative_eq!(*physical_points.get(0, 0).unwrap(), -0.2);
        assert_relative_eq!(*physical_points.get(0, 1).unwrap(), 0.0);
        assert_relative_eq!(*physical_points.get(0, 2).unwrap(), 1.0);
        assert_relative_eq!(*physical_points.get(1, 0).unwrap(), -0.5);
        assert_relative_eq!(*physical_points.get(1, 1).unwrap(), -0.5);
        assert_relative_eq!(*physical_points.get(1, 2).unwrap(), 1.0);
        assert_relative_eq!(*physical_points.get(2, 0).unwrap(), -1.0 / 3.0);
        assert_relative_eq!(*physical_points.get(2, 1).unwrap(), -1.0 / 3.0);
        assert_relative_eq!(*physical_points.get(2, 2).unwrap(), 1.0);
        assert_relative_eq!(*physical_points.get(3, 0).unwrap(), -0.15);
        assert_relative_eq!(*physical_points.get(3, 1).unwrap(), -0.3);
        assert_relative_eq!(*physical_points.get(3, 2).unwrap(), 1.0);

        // Test compute_jacobians
        let mut jacobians = Array2D::new((points.shape().0, 6));
        g.geometry().compute_jacobians(&points, 0, &mut jacobians);
        for i in 0..3 {
            assert_relative_eq!(*jacobians.get(i, 0).unwrap(), 2.0);
            assert_relative_eq!(*jacobians.get(i, 1).unwrap(), 3.0);
            assert_relative_eq!(*jacobians.get(i, 2).unwrap(), 0.0);
            assert_relative_eq!(*jacobians.get(i, 3).unwrap(), 1.0);
            assert_relative_eq!(*jacobians.get(i, 4).unwrap(), 0.0);
            assert_relative_eq!(*jacobians.get(i, 5).unwrap(), 1.0);
        }
        g.geometry().compute_jacobians(&points, 1, &mut jacobians);
        for i in 0..3 {
            assert_relative_eq!(*jacobians.get(i, 0).unwrap(), -1.0);
            assert_relative_eq!(*jacobians.get(i, 1).unwrap(), 0.0);
            assert_relative_eq!(*jacobians.get(i, 2).unwrap(), 0.0);
            assert_relative_eq!(*jacobians.get(i, 3).unwrap(), -1.0);
            assert_relative_eq!(*jacobians.get(i, 4).unwrap(), 0.0);
            assert_relative_eq!(*jacobians.get(i, 5).unwrap(), 0.0);
        }

        // test compute_jacobian_determinants
        let mut dets = vec![0.0; points.shape().0];
        g.geometry()
            .compute_jacobian_determinants(&points, 0, &mut dets);
        for d in &dets {
            assert_relative_eq!(*d, 2.0 * 2.0_f64.sqrt());
        }
        g.geometry()
            .compute_jacobian_determinants(&points, 1, &mut dets);
        for d in &dets {
            assert_relative_eq!(*d, 1.0);
        }

        // Test compute_jacobian_inverses
        let mut jinvs = Array2D::new((points.shape().0, 6));
        g.geometry()
            .compute_jacobian_inverses(&points, 0, &mut jinvs);
        for i in 0..3 {
            assert_relative_eq!(*jinvs.get(i, 0).unwrap(), 0.5);
            assert_relative_eq!(*jinvs.get(i, 1).unwrap(), -0.75);
            assert_relative_eq!(*jinvs.get(i, 2).unwrap(), -0.75);
            assert_relative_eq!(*jinvs.get(i, 3).unwrap(), 0.0);
            assert_relative_eq!(*jinvs.get(i, 4).unwrap(), 0.5);
            assert_relative_eq!(*jinvs.get(i, 5).unwrap(), 0.5);
        }
        g.geometry()
            .compute_jacobian_inverses(&points, 1, &mut jinvs);
        for i in 0..3 {
            assert_relative_eq!(*jinvs.get(i, 0).unwrap(), -1.0);
            assert_relative_eq!(*jinvs.get(i, 1).unwrap(), 0.0);
            assert_relative_eq!(*jinvs.get(i, 2).unwrap(), 0.0);
            assert_relative_eq!(*jinvs.get(i, 3).unwrap(), 0.0);
            assert_relative_eq!(*jinvs.get(i, 4).unwrap(), -1.0);
            assert_relative_eq!(*jinvs.get(i, 5).unwrap(), 0.0);
        }
    }

    #[test]
    fn test_normals() {
        let g = SerialGrid::new(
            Array2D::from_data(
                vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0,
                ],
                (5, 3),
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

        let pt = Array2D::from_data(vec![1.0 / 3.0, 1.0 / 3.0], (1, 3));

        let mut normal = Array2D::<f64>::new((1, 3));

        g.geometry().compute_normals(&pt, 0, &mut normal);
        assert_relative_eq!(*normal.get(0, 0).unwrap(), 0.0);
        assert_relative_eq!(*normal.get(0, 1).unwrap(), -1.0);
        assert_relative_eq!(*normal.get(0, 2).unwrap(), 0.0);

        g.geometry().compute_normals(&pt, 1, &mut normal);
        let a = f64::sqrt(1.0 / 3.0);
        assert_relative_eq!(*normal.get(0, 0).unwrap(), a);
        assert_relative_eq!(*normal.get(0, 1).unwrap(), a);
        assert_relative_eq!(*normal.get(0, 2).unwrap(), a);

        g.geometry().compute_normals(&pt, 2, &mut normal);
        assert_relative_eq!(*normal.get(0, 0).unwrap(), 0.0);
        assert_relative_eq!(*normal.get(0, 1).unwrap(), 0.0);
        assert_relative_eq!(*normal.get(0, 2).unwrap(), 1.0);

        // Test a curved quadrilateral cell
        let curved_g = SerialGrid::new(
            Array2D::from_data(
                vec![
                    -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0,
                    -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                ],
                (9, 3),
            ),
            AdjacencyList::from_data(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![0, 9]),
            vec![ReferenceCellType::Quadrilateral],
        );

        let points = Array2D::from_data(
            vec![0.0, 0.0, 0.2, 0.3, 0.5, 0.9, 0.7, 1.0, 1.0, 0.3],
            (5, 2),
        );
        let mut normals = Array2D::<f64>::new((5, 3));

        curved_g
            .geometry()
            .compute_normals(&points, 0, &mut normals);

        assert_relative_eq!(
            *normals.get(0, 0).unwrap(),
            2.0 * f64::sqrt(1.0 / 5.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(*normals.get(0, 1).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            *normals.get(0, 2).unwrap(),
            f64::sqrt(1.0 / 5.0),
            epsilon = 1e-12
        );

        assert_relative_eq!(
            *normals.get(1, 0).unwrap(),
            1.2 * f64::sqrt(1.0 / 2.44),
            epsilon = 1e-12
        );
        assert_relative_eq!(*normals.get(1, 1).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            *normals.get(1, 2).unwrap(),
            f64::sqrt(1.0 / 2.44),
            epsilon = 1e-12
        );

        assert_relative_eq!(*normals.get(2, 0).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(*normals.get(2, 1).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(*normals.get(2, 2).unwrap(), 1.0, epsilon = 1e-12);

        assert_relative_eq!(
            *normals.get(3, 0).unwrap(),
            -0.8 * f64::sqrt(1.0 / 1.64),
            epsilon = 1e-12
        );
        assert_relative_eq!(*normals.get(3, 1).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            *normals.get(3, 2).unwrap(),
            f64::sqrt(1.0 / 1.64),
            epsilon = 1e-12
        );

        assert_relative_eq!(
            *normals.get(4, 0).unwrap(),
            -2.0 * f64::sqrt(1.0 / 5.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(*normals.get(4, 1).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            *normals.get(4, 2).unwrap(),
            f64::sqrt(1.0 / 5.0),
            epsilon = 1e-12
        );
    }
}
