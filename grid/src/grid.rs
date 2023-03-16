//! A serial implementation of a grid
use itertools::izip;
use bempp_element::cell;
use bempp_element::element;
use bempp_tools::arrays::{AdjacencyList, Array2D};
use bempp_traits::cell::{ReferenceCell, ReferenceCellType};
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};
use std::cell::{Ref, RefCell};

/// Geometry of a serial grid
pub struct SerialGeometry {
    coordinate_elements: Vec<Box<dyn FiniteElement>>,
    coordinates: Array2D<f64>,
    cells: AdjacencyList<usize>,
    element_changes: Vec<usize>,
    index_map: Vec<usize>,
}

fn create_element_from_npts(cell_type: ReferenceCellType, npts: usize) -> Box<dyn FiniteElement> {
    match cell_type {
        ReferenceCellType::Triangle => {
            let degree = (((1 + 8 * npts) as f64).sqrt() as usize - 1) / 2 - 1;
            match degree {
                1 => Box::new(element::LagrangeElementTriangleDegree1 {}),
                2 => Box::new(element::LagrangeElementTriangleDegree2 {}),
                _ => {
                    panic!("Unsupported degree (for now)");
                }
            }
        }
        ReferenceCellType::Quadrilateral => {
            let degree = (npts as f64).sqrt() as usize - 1;
            match degree {
                1 => Box::new(element::LagrangeElementQuadrilateralDegree1 {}),
                2 => Box::new(element::LagrangeElementQuadrilateralDegree2 {}),
                _ => {
                    panic!("Unsupported degree (for now)");
                }
            }
        }
        _ => {
            panic!("Unsupported cell type (for now)");
        }
    }
}

impl SerialGeometry {
    pub fn new(
        coordinates: Array2D<f64>,
        cells: &AdjacencyList<usize>,
        cell_types: &Vec<ReferenceCellType>,
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
                coordinate_elements.push(create_element_from_npts(cell_type, npts));
                for (j, cell_j) in cells.iter_rows().enumerate() {
                    if cell_type == cell_types[j] && npts == cell_j.len() {
                        new_cells.add_row(cells.row(j).unwrap());
                        index_map.push(j);
                    }
                }
            }
        }

        Self {
            coordinate_elements: coordinate_elements,
            coordinates: coordinates,
            cells: new_cells,
            element_changes: element_changes,
            index_map: index_map,
        }
    }

    pub fn coordinate_elements(&self) -> &Vec<Box<dyn FiniteElement>> {
        &self.coordinate_elements
    }

    pub fn element_changes(&self) -> &Vec<usize> {
        &self.element_changes
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
}

/// Topology of a serial grid
pub struct SerialTopology {
    dim: usize,
    connectivity: Vec<Vec<RefCell<AdjacencyList<usize>>>>,
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

impl SerialTopology {
    pub fn new(cells: &AdjacencyList<usize>, cell_types: &Vec<ReferenceCellType>) -> Self {
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

        Self {
            dim: dim,
            connectivity: connectivity,
            index_map: index_map,
            starts: starts,
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

impl Topology for SerialTopology {
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
                let ref_entities = (0..ref_cell.entity_count(dim0).unwrap())
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
                let etypes = ref_cell.entity_types(dim0).unwrap();

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
                for i in 0..entity.entity_count(dim1).unwrap() {
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

    fn connectivity(&self, dim0: usize, dim1: usize) -> Ref<AdjacencyList<usize>> {
        self.create_connectivity(dim0, dim1);
        self.connectivity[dim0][dim1].borrow()
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
impl Grid for SerialGrid {
    type Topology = SerialTopology;

    type Geometry = SerialGeometry;

    fn topology(&self) -> &Self::Topology {
        &self.topology
    }

    fn geometry(&self) -> &Self::Geometry {
        &self.geometry
    }
}

#[cfg(test)]
mod test {
    use crate::grid::*;

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
        let s = 1.0 / (2.0 as f64).sqrt();
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
}
