//! A serial implementation of a grid
use solvers_element::cell;
use solvers_element::element::*;
use solvers_tools::arrays::AdjacencyList;
use solvers_tools::arrays::Array2D;
use solvers_traits::cell::{ReferenceCell, ReferenceCellType};
use solvers_traits::element::FiniteElement;
use solvers_traits::grid::{Geometry, Grid, Topology};
use std::cmp::max;
use std::cmp::min;
use std::ops::Range;

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
                1 => Box::new(LagrangeElementTriangleDegree1 {}),
                2 => Box::new(LagrangeElementTriangleDegree2 {}),
                _ => {
                    panic!("Unsupported degree (for now)");
                }
            }
        }
        ReferenceCellType::Quadrilateral => {
            let degree = (npts as f64).sqrt() as usize - 1;
            match degree {
                1 => Box::new(LagrangeElementQuadrilateralDegree1 {}),
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
        cells: AdjacencyList<usize>,
        cell_types: &Vec<ReferenceCellType>,
    ) -> Self {
        let mut index_map = vec![];
        let mut element_changes = vec![];
        let mut coordinate_elements = vec![];
        for (i, cell) in cells.iter_rows().enumerate() {
            if !index_map.contains(&i) {
                let cell_type = cell_types[i];
                let npts = cell.len();

                element_changes.push(index_map.len());
                coordinate_elements.push(create_element_from_npts(cell_type, npts));
                for (j, cell_j) in cells.iter_rows().enumerate() {
                    if cell_type == cell_types[j] && npts == cell_j.len() {
                        index_map.push(j);
                    }
                }
            }
        }

        Self {
            coordinate_elements: coordinate_elements,
            coordinates: coordinates,
            cells: cells,
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
    fn local2global(&self, local_id: usize) -> usize {
        self.index_map[local_id]
    }
    fn global2local(&self, global_id: usize) -> Option<usize> {
        for (i, j) in self.index_map.iter().enumerate() {
            if *j == global_id {
                return Some(i);
            }
        }
        None
    }
    fn cell_count(&self) -> usize {
        self.index_map.len()
    }
}

/// Topology of a serial grid
pub struct Serial2DTopology {
    connectivity: [[AdjacencyList<usize>; 3]; 3],
    index_map: Vec<usize>,
    quad_start: usize,
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

impl Serial2DTopology {
    pub fn new(cells: &AdjacencyList<usize>, cell_types: &Vec<ReferenceCellType>) -> Self {
        let mut c20 = AdjacencyList::<usize>::new();
        let mut index_map = vec![];
        for (i, cell) in cells.iter_rows().enumerate() {
            if cell_types[i] == ReferenceCellType::Triangle {
                index_map.push(i);
                c20.add_row(&cell[..3]);
            }
        }
        let quad_start = index_map.len();
        for (i, cell) in cells.iter_rows().enumerate() {
            if cell_types[i] == ReferenceCellType::Quadrilateral {
                index_map.push(i);
                c20.add_row(&cell[..4]);
            }
        }
        Self {
            connectivity: [
                [
                    AdjacencyList::<usize>::new(),
                    AdjacencyList::<usize>::new(),
                    AdjacencyList::<usize>::new(),
                ],
                [
                    AdjacencyList::<usize>::new(),
                    AdjacencyList::<usize>::new(),
                    AdjacencyList::<usize>::new(),
                ],
                [
                    c20,
                    AdjacencyList::<usize>::new(),
                    AdjacencyList::<usize>::new(),
                ],
            ],
            index_map: index_map,
            quad_start: quad_start,
        }
    }

    fn create_connectivity_00(&mut self) {
        let mut nvertices = 0;
        let cells = &self.connectivity[2][0];
        for cell in cells.iter_rows() {
            for j in cell {
                if *j >= nvertices {
                    nvertices = *j + 1;
                }
            }
        }
        for i in 0..nvertices {
            self.connectivity[0][0].add_row(&[i]);
        }
    }
    fn create_connectivity_01(&mut self) {
        self.create_connectivity(0, 0);
        self.create_connectivity(1, 0);
        let mut data = vec![vec![]; self.connectivity[0][0].num_rows()];
        for (i, row) in self.connectivity[1][0].iter_rows().enumerate() {
            for v in row {
                data[*v].push(i);
            }
        }
        for row in data {
            self.connectivity[0][1].add_row(&row);
        }
    }
    fn create_connectivity_02(&mut self) {
        self.create_connectivity(0, 0);
        let mut data = vec![vec![]; self.connectivity[0][0].num_rows()];
        for (i, row) in self.connectivity[2][0].iter_rows().enumerate() {
            for v in row {
                data[*v].push(i);
            }
        }
        for row in data {
            self.connectivity[0][2].add_row(&row);
        }
    }
    fn create_connectivity_10(&mut self) {
        let mut data = AdjacencyList::<usize>::new();
        let cells = &self.connectivity[2][0];
        for cell_type in [
            ReferenceCellType::Triangle,
            ReferenceCellType::Quadrilateral,
        ] {
            let ref_cell = get_reference_cell(cell_type);
            let ref_edges = (0..ref_cell.edge_count())
                .map(|x| ref_cell.connectivity(1, x, 0).unwrap())
                .collect::<Vec<Vec<usize>>>();
            for i in self.get_cells_range(cell_type).unwrap() {
                let cell = unsafe { cells.row_unchecked(i) };
                for e in &ref_edges {
                    let mut found = false;
                    let start = min(cell[e[0]], cell[e[1]]);
                    let end = max(cell[e[0]], cell[e[1]]);
                    for edge in data.iter_rows() {
                        if edge[0] == start && edge[1] == end {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        data.add_row(&[start, end]);
                    }
                }
            }
        }
        self.connectivity[1][0] = data;
    }
    fn create_connectivity_11(&mut self) {
        self.create_connectivity(1, 0);
        for i in 0..self.connectivity[0][1].num_rows() {
            self.connectivity[1][1].add_row(&[i]);
        }
    }
    fn create_connectivity_12(&mut self) {
        self.create_connectivity(1, 0);
        self.create_connectivity(2, 1);
        let mut data = vec![vec![]; self.connectivity[1][0].num_rows()];
        for (i, row) in self.connectivity[2][1].iter_rows().enumerate() {
            for v in row {
                data[*v].push(i);
            }
        }
        for row in data {
            self.connectivity[1][2].add_row(&row);
        }
    }
    fn create_connectivity_21(&mut self) {
        self.create_connectivity(1, 0);
        let mut data = AdjacencyList::<usize>::new();
        let cells = &self.connectivity[2][0];
        let edges = &self.connectivity[1][0];
        for cell_type in [
            ReferenceCellType::Triangle,
            ReferenceCellType::Quadrilateral,
        ] {
            let ref_cell = get_reference_cell(cell_type);
            let ref_edges = (0..ref_cell.edge_count())
                .map(|x| ref_cell.connectivity(1, x, 0).unwrap())
                .collect::<Vec<Vec<usize>>>();
            for i in self.get_cells_range(cell_type).unwrap() {
                let cell = unsafe { cells.row_unchecked(i) };
                let mut row = vec![];
                for e in &ref_edges {
                    let start = min(cell[e[0]], cell[e[1]]);
                    let end = max(cell[e[0]], cell[e[1]]);
                    for (i, edge) in edges.iter_rows().enumerate() {
                        if edge[0] == start && edge[1] == end {
                            row.push(i);
                            break;
                        }
                    }
                }
                data.add_row(&row);
            }
        }
        self.connectivity[2][1] = data;
    }
    fn create_connectivity_22(&mut self) {
        for i in 0..self.connectivity[2][0].num_rows() {
            self.connectivity[2][2].add_row(&[i]);
        }
    }
}

impl Topology for Serial2DTopology {
    fn index_map(&self) -> &[usize] {
        &self.index_map
    }
    fn get_cells(&self, cell_type: ReferenceCellType) -> Vec<usize> {
        match cell_type {
            ReferenceCellType::Triangle => {
                let mut cells = vec![];
                for (i, cell) in self.connectivity[2][0].iter_rows().enumerate() {
                    if cell.len() == 3 {
                        cells.push(i);
                    }
                }
                cells
            }
            ReferenceCellType::Quadrilateral => {
                let mut cells = vec![];
                for (i, cell) in self.connectivity[2][0].iter_rows().enumerate() {
                    if cell.len() == 4 {
                        cells.push(i);
                    }
                }
                cells
            }
            _ => vec![],
        }
    }
    fn get_cells_range(&self, cell_type: ReferenceCellType) -> Option<Range<usize>> {
        match cell_type {
            ReferenceCellType::Triangle => Some(0..self.quad_start),
            ReferenceCellType::Quadrilateral => Some(self.quad_start..self.index_map.len()),
            _ => Some(0..0),
        }
    }
    fn local2global(&self, local_id: usize) -> usize {
        self.index_map[local_id]
    }
    fn global2local(&self, global_id: usize) -> Option<usize> {
        for (i, j) in self.index_map.iter().enumerate() {
            if *j == global_id {
                return Some(i);
            }
        }
        None
    }
    fn dim(&self) -> usize {
        2
    }
    fn entity_count(&self, dim: usize) -> usize {
        for c in &self.connectivity[dim] {
            if c.num_rows() > 0 {
                return c.num_rows();
            }
        }
        panic!("Some connectivity including the relevant entities must be created first.");
    }
    fn cell(&self, index: usize) -> Option<&[usize]> {
        self.connectivity[2][0].row(index)
    }
    fn create_connectivity(&mut self, dim0: usize, dim1: usize) {
        if self.connectivity[dim0][dim1].num_rows() > 0 {
            return;
        }

        match dim0 {
            0 => match dim1 {
                0 => self.create_connectivity_00(),
                1 => self.create_connectivity_01(),
                2 => self.create_connectivity_02(),
                _ => {}
            },
            1 => match dim1 {
                0 => self.create_connectivity_10(),
                1 => self.create_connectivity_11(),
                2 => self.create_connectivity_12(),
                _ => {}
            },
            2 => match dim1 {
                1 => self.create_connectivity_21(),
                2 => self.create_connectivity_22(),
                _ => {}
            },
            _ => {}
        }
    }

    fn connectivity(&self, dim0: usize, dim1: usize) -> &AdjacencyList<usize> {
        if self.connectivity[dim0][dim1].num_rows() == 0 {
            panic!("Connectivity must be created first");
        }
        &self.connectivity[dim0][dim1]
    }
}

/// Serial grid
pub struct SerialGrid {
    topology: Serial2DTopology,
    geometry: SerialGeometry,
}

impl SerialGrid {
    pub fn new(
        coordinates: Array2D<f64>,
        cells: AdjacencyList<usize>,
        cell_types: Vec<ReferenceCellType>,
    ) -> Self {
        Self {
            topology: Serial2DTopology::new(&cells, &cell_types),
            geometry: SerialGeometry::new(coordinates, cells, &cell_types),
        }
    }
}
impl Grid for SerialGrid {
    type Topology = Serial2DTopology;
    type Geometry = SerialGeometry;

    fn topology(&self) -> &Self::Topology {
        &self.topology
    }

    fn topology_mut(&mut self) -> &mut Self::Topology {
        &mut self.topology
    }

    fn geometry(&self) -> &Self::Geometry {
        &self.geometry
    }
}

#[cfg(test)]
mod test {
    use crate::grid::*;

    #[test]
    fn test_serial_triangle_grid_octahedron() {
        let mut g = SerialGrid::new(
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
        g.topology_mut().create_connectivity_all();
        assert_eq!(g.topology().entity_count(0), 6);
        assert_eq!(g.topology().entity_count(1), 12);
        assert_eq!(g.topology().entity_count(2), 8);
    }

    #[test]
    fn test_serial_triangle_grid_screen() {
        let mut g = SerialGrid::new(
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
        g.topology_mut().create_connectivity_all();
        assert_eq!(g.topology().entity_count(0), 9);
        assert_eq!(g.topology().entity_count(1), 16);
        assert_eq!(g.topology().entity_count(2), 8);
    }

    #[test]
    fn test_serial_mixed_grid_screen() {
        let mut g = SerialGrid::new(
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
        g.topology_mut().create_connectivity_all();
        assert_eq!(g.topology().entity_count(0), 9);
        assert_eq!(g.topology().entity_count(1), 14);
        assert_eq!(g.topology().entity_count(2), 6);
    }
}
