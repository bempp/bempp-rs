pub use solvers_element::cell::Triangle;
use solvers_tools::arrays::AdjacencyList;
use solvers_tools::arrays::Array2D;
pub use solvers_traits::grid::Geometry;
pub use solvers_traits::grid::Grid;
pub use solvers_traits::grid::Locality;
pub use solvers_traits::grid::Topology;
use std::cmp::max;
use std::cmp::min;

pub struct SerialGeometry {
    pub coordinates: Array2D<f64>,
}

impl SerialGeometry {
    pub fn new(coordinates: Array2D<f64>) -> Self {
        Self {
            coordinates: coordinates,
        }
    }
}

impl Geometry for SerialGeometry {
    fn dim(&self) -> usize {
        self.coordinates.shape().1
    }

    fn point(&self, i: usize) -> Option<&[f64]> {
        self.coordinates.row(i)
    }
    unsafe fn point_unchecked(&self, i: usize) -> &[f64] {
        self.coordinates.row_unchecked(i)
    }

    fn point_count(&self) -> usize {
        self.coordinates.shape().0
    }
}

pub struct Serial2DTopology {
    connectivity: [[AdjacencyList<usize>; 3]; 3],
}

impl Serial2DTopology {
    pub fn new(cells: AdjacencyList<usize>) -> Self {
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
                    cells,
                    AdjacencyList::<usize>::new(),
                    AdjacencyList::<usize>::new(),
                ],
            ],
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
            self.connectivity[0][0].add_row(vec![i]);
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
            self.connectivity[0][1].add_row(row);
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
            self.connectivity[0][2].add_row(row);
        }
    }
    fn create_connectivity_10(&mut self) {
        let mut data = AdjacencyList::<usize>::new();
        let cells = &self.connectivity[2][0];
        for cell in cells.iter_rows() {
            let cell_edges = match cell.len() {
                // TODO: remove hard coding here
                3 => vec![(1, 2), (0, 2), (0, 1)],
                4 => vec![(0, 1), (0, 2), (1, 3), (2, 3)],
                _ => {
                    panic!("Unsupported cell type.")
                }
            };
            for e in cell_edges {
                let start = min(cell[e.0], cell[e.1]);
                let end = max(cell[e.0], cell[e.1]);
                let mut found = false;
                for edge in data.iter_rows() {
                    if edge[0] == start && edge[1] == end {
                        found = true;
                        break;
                    }
                }
                if !found {
                    data.add_row(vec![start, end]);
                }
            }
        }
        self.connectivity[1][0] = data;
    }
    fn create_connectivity_11(&mut self) {
        self.create_connectivity(1, 0);
        for i in 0..self.connectivity[0][1].num_rows() {
            self.connectivity[1][1].add_row(vec![i]);
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
            self.connectivity[1][2].add_row(row);
        }
    }
    fn create_connectivity_21(&mut self) {
        self.create_connectivity(1, 0);
        let mut data = AdjacencyList::<usize>::new();
        let cells = &self.connectivity[2][0];
        let edges = &self.connectivity[1][0];
        for cell in cells.iter_rows() {
            let mut row = vec![];
            let cell_edges = match cell.len() {
                // TODO: remove hard coding here
                3 => vec![(1, 2), (0, 2), (0, 1)],
                4 => vec![(0, 1), (0, 2), (1, 3), (2, 3)],
                _ => {
                    panic!("Unsupported cell type.")
                }
            };
            for e in cell_edges {
                let start = min(cell[e.0], cell[e.1]);
                let end = max(cell[e.0], cell[e.1]);
                for (i, edge) in edges.iter_rows().enumerate() {
                    if edge[0] == start && edge[1] == end {
                        row.push(i);
                        break;
                    }
                }
            }
            data.add_row(row);
        }
        self.connectivity[2][1] = data;
    }
    fn create_connectivity_22(&mut self) {
        for i in 0..self.connectivity[2][0].num_rows() {
            self.connectivity[2][2].add_row(vec![i]);
        }
    }
}

impl Topology for Serial2DTopology {
    fn local2global(&self, local_id: usize) -> usize {
        local_id
    }
    fn global2local(&self, global_id: usize) -> Option<usize> {
        Some(global_id)
    }
    fn locality(&self, _global_id: usize) -> Locality {
        Locality::Local
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
    unsafe fn cell_unchecked(&self, index: usize) -> &[usize] {
        self.connectivity[2][0].row_unchecked(index)
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

pub struct SerialGrid {
    geometry: SerialGeometry,
    topology: Serial2DTopology,
}

impl SerialGrid {
    pub fn new(coordinates: Array2D<f64>, cells: AdjacencyList<usize>) -> Self {
        Self {
            geometry: SerialGeometry::new(coordinates),
            topology: Serial2DTopology::new(cells),
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
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 2);
        g.topology_mut().create_connectivity_all();
        assert_eq!(g.topology().entity_count(0), 9);
        assert_eq!(g.topology().entity_count(1), 14);
        assert_eq!(g.topology().entity_count(2), 6);
    }
}
