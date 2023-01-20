pub use solvers_element::cell::Triangle;
use solvers_tools::arrays::AdjacencyList;
use solvers_tools::arrays::Array2D;
pub use solvers_traits::cell::ReferenceCell;
pub use solvers_traits::grid::Geometry;
pub use solvers_traits::grid::Grid;
pub use solvers_traits::grid::Locality;
pub use solvers_traits::grid::Topology;
use std::cmp::max;
use std::cmp::min;

pub struct SerialGeometry {
    pub coordinates: Array2D<f64>,
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

pub struct SerialTriangle3DTopology {
    cells: AdjacencyList<usize>,
    pub connectivity_2_1: Vec<usize>,
    // pub connectivity_1_2: Vec<usize>,
    pub connectivity_1_0: Vec<usize>,
    // pub connectivity_0_2: Vec<usize>,
    // pub connectivity_0_1: Vec<usize>,
    // TODO: adjacency lists
    // TODO: only create this data later when needed
}

impl Topology for SerialTriangle3DTopology {
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
        match dim {
            0 => {
                let mut nvertices = 0;
                for i in 0..self.cells.num_rows() {
                    let row = self.cells.row(i).unwrap();
                    for v in row {
                        if *v >= nvertices {
                            nvertices = *v + 1;
                        }
                    }
                }
                nvertices
            }
            1 => self.connectivity_1_0.len() / 2,
            2 => self.cells.num_rows(),
            _ => 0,
        }
    }
    fn cell(&self, index: usize) -> Option<&[usize]> {
        self.cells.row(index)
    }
    unsafe fn cell_unchecked(&self, index: usize) -> &[usize] {
        self.cells.row_unchecked(index)
    }
}

pub struct SerialTriangle3DGrid {
    geometry: SerialGeometry,
    topology: SerialTriangle3DTopology,
}

impl SerialTriangle3DGrid {
    pub fn new(coordinates: Array2D<f64>, cells: AdjacencyList<usize>) -> Self {
        let mut edges = vec![];
        let mut triangles_to_edges = vec![];
        for triangle in 0..cells.num_rows() {
            for edge in [(1, 2), (0, 2), (0, 1)] {
                let start = min(
                    *cells.get(triangle, edge.0).unwrap(),
                    *cells.get(triangle, edge.1).unwrap(),
                );
                let end = max(
                    *cells.get(triangle, edge.0).unwrap(),
                    *cells.get(triangle, edge.1).unwrap(),
                );
                let mut found = false;
                for i in 0..edges.len() / 2 {
                    if edges[2 * i] == start && edges[2 * i + 1] == end {
                        found = true;
                        triangles_to_edges.push(i);
                        break;
                    }
                }
                if !found {
                    triangles_to_edges.push(edges.len() / 2);
                    edges.push(start);
                    edges.push(end);
                }
            }
        }

        Self {
            geometry: SerialGeometry {
                coordinates: coordinates,
            },
            topology: SerialTriangle3DTopology {
                cells: cells,
                connectivity_2_1: triangles_to_edges,
                connectivity_1_0: edges,
            },
        }
    }
}
impl Grid for SerialTriangle3DGrid {
    type Topology = SerialTriangle3DTopology;
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
    fn test_serial_triangle_grid_octahedron() {
        let g = SerialTriangle3DGrid::new(
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
    }

    #[test]
    fn test_serial_triangle_grid_screen() {
        let g = SerialTriangle3DGrid::new(
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
    }

    #[test]
    fn test_serial_mixed_grid_screen() {
        let g = SerialTriangle3DGrid::new(
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
    }
}
