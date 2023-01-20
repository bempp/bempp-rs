pub use solvers_element::cell::Triangle;
pub use solvers_tools::arrays::AdjacencyList;
pub use solvers_tools::arrays::Array2D;
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
    cells: Vec<usize>,
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
                for v in &self.cells {
                    if *v >= nvertices {
                        nvertices = *v + 1;
                    }
                }
                nvertices
            }
            1 => self.connectivity_1_0.len() / 2,
            2 => self.cells.len() / 3,
            _ => 0,
        }
    }
    fn cell(&self, index: usize) -> &[usize] {
        &   self.cells[3*index..3*(index + 1)]
    }
}

pub struct SerialTriangle3DGrid {
    geometry: SerialGeometry,
    topology: SerialTriangle3DTopology,
}

impl SerialTriangle3DGrid {
    pub fn new(coordinates: Array2D<f64>, cells: Vec<usize>) -> Self {
        let mut edges = vec![];
        let mut triangles_to_edges = vec![];
        for triangle in 0..&cells.len() / 3 {
            for edge in [(1, 2), (0, 2), (0, 1)] {
                let start = min(cells[3 * triangle + edge.0], cells[3 * triangle + edge.1]);
                let end = max(cells[3 * triangle + edge.0], cells[3 * triangle + edge.1]);
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
            vec![
                0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 1,
            ],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 3);
    }
    fn test_serial_triangle_grid_screen() {
        let g = SerialTriangle3DGrid::new(
            Array2D::from_data(
                vec![
                    0.0, 0.0,
                    0.5, 0.0,
                    1.0, 0.0,
                    0.0, 0.5,
                    0.5, 0.5,
                    1.0, 0.5,
                    0.0, 1.0,
                    0.5, 1.0,
                    1.0, 1.0,
                ],
                (9, 2),
            ),
            vec![
                0, 1, 4, 1, 2, 5, 0, 4, 3, 1, 5, 4, 3, 4, 7, 4, 5, 8, 3, 7, 6, 4, 8, 7,
            ],
        );
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.geometry().dim(), 2);
    }
}
