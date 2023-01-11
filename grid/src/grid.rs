pub use solvers_element::cell::Triangle;
pub use solvers_traits::cell::ReferenceCell;
pub use solvers_traits::grid::Geometry;
pub use solvers_traits::grid::Grid;
pub use solvers_traits::grid::Locality;
pub use solvers_traits::grid::Topology;
use std::cmp::max;
use std::cmp::min;

pub struct SerialTriangle3DGeometry<'a> {
    pub coordinates: &'a [f64],
    pub vertex_numbers: &'a [usize],
}

impl Geometry for SerialTriangle3DGeometry<'_> {
    fn reference_cell(&self) -> &dyn ReferenceCell {
        &Triangle {}
    }

    fn map(&self, reference_coords: &[f64], physical_coords: &mut [f64]) {
        for i in 0..3 {
            physical_coords[i] = self.coordinates[self.vertex_numbers[0] * 3 + i]
                + reference_coords[0]
                    * (self.coordinates[self.vertex_numbers[1] * 3 + i]
                        - self.coordinates[self.vertex_numbers[0] * 3 + i])
                + reference_coords[2]
                    * (self.coordinates[self.vertex_numbers[2] * 3 + i]
                        - self.coordinates[self.vertex_numbers[0] * 3 + i]);
        }
    }

    fn dim(&self) -> usize {
        3
    }

    fn midpoint(&self) -> Vec<f64> {
        let mut mid = vec![0.0, 0.0, 0.0];
        for i in 0..3 {
            for j in 0..3 {
                mid[i] += self.coordinates[self.vertex_numbers[j] * 3 + i];
            }
            mid[i] /= 3.0;
        }
        mid
    }
    fn volume(&self) -> f64 {
        let a = [
            self.coordinates[self.vertex_numbers[1] * 3]
                - self.coordinates[self.vertex_numbers[0] * 3],
            self.coordinates[self.vertex_numbers[1] * 3 + 1]
                - self.coordinates[self.vertex_numbers[0] * 3 + 1],
            self.coordinates[self.vertex_numbers[1] * 3 + 2]
                - self.coordinates[self.vertex_numbers[0] * 3 + 2],
        ];
        let b = [
            self.coordinates[self.vertex_numbers[2] * 3]
                - self.coordinates[self.vertex_numbers[0] * 3],
            self.coordinates[self.vertex_numbers[2] * 3 + 1]
                - self.coordinates[self.vertex_numbers[0] * 3 + 1],
            self.coordinates[self.vertex_numbers[2] * 3 + 2]
                - self.coordinates[self.vertex_numbers[0] * 3 + 2],
        ];
        ((a[0] * b[1] - a[1] * b[0]).powi(2)
            + (a[1] * b[2] - a[2] * b[1]).powi(2)
            + (a[2] * b[0] - a[0] * b[2]).powi(2))
        .sqrt()
    }
}

pub struct SerialTriangle3DTopology<'a> {
    pub cells: &'a [usize],
    pub connectivity_2_1: Vec<usize>,
    // pub connectivity_1_2: Vec<usize>,
    pub connectivity_1_0: Vec<usize>,
    // pub connectivity_0_2: Vec<usize>,
    // pub connectivity_0_1: Vec<usize>,
    // TODO: adjacency lists
    // TODO: only create this data later when needed
}

impl Topology for SerialTriangle3DTopology<'_> {
    fn local2global(&self, local_id: usize) -> usize {
        local_id
    }
    fn global2local(&self, global_id: usize) -> Option<usize> {
        Some(global_id)
    }
    fn locality(&self, global_id: usize) -> Locality {
        Locality::Local
    }
    fn dim(&self) -> usize {
        2
    }
    fn entity_count(&self, dim: usize) -> usize {
        match dim {
            0 => {
                let mut nvertices = 0;
                for v in self.cells {
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
}

pub struct SerialTriangle3DGrid {
    pub coordinates: Vec<f64>,
    pub cells: Vec<usize>,
}

impl<'a> Grid for SerialTriangle3DGrid {
    type Topology<'b> = SerialTriangle3DTopology<'b> where Self: 'b;
    type Geometry<'b> = SerialTriangle3DGeometry<'b> where Self: 'b;

    fn topology<'b>(&'b self) -> Self::Topology<'b> {
        let mut edges = vec![];
        let mut triangles_to_edges = vec![];
        for triangle in 0..&self.cells.len() / 3 {
            for edge in [(1, 2), (0, 2), (0, 1)] {
                let start = min(
                    self.cells[3 * triangle + edge.0],
                    self.cells[3 * triangle + edge.1],
                );
                let end = max(
                    self.cells[3 * triangle + edge.0],
                    self.cells[3 * triangle + edge.1],
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
        SerialTriangle3DTopology {
            cells: &self.cells,
            connectivity_2_1: triangles_to_edges,
            connectivity_1_0: edges,
        }
    }

    fn cell_geometry<'b>(&'b self, local_index: usize) -> Self::Geometry<'b> {
        SerialTriangle3DGeometry::<'b> {
            coordinates: &self.coordinates,
            vertex_numbers: &self.cells[3 * local_index..3 * (local_index + 1)],
        }
    }
}

#[cfg(test)]
mod test {
    use crate::grid::*;
    use approx::*;

    #[test]
    fn test_serial_triangle_grid() {
        let g = SerialTriangle3DGrid {
            coordinates: vec![
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
                0.0, -1.0,
            ],
            cells: vec![
                0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 1,
            ],
        };
        assert_eq!(g.topology().dim(), 2);
        assert_eq!(g.cell_geometry(0).dim(), 3);

        let m = g.cell_geometry(0).midpoint();
        for i in 0..3 {
            assert_relative_eq!(m[i], 1.0 / 3.0);
        }
    }
}
