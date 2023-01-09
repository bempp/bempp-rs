pub use solvers_element::cell::Triangle;
pub use solvers_traits::cell::ReferenceCell;
pub use solvers_traits::grid::Geometry;
pub use solvers_traits::grid::Grid;
pub use solvers_traits::grid::Locality;
pub use solvers_traits::grid::Topology;

pub struct SerialTriangle3DGeometry<'a> {
    pub coordinates: &'a [f64],
    pub vertices: &'a [usize],
}

impl Geometry for SerialTriangle3DGeometry<'_> {
    fn reference_cell(&self) -> &dyn ReferenceCell {
        &Triangle {}
    }

    fn map(&self, reference_coords: &[f64], physical_coords: &mut [f64]) {
        for i in 0..3 {
            physical_coords[i] = self.coordinates[self.vertices[0] * 3 + i]
                + reference_coords[0]
                    * (self.coordinates[self.vertices[1] * 3 + i]
                        - self.coordinates[self.vertices[0] * 3 + i])
                + reference_coords[2]
                    * (self.coordinates[self.vertices[2] * 3 + i]
                        - self.coordinates[self.vertices[0] * 3 + i]);
        }
    }

    fn dim(&self) -> usize {
        3
    }

    fn midpoint(&self) -> Vec<f64> {
        let mut mid = vec![0.0, 0.0, 0.0];
        for i in 0..3 {
            for j in 0..3 {
                mid[i] += self.coordinates[self.vertices[j] * 3 + i];
            }
            mid[i] /= 3.0;
        }
        mid
    }
}

pub struct SerialTriangle3DTopology {}

impl Topology for SerialTriangle3DTopology {
    fn local2global(local_id: usize) -> usize {
        local_id
    }
    fn global2local(global_id: usize) -> Option<usize> {
        Some(global_id)
    }
    fn locality(global_id: usize) -> Locality {
        Locality::Local
    }
    fn dim(&self) -> usize {
        2
    }
}

pub struct SerialTriangle3DGrid {
    pub coordinates: Vec<f64>,
    pub cells: Vec<usize>,
}

impl<'a> Grid for SerialTriangle3DGrid {
    type Topology<'b> = SerialTriangle3DTopology where Self: 'b;
    type Geometry<'b> = SerialTriangle3DGeometry<'b> where Self: 'b;

    fn topology(&self) -> Self::Topology<'a> {
        SerialTriangle3DTopology {}
    }

    fn cell_geometry<'b>(&'b self, local_index: usize) -> Self::Geometry<'b> {
        SerialTriangle3DGeometry::<'b> {
            coordinates: &self.coordinates,
            vertices: &self.cells[3 * local_index..3 * (local_index + 1)],
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
