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

pub struct SerialTriangle3DGrid<'a> {
    pub coordinates: &'a [f64],
    pub cells: &'a [usize],
}

impl<'a> Grid for SerialTriangle3DGrid<'a> {
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
