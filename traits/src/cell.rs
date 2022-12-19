use crate::element::FiniteElement;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ReferenceCellType {
    Interval = 0,
    Triangle = 1,
    Quadrilateral = 2,
    Tetrahedron = 3,
    Hexahedron = 4,
    Prism = 5,
    Pyramid = 6,
}

/// A 0- to 3- dimensional reference cell
pub trait ReferenceCell {
    /// The dimension of the reference cell (eg a triangle's dimension is 2, tetrahedron's dimension is 3)
    fn dim(&self) -> usize;

    /// The vertices of the cell
    ///
    /// The first dim components represent the first vertex, the next dim the second vertex, and so on.
    fn vertices(&self) -> &[f64];

    /// The edges of the cell
    ///
    /// The first 2 components are the vertex numbers of the endpoints of the first edge, the next 2 the second edge, and so on.
    fn edges(&self) -> &[usize];

    /// The faces of the cell
    ///
    /// The first `self.faces_nvertices()[0]` components are the vertex numbers of vertices of the first face, the next `self.faces_nvertices()[1]` the second edge, and so on.
    fn faces(&self) -> &[usize];

    /// The number of vertices adjacent to each face
    fn faces_nvertices(&self) -> &[usize];

    /// The number of entities of dimension `dim`
    fn entity_count(&self, dim: usize) -> Result<usize, ()> {
        match dim {
            0 => Ok(self.vertex_count()),
            1 => Ok(self.edge_count()),
            2 => Ok(self.face_count()),
            3 => Ok(self.volume_count()),
            _ => Err(()),
        }
    }

    /// The number of vertices
    fn vertex_count(&self) -> usize;

    /// The number of edges
    fn edge_count(&self) -> usize;

    /// The number of faces
    fn face_count(&self) -> usize;

    /// The number of volumes
    fn volume_count(&self) -> usize;

    /// Get the entities connected to an entity
    ///
    /// This function returns a list of entity numbers of entities of dimension `connected_dim` that are attached to the entity numbered `entity_dim` of number entity_number.
    /// For example connectivity(1, 0, 2) will return a list of faces (2D entities) that are connected to edge (1D entity) 0.
    fn connectivity(
        &self,
        entity_dim: usize,
        entity_number: usize,
        connected_dim: usize,
    ) -> Result<Vec<usize>, ()>;

    /// The reference cell type
    fn cell_type(&self) -> ReferenceCellType;

    /// The reference cell label
    fn label(&self) -> &'static str;
}

pub trait PhysicalCell<'a, F: FiniteElement> {
    fn tdim(&self) -> usize;
    fn gdim(&self) -> usize;
    fn coordinate_element(&self) -> &'a F;
    fn npts(&self) -> usize;
    fn vertex(&self, vertex_number: usize) -> &'a [f64];
}
