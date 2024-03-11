use crate::element::FiniteElement;
use crate::types::ReferenceCellType;
use std::fmt;

#[derive(Debug)]
pub struct InvalidConnectivity;

impl fmt::Display for InvalidConnectivity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid connectivity")
    }
}

impl std::error::Error for InvalidConnectivity {}

/// A 0- to 3- dimensional reference cell
pub trait ReferenceCell {
    /// The dimension of the reference cell (eg a triangle's dimension is 2, tetrahedron's dimension is 3)
    fn dim(&self) -> usize;

    /// Check if the cell is a simplex
    fn is_simplex(&self) -> bool;

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
    /// The first `self.faces_nvertices()[0]` components are the vertex numbers of vertices of the first face, the next `self.faces_nvertices()[1]` the second face, and so on.
    fn faces(&self) -> &[usize];

    /// The number of vertices adjacent to each face
    fn faces_nvertices(&self) -> &[usize];

    /// The number of entities of dimension `dim`
    fn entity_count(&self, dim: usize) -> usize {
        match dim {
            0 => self.vertex_count(),
            1 => self.edge_count(),
            2 => self.face_count(),
            3 => self.volume_count(),
            _ => 0,
        }
    }

    /// The cell types of entities of dimension `dim`
    fn entity_types(&self, dim: usize) -> Vec<ReferenceCellType>;

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
    ) -> Result<Vec<usize>, InvalidConnectivity>;

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
