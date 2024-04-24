//! Traits used in the implementation of a grid

use crate::traits::element::FiniteElement;
use crate::traits::types::{CellLocalIndexPair, Ownership, ReferenceCellType};
use num::Float;
use rlst::RlstScalar;
use std::hash::Hash;

/// The topology of a grid.
///
/// This provides information about which mesh entities are connected to other mesh entities
pub trait Topology {
    /// Type used to indices topological entities
    type IndexType: std::fmt::Debug + Eq + Copy + Hash;

    /// The dimension of the topology (eg a triangle's dimension is 2, tetrahedron's dimension is 3)
    fn dim(&self) -> usize;

    /// Return the index map from the input cell numbers to the storage numbers
    fn index_map(&self) -> &[Self::IndexType];

    /// The number of entities of type `etype`
    fn entity_count(&self, etype: ReferenceCellType) -> usize;

    /// The number of entities of dimension `dim`
    fn entity_count_by_dim(&self, dim: usize) -> usize;

    /// The indices of the vertices of the cell with topological index `index`
    fn cell(&self, index: Self::IndexType) -> Option<&[Self::IndexType]>;

    /// The cell type of the cell with topological index `index`
    fn cell_type(&self, index: Self::IndexType) -> Option<ReferenceCellType>;

    /// All entity types of the given dimension that are included in the grid
    fn entity_types(&self, dim: usize) -> &[ReferenceCellType];

    /// Get the indices of entities of dimension `dim` that are connected to the cell with index `index`
    fn cell_to_entities(&self, index: Self::IndexType, dim: usize) -> Option<&[Self::IndexType]>;

    /// Get the flat indices of entities of dimension `dim` that are connected to the cell with index `index`
    fn cell_to_flat_entities(&self, index: Self::IndexType, dim: usize) -> Option<&[usize]>;

    /// Get the indices of cells that are connected to the entity with dimension `dim` and index `index`
    fn entity_to_cells(
        &self,
        dim: usize,
        index: Self::IndexType,
    ) -> Option<&[CellLocalIndexPair<Self::IndexType>]>;

    /// Get the flat indices of cells that are connected to the entity with dimension `dim` and index `index`
    fn entity_to_flat_cells(
        &self,
        dim: usize,
        index: Self::IndexType,
    ) -> Option<&[CellLocalIndexPair<usize>]>;

    /// Get the indices of the vertices that are connect to theentity with dimension `dim` and index `index`
    fn entity_vertices(&self, dim: usize, index: Self::IndexType) -> Option<&[Self::IndexType]>;

    /// Get the ownership of a cell
    fn cell_ownership(&self, index: Self::IndexType) -> Ownership;

    /// Get the ownership of a vertex
    fn vertex_ownership(&self, index: Self::IndexType) -> Ownership;

    /// Get the id of a vertex from its index
    fn vertex_index_to_id(&self, index: Self::IndexType) -> usize;

    /// Get the id of a cell from its index
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize;

    /// Get the index of a vertex from its id
    fn vertex_id_to_index(&self, id: usize) -> Self::IndexType;

    /// Get the index of a cell from its id
    fn cell_id_to_index(&self, id: usize) -> Self::IndexType;

    /// Get the flat index from the index of a vertex
    fn vertex_index_to_flat_index(&self, index: Self::IndexType) -> usize;

    /// Get the flat index from the index of an edge
    fn edge_index_to_flat_index(&self, index: Self::IndexType) -> usize;

    /// Get the flat index from the index of a face
    fn face_index_to_flat_index(&self, index: Self::IndexType) -> usize;

    /// Get the index from the flat index of a vertex
    fn vertex_flat_index_to_index(&self, index: usize) -> Self::IndexType;

    /// Get the index from the flat index of an edge
    fn edge_flat_index_to_index(&self, index: usize) -> Self::IndexType;

    /// Get the index from the flat index of a face
    fn face_flat_index_to_index(&self, index: usize) -> Self::IndexType;

    /// The cell types included in the grid topology
    fn cell_types(&self) -> &[ReferenceCellType];
}

/// The geometry of a grid
///
/// This provides information about the physical locations of mesh points in space
pub trait Geometry {
    /// Type used to index cells
    type IndexType: std::fmt::Debug + Eq + Copy;
    /// Scalar type
    type T: Float + RlstScalar<Real = Self::T>;
    /// Element type
    type Element: FiniteElement;
    /// Type of geometry evaluator
    type Evaluator<'a>: GeometryEvaluator<T = Self::T>
    where
        Self: 'a;

    /// The geometric dimension
    fn dim(&self) -> usize;

    /// Return the index map from the input cell numbers to the storage numbers
    fn index_map(&self) -> &[Self::IndexType];

    /// Get one of the coordinates of a point
    fn coordinate(&self, point_index: usize, coord_index: usize) -> Option<&Self::T>;

    /// The number of points stored in the geometry
    fn point_count(&self) -> usize;

    /// Get the indices of the points of a cell
    fn cell_points(&self, index: Self::IndexType) -> Option<&[usize]>;

    /// The number of cells
    fn cell_count(&self) -> usize;

    /// Get the element used to represent a cell
    fn cell_element(&self, index: Self::IndexType) -> Option<&Self::Element>;

    /// Get the number of distinct geometry elements
    fn element_count(&self) -> usize;
    /// Get the `i`th element
    fn element(&self, i: usize) -> Option<&Self::Element>;
    /// Get the cells associated with the `i`th element
    fn cell_indices(&self, i: usize) -> Option<&[Self::IndexType]>;

    /// Midpoint of a cell
    fn midpoint(&self, index: Self::IndexType, point: &mut [Self::T]);

    /// Diameter of a cell
    fn diameter(&self, index: Self::IndexType) -> Self::T;

    /// Volume of a cell
    fn volume(&self, index: Self::IndexType) -> Self::T;

    /// Get the geometry evaluator for the given points
    fn get_evaluator<'a>(&'a self, points: &'a [Self::T]) -> Self::Evaluator<'a>;

    /// Get the id of a point from its index
    fn point_index_to_id(&self, index: usize) -> usize;

    /// Get the id of a cell from its index
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize;

    /// Get the index of a point from its id
    fn point_id_to_index(&self, id: usize) -> usize;

    /// Get the index of a cell from its id
    fn cell_id_to_index(&self, id: usize) -> Self::IndexType;
}

/// Geometry evaluator
///
/// A geometry evaluator can compute points and jacobians on physical cells
pub trait GeometryEvaluator {
    /// Scalar type
    type T: Float + RlstScalar<Real = Self::T>;

    /// The number of points on the reference cell used by this evaluator
    fn point_count(&self) -> usize;

    /// Compute points on a physical cell
    fn compute_points(&self, cell_index: usize, point: &mut [Self::T]);

    /// Compute jacobians on a physical cell
    fn compute_jacobians(&self, cell_index: usize, jacobian: &mut [Self::T]);

    /// Compute normals on a physical cell
    fn compute_normals(&self, cell_index: usize, normal: &mut [Self::T]);
}

/// A grid
pub trait Grid {
    /// Scalar type
    type T: Float + RlstScalar<Real = Self::T>;

    /// The type that implements [Topology]
    type Topology: Topology;

    /// The type that implements [Geometry]
    type Geometry: Geometry<T = Self::T>;

    /// MPI rank
    fn mpi_rank(&self) -> usize {
        0
    }

    /// Get the grid topology (See [Topology])
    fn topology(&self) -> &Self::Topology;

    /// Get the grid geometry (See [Geometry])
    fn geometry(&self) -> &Self::Geometry;

    /// Check if the grid is stored in serial
    fn is_serial(&self) -> bool;
}
