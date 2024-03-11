//! Traits used in the implementation of a grid

use bempp_traits::types::{ReferenceCellType, CellLocalIndexPair};
use bempp_traits::element::FiniteElement;
use rlst_dense::types::RlstScalar;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Ownership {
    Owned,
    Ghost(usize, usize),
}

/// The topology of a grid.
///
/// This provides information about which mesh entities are connected to other mesh entities
pub trait Topology {
    type IndexType: std::fmt::Debug + Eq + Copy;

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

    /// Get the indices of entities of cell that are connected to the entity with dimension `dim` and index `index`
    fn entity_to_cells(
        &self,
        dim: usize,
        index: Self::IndexType,
    ) -> Option<&[CellLocalIndexPair<Self::IndexType>]>;

    /// Get the indices of the vertices that are connect to theentity with dimension `dim` and index `index`
    fn entity_vertices(&self, dim: usize, index: Self::IndexType) -> Option<&[Self::IndexType]>;

    /// Get the ownership of a mesh entity
    fn entity_ownership(&self, dim: usize, index: Self::IndexType) -> Ownership;

    /// Get the id of a vertex from its index
    fn vertex_index_to_id(&self, index: Self::IndexType) -> usize;

    /// Get the id of a cell from its index
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize;

    /// Get the index of a vertex from its id
    fn vertex_id_to_index(&self, id: usize) -> Self::IndexType;

    /// Get the index of a cell from its id
    fn cell_id_to_index(&self, id: usize) -> Self::IndexType;
}

/// The geometry of a grid
///
/// This provides information about the physical locations of mesh points in space
pub trait Geometry {
    type IndexType: std::fmt::Debug + Eq + Copy;
    type T: RlstScalar<Real=Self::T>;
    type Element: FiniteElement;
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
    type T: RlstScalar<Real=Self::T>;

    /// The number of points on the reference cell used by this evaluator
    fn point_count(&self) -> usize;

    /// Compute a point on a physical cell
    fn compute_point(&self, cell_index: usize, point_index: usize, point: &mut [Self::T]);

    /// Compute a jacobian on a physical cell
    fn compute_jacobian(&self, cell_index: usize, point_index: usize, jacobian: &mut [Self::T]);

    /// Compute a normal on a physical cell
    fn compute_normal(&self, cell_index: usize, point_index: usize, normal: &mut [Self::T]);
}

/// A grid
pub trait Grid {
    type T: RlstScalar<Real=Self::T>;

    /// The type that implements [Topology]
    type Topology: Topology;

    /// The type that implements [Geometry]
    type Geometry: Geometry<T = Self::T>;

    /// Get the grid topology (See [Topology])
    fn topology(&self) -> &Self::Topology;

    /// Get the grid geometry (See [Geometry])
    fn geometry(&self) -> &Self::Geometry;

    /// Check if the grid is stored in serial
    fn is_serial(&self) -> bool;
}
