//! Geometry and topology definitions

use crate::arrays::AdjacencyListAccess;
use crate::cell::ReferenceCellType;
use crate::element::FiniteElement;
use rlst_dense::traits::{RandomAccessByRef, RandomAccessMut, Shape};

/// The ownership of a mesh entity
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Ownership {
    Owned,
    Ghost(usize, usize),
}

pub trait GeometryEvaluator<
    T: RandomAccessByRef<2, Item = f64> + Shape<2>,
    TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
>
{
    /// The points on the reference cell that this evaluator computes information at
    fn points(&self) -> &T;

    /// Compute the points in a physical cell
    fn compute_points(&self, cell_index: usize, points: &mut TMut);

    /// Compute the normals and jacobian determinants at this evaluator's points
    fn compute_normals_and_jacobian_determinants(
        &self,
        cell_index: usize,
        normals: &mut TMut,
        jdets: &mut [f64],
    );
}

pub trait Geometry {
    //! Grid geometry
    //!
    //! Grid geometry provides information about the physical locations of mesh points in space
    type T: RandomAccessByRef<2, Item = f64> + Shape<2>;
    type TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>;

    /// The geometric dimension
    fn dim(&self) -> usize;

    /// Get one of the coordinates of a point
    fn coordinate(&self, point_index: usize, coord_index: usize) -> Option<&f64>;

    /// The number of points stored in the geometry
    fn point_count(&self) -> usize;

    /// Get the vertex numbers of a cell
    fn cell_vertices(&self, index: usize) -> Option<&[usize]>;

    /// The number of cells
    fn cell_count(&self) -> usize;

    /// Return the index map from the input cell numbers to the storage numbers
    fn index_map(&self) -> &[usize];

    /// Get the evaluator for the given points
    fn get_evaluator<'a>(
        &'a self,
        element: &impl FiniteElement,
        points: &'a Self::T,
    ) -> Box<dyn GeometryEvaluator<Self::T, Self::TMut> + 'a>;

    /// Compute the physical coordinates of a set of points in a given cell
    fn compute_points<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
    >(
        &self,
        points: &T,
        cell: usize,
        physical_points: &mut TMut,
    );

    /// Compute the normals to a set of points in a given cell
    fn compute_normals<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
    >(
        &self,
        points: &T,
        cell: usize,
        normals: &mut TMut,
    );

    /// Evaluate the jacobian at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobians<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
    >(
        &self,
        points: &T,
        cell: usize,
        jacobians: &mut TMut,
    );

    /// Evaluate the determinand of the jacobian at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobian_determinants<T: RandomAccessByRef<2, Item = f64> + Shape<2>>(
        &self,
        points: &T,
        cell: usize,
        jacobian_determinants: &mut [f64],
    );

    /// Evaluate the jacobian inverse at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobian_inverses<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        TMut: RandomAccessByRef<2, Item = f64> + RandomAccessMut<2, Item = f64> + Shape<2>,
    >(
        &self,
        points: &T,
        cell: usize,
        jacobian_inverses: &mut TMut,
    );
}

pub trait Topology<'a> {
    //! Grid topology
    //!
    //! Grid topology provides information about which mesh entities are connected to other mesh entities

    type Connectivity: AdjacencyListAccess<'a, usize>;

    /// The dimension of the grid
    fn dim(&self) -> usize;

    /// Return the index map from the input cell numbers to the storage numbers
    fn index_map(&self) -> &[usize];

    /// The number of entities of dimension `dim`
    fn entity_count(&self, dim: usize) -> usize;

    /// The indices of the vertices of the cell with topological index `index`
    fn cell(&self, index: usize) -> Option<&[usize]>;

    /// The cell type of the cell with topological index `index`
    fn cell_type(&self, index: usize) -> Option<ReferenceCellType>;

    /// Get the connectivity of entities of dimension `dim0` to entities of dimension `dim1`
    fn connectivity(&self, dim0: usize, dim1: usize) -> &Self::Connectivity;

    /// Get the ownership of a mesh entity
    fn entity_ownership(&self, dim: usize, index: usize) -> Ownership;
}

pub trait Grid<'a> {
    //! A grid

    /// The type that implements [Topology]
    type Topology: Topology<'a>;

    /// The type that implements [Geometry]
    type Geometry: Geometry;

    /// Get the grid topology (See [Topology])
    fn topology(&self) -> &Self::Topology;

    /// Get the grid geometry (See [Geometry])
    fn geometry(&self) -> &Self::Geometry;

    // Check if the function space is stored in serial
    fn is_serial(&self) -> bool;
}
