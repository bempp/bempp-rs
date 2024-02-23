//! Finite element definitions

use crate::cell::ReferenceCellType;
use rlst_dense::traits::{RandomAccessByRef, RandomAccessMut, Shape};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum Continuity {
    Continuous = 0,
    Discontinuous = 1,
}

/// The map type used by an element
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum MapType {
    Identity = 0,
    CovariantPiola = 1,
    ContravariantPiola = 2,
    L2Piola = 3,
}

/// Compute the number of derivatives for a cell
fn compute_derivative_count(nderivs: usize, cell_type: ReferenceCellType) -> usize {
    match cell_type {
        ReferenceCellType::Point => 0,
        ReferenceCellType::Interval => nderivs + 1,
        ReferenceCellType::Triangle => (nderivs + 1) * (nderivs + 2) / 2,
        ReferenceCellType::Quadrilateral => (nderivs + 1) * (nderivs + 2) / 2,
        ReferenceCellType::Tetrahedron => (nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6,
        ReferenceCellType::Hexahedron => (nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6,
        ReferenceCellType::Prism => (nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6,
        ReferenceCellType::Pyramid => (nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6,
    }
}

pub trait FiniteElement {
    //! A finite element defined on a reference cell

    /// The reference cell type
    fn cell_type(&self) -> ReferenceCellType;

    /// The highest degree polynomial in the element's polynomial set
    fn embedded_superdegree(&self) -> usize;

    /// Check if the element is a Lagrange element
    fn is_lagrange(&self) -> bool;

    /// The number of basis functions
    fn dim(&self) -> usize;

    /// Type of continuity between cells
    fn continuity(&self) -> Continuity;

    /// The value shape
    fn value_shape(&self) -> &[usize];

    /// The value size
    fn value_size(&self) -> usize;

    /// Tabulate the values of the basis functions and their derivatives at a set of points
    fn tabulate<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        T4Mut: RandomAccessMut<4, Item = f64>,
    >(
        &self,
        points: &T,
        nderivs: usize,
        data: &mut T4Mut,
    );

    /// The DOFs that are associated with a subentity of the reference cell
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]>;

    /// The push forward / pull back map to use for this element
    fn map_type(&self) -> MapType;

    /// Get the required shape for a tabulation array
    fn tabulate_array_shape(&self, nderivs: usize, npoints: usize) -> [usize; 4] {
        let deriv_count = compute_derivative_count(nderivs, self.cell_type());
        let point_count = npoints;
        let basis_count = self.dim();
        let value_size = self.value_size();
        [deriv_count, point_count, basis_count, value_size]
    }
}
