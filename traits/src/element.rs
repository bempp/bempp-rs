//! Finite element definitions

use crate::types::ReferenceCellType;
use rlst_dense::traits::{RandomAccessByRef, RandomAccessMut, Shape};
use rlst_dense::types::RlstScalar;

/// Continuity type
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum Continuity {
    /// The element has standard continuity between cells
    ///
    /// For some element, this option does not indicate that the values are fully continuous.
    /// For example, for Raviart-Thomas elements it only indicates that the normal components
    /// are continuous across edges
    Continuous = 0,
    /// The element is discontinuous betweeen cells
    Discontinuous = 1,
}

/// The map type used by an element
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum MapType {
    /// Identity map
    Identity = 0,
    /// Covariant Piola map
    ///
    /// This map is used by H(curl) elements
    CovariantPiola = 1,
    /// Contravariant Piola map
    ///
    /// This map is used by H(div) elements
    ContravariantPiola = 2,
    /// L2 Piola map
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
    /// The scalar type
    type T: RlstScalar;

    /// The reference cell type
    fn cell_type(&self) -> ReferenceCellType;

    /// The highest degree polynomial in the element's polynomial set
    fn embedded_superdegree(&self) -> usize;

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
        Array2: RandomAccessByRef<2, Item = <Self::T as RlstScalar>::Real> + Shape<2>,
        Array4Mut: RandomAccessMut<4, Item = Self::T>,
    >(
        &self,
        points: &Array2,
        nderivs: usize,
        data: &mut Array4Mut,
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
