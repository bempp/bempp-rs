//! Finite element definitions

use crate::arrays::{Array2DAccess, Array4DAccess};
use crate::cell::ReferenceCellType;

/// The family of an element
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ElementFamily {
    Lagrange = 0,
    RaviartThomas = 1,
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
fn compute_derivative_count(nderivs: usize, cell_type: ReferenceCellType) -> Result<usize, ()> {
    match cell_type {
        ReferenceCellType::Point => Ok(0),
        ReferenceCellType::Interval => Ok(nderivs + 1),
        ReferenceCellType::Triangle => Ok((nderivs + 1) * (nderivs + 2) / 2),
        ReferenceCellType::Quadrilateral => Ok((nderivs + 1) * (nderivs + 2) / 2),
        ReferenceCellType::Tetrahedron => Ok((nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6),
        ReferenceCellType::Hexahedron => Ok((nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6),
        ReferenceCellType::Prism => Ok((nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6),
        ReferenceCellType::Pyramid => Ok((nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6),
    }
}

pub trait FiniteElement {
    //! A finite element defined on a reference cell

    /// The reference cell type
    fn cell_type(&self) -> ReferenceCellType;

    /// The polynomial degree
    fn degree(&self) -> usize;

    /// The highest degree polynomial in the element's polynomial set
    fn highest_degree(&self) -> usize;

    // The element family
    fn family(&self) -> ElementFamily;

    /// The number of basis functions
    fn dim(&self) -> usize;

    /// Is the element discontinuous between cells?
    fn discontinuous(&self) -> bool;

    /// The value size
    fn value_size(&self) -> usize;

    /// Tabulate the values of the basis functions and their derivatives at a set of points
    fn tabulate<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        nderivs: usize,
        data: &mut impl Array4DAccess<f64>,
    );

    /// The DOFs that are associated with a subentity of the reference cell
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize>;

    /// The push forward / pull back map to use for this element
    fn map_type(&self) -> MapType;

    /// Get the required shape for a tabulation array
    fn tabulate_array_shape(&self, nderivs: usize, npoints: usize) -> (usize, usize, usize, usize) {
        let deriv_count = compute_derivative_count(nderivs, self.cell_type()).unwrap();
        let point_count = npoints;
        let basis_count = self.dim();
        let value_size = self.value_size();
        (deriv_count, point_count, basis_count, value_size)
    }
}
