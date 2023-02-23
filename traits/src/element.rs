//! Finite element definitions

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

/// Tabulated data
pub struct TabulatedData {
    data: Vec<f64>,
    deriv_count: usize,
    point_count: usize,
    basis_count: usize,
    value_size: usize,
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

impl TabulatedData {
    /// Create a data array full of zeros
    pub fn new(element: &impl FiniteElement, nderivs: usize, npoints: usize) -> Self {
        let deriv_count = compute_derivative_count(nderivs, element.cell_type()).unwrap();
        let point_count = npoints;
        let basis_count = element.dim();
        let value_size = element.value_size();
        let data = vec![0.0; deriv_count * point_count * basis_count * value_size];
        Self {
            data,
            deriv_count,
            point_count,
            basis_count,
            value_size,
        }
    }

    /// Get a mutable item from the tabulated data
    pub fn get_mut(
        &mut self,
        deriv: usize,
        point: usize,
        basis: usize,
        component: usize,
    ) -> &mut f64 {
        // TODO: Debug here
        let index = ((deriv * self.point_count + point) * self.basis_count + basis)
            * self.value_size
            + component;
        self.data.get_mut(index).unwrap()
    }

    /// Get an item from the tabulated data
    pub fn get(&mut self, deriv: usize, point: usize, basis: usize, component: usize) -> &f64 {
        // TODO: Debug here
        let index = ((deriv * self.point_count + point) * self.basis_count + basis)
            * self.value_size
            + component;
        self.data.get(index).unwrap()
    }

    /// The number of derivatives (first dimension of the data)
    pub fn deriv_count(&self) -> usize {
        self.deriv_count
    }
    /// The number of points (second dimension of the data)
    pub fn point_count(&self) -> usize {
        self.point_count
    }
    /// The number of basis functions (third dimension of the data)
    pub fn basis_count(&self) -> usize {
        self.basis_count
    }
    /// The value size (fourth dimension of the data)
    pub fn value_size(&self) -> usize {
        self.value_size
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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData);

    /// The DOFs that are associated with a subentity of the reference cell
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize>;

    /// The push forward / pull back map to use for this element
    fn map_type(&self) -> MapType;
}
