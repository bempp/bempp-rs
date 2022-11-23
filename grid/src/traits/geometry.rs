//! Geometry definition

pub use solvers_element::cell::ReferenceCell;
pub use solvers_element::element::FiniteElement;

pub trait Geometry {
    fn reference_cell(&self) -> &dyn ReferenceCell;
    fn map(&self, reference_coords: &[f64], physical_coords: &mut [f64]);

    fn dim(&self) -> usize;
}
