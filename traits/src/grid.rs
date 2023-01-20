//! Geometry and topology definitions

pub use crate::cell::ReferenceCell;
pub use crate::element::FiniteElement;

pub trait Geometry {
    fn dim(&self) -> usize;

    fn point(&self, i: usize) -> Option<&[f64]>;

    unsafe fn point_unchecked(&self, i: usize) -> &[f64];

    fn point_count(&self) -> usize;
}

pub enum Locality {
    Local,
    Ghost(usize),
    Remote,
}

pub trait Topology {
    // The dimension of the grid
    fn dim(&self) -> usize;

    // Check the locality of an element
    fn locality(&self, global_id: usize) -> Locality;

    // Convert local to global id
    fn local2global(&self, local_id: usize) -> usize;

    // Convert global to local id
    fn global2local(&self, global_id: usize) -> Option<usize>;

    fn entity_count(&self, dim: usize) -> usize;

    fn cell(&self, index: usize) -> &[usize];
}

pub trait Grid {
    type Topology: Topology;
    type Geometry: Geometry;

    fn topology(&self) -> &Self::Topology;

    fn geometry(&self) -> &Self::Geometry;
}
