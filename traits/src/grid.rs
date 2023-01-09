//! Geometry and topology definitions

pub use crate::cell::ReferenceCell;
pub use crate::element::FiniteElement;

pub trait Geometry {
    fn reference_cell(&self) -> &dyn ReferenceCell;
    fn map(&self, reference_coords: &[f64], physical_coords: &mut [f64]);

    fn dim(&self) -> usize;
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
    fn locality(global_id: usize) -> Locality;

    // Convert local to global id
    fn local2global(local_id: usize) -> usize;

    // Convert global to local id
    fn global2local(global_id: usize) -> Option<usize>;
}

pub trait Grid {
    type Topology<'a>: Topology
    where
        Self: 'a;
    type Geometry<'a>: Geometry
    where
        Self: 'a;

    fn topology<'a>(&'a self) -> Self::Topology<'a>;

    fn cell_geometry<'a>(&'a self, local_index: usize) -> Self::Geometry<'a>;
}
