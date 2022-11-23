//! Basic trait description of a Grid.

pub use crate::traits::Geometry;
pub use crate::traits::Topology;

pub trait Grid {
    type Topology<'a>: Topology
    where
        Self: 'a;

    fn topology<'a>(&'a self) -> Self::Topology<'a>;

    fn cell_geometry<'a>(&'a self, local_index: usize) -> &'a dyn Geometry;
}
