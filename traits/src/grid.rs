//! Geometry and topology definitions

use solvers_tools::arrays::AdjacencyList;

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

    // Return the index map from the input order to the storage order
    fn index_map(&self) -> &[usize];

    // Check the locality of an element
    fn locality(&self, global_id: usize) -> Locality;

    // Convert local to global id
    fn local2global(&self, local_id: usize) -> usize;

    // Convert global to local id
    fn global2local(&self, global_id: usize) -> Option<usize>;

    fn entity_count(&self, dim: usize) -> usize;

    fn cell(&self, index: usize) -> Option<&[usize]>;
    unsafe fn cell_unchecked(&self, index: usize) -> &[usize];

    fn create_connectivity(&mut self, dim0: usize, dim1: usize);
    fn create_connectivity_all(&mut self) {
        for dim0 in 0..self.dim() {
            for dim1 in 0..self.dim() {
                self.create_connectivity(dim0, dim1);
            }
        }
    }
    fn connectivity(&self, dim0: usize, dim1: usize) -> &AdjacencyList<usize>;
}

pub trait Grid {
    type Topology: Topology;
    type Geometry: Geometry;

    fn topology(&self) -> &Self::Topology;
    fn topology_mut(&mut self) -> &mut Self::Topology;

    fn geometry(&self) -> &Self::Geometry;
}
