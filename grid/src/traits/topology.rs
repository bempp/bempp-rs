//! Definition of topologies

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
