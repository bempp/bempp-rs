//! General type definitions
mod cell;
mod cell_iterator;
mod edge_iterator;
mod vertex_iterator;

pub use cell::*;
pub use cell_iterator::*;
pub use edge_iterator::*;
pub use vertex_iterator::*;

/// Ownership
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Ownership {
    /// Owned
    Owned,
    /// Ghost
    Ghost(usize, usize),
}
