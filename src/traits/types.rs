//! General type definitions
pub mod cell;
pub mod cell_iterator;
pub mod point_iterator;

pub use cell::{CellLocalIndexPair, ReferenceCellType};

/// Ownership
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Ownership {
    /// Owned
    Owned,
    /// Ghost
    Ghost(usize, usize),
}
