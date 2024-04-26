//! General type definitions
mod cell;
mod cell_iterator;
mod point_iterator;

pub use cell::*;
pub use cell_iterator::*;
pub use point_iterator::*;

/// Ownership
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Ownership {
    /// Owned
    Owned,
    /// Ghost
    Ghost(usize, usize),
}
