//! A single element grid
//!
//! In this grid, cells can be of any type but must all be the same type

mod builder;
mod geometry;
mod grid;
mod topology;

pub use self::builder::SerialSingleElementGridBuilder;
pub use self::grid::SerialSingleElementGrid;
