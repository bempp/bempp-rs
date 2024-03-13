//! A mixed grid
//!
//! In this grid, cells can be of any type, and the grid may contain multiple cell types

mod builder;
mod geometry;
mod grid;
mod topology;

pub use self::builder::SerialMixedGridBuilder;
pub use self::grid::SerialMixedGrid;
