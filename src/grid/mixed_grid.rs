//! A mixed grid
//!
//! In this grid, cells can be of any type, and the grid may contain multiple cell types

mod builder;
mod geometry;
mod grid;
mod io;
mod topology;

pub use self::builder::MixedGridBuilder;
pub use self::grid::MixedGrid;
