//! A flat triangle grid
//!
//! In this grid, every cell is a flat triangle

mod builder;
mod entities;
mod grid;
mod io;
#[cfg(feature = "mpi")]
mod parallel;

// pub use self::builder::FlatTriangleGridBuilder;
pub use self::grid::FlatTriangleGrid;
