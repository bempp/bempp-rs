//! A single element grid
//!
//! In this grid, cells can be of any type but must all be the same type

mod builder;
mod geometry;
mod grid;
mod io;
#[cfg(feature = "mpi")]
mod parallel;
mod topology;

pub use self::builder::SingleElementGridBuilder;
pub use self::grid::SingleElementGrid;
