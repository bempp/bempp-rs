//! A flat triangle grid
//!
//! In this grid, every cell is a flat triangle

mod builder;
mod grid;
mod io;

pub use self::builder::SerialFlatTriangleGridBuilder;
pub use self::grid::SerialFlatTriangleGrid;
