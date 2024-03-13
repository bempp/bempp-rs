//! A Rust grid library
#![cfg_attr(feature = "strict", deny(warnings))]

//pub mod io;

pub mod common;
pub mod flat_triangle_grid;
pub mod mixed_grid;
pub mod shapes;
pub mod single_element_grid;
pub mod traits;
pub mod traits_impl;

pub use self::traits::Geometry;
pub use self::traits::Grid;
pub use self::traits::Topology;
