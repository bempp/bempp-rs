//! Definitions of Duffy transformation rules.
//!
//! Duffy transformation rules are important for the numerical
//! integration of adjacent alements for functions with weakly singualar
//! kernels as for example in boundary element methods.
//! This module defines Duffy transformation rules for triangular
//! and quadrilateral surface elements. The corresponding formulas can be found
//! in the book *Boundary Element Methods* by S. Sauter and C. Schwab.

mod common;
pub mod quadrilateral;
pub mod triangle;
pub mod triangle_quadrilateral;

pub use quadrilateral::quadrilateral_duffy;
pub use triangle::triangle_duffy;
pub use triangle_quadrilateral::{quadrilateral_triangle_duffy, triangle_quadrilateral_duffy};
