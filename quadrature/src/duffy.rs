//! Definitions of Duffy transformation rules.
//!
//! Duffy transformation rules are important for the numerical
//! integration of adjacent alements for functions with weakly singualar
//! kernels as for example in boundary element methods.
//! This module defines Duffy transformation rules for triangular
//! and quadrilateral surface elements. The corresponding formulas can be found
//! in the book *Boundary Element Methods* by S. Sauter and C. Schwab.

pub mod quadrilateral;
pub mod triangle;

pub use quadrilateral::quadrilateral_duffy;
pub use triangle::triangle_duffy;

pub use quadrilateral::quadrilateral_duffy as triangle_quadrilateral_duffy;
pub use quadrilateral::quadrilateral_duffy as quadrilateral_triangle_duffy;
