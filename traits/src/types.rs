//! General type definitions
pub use rlst_dense::types::{c32, c64, RlstScalar};
pub mod cell;
pub mod cell_iterator;
pub mod point_iterator;

pub use cell::{CellLocalIndexPair, ReferenceCellType};

// Declare if entity is local, a ghost, or remote.
pub enum Locality {
    Local,
    Ghost,
    Remote,
}

//Generic error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Solver Error: {0}")]
    Generic(String),
}

// Result Type
pub type Result<T> = std::result::Result<T, Error>;

/// Evaluation Mode.
///
/// - `Value`: Declares that only values required.
/// - `Deriv`: Declare that only derivative required.
/// - `ValueDeriv` Both values and derivatives required.
#[derive(Clone, Copy)]
pub enum EvalType {
    Value,
    ValueDeriv,
}
