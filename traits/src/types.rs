//! General type definitions

// Definition of scalar types.
// For now we simply derive from the `caucy::Scalar` type.
pub use cauchy::Scalar;
pub use cauchy::{c32, c64};

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
