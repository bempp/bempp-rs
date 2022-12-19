//! General type definitions

// Definition of scalar types.
// For now we simply derive from the `caucy::Scalar` type.
pub trait Scalar {}
impl<T: cauchy::Scalar> Scalar for T {}

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

// Evaluation Mode.
//
// - `Value`: Declares that only values required.
// - `Deriv`: Declare that only derivative required.
// - `ValueDeriv` Both values and derivatives required.
pub enum EvalType {
    Value,
    Deriv,
    ValueDeriv,
}
