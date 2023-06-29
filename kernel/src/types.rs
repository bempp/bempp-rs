use bempp_traits::types::c64;

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

/// This enum defines the type of the kernel.
#[derive(Clone, Copy)]
pub enum KernelType {
    /// The Laplace kernel defined as g(x, y) = 1 / (4 pi | x- y| )
    Laplace,
    /// The Helmholtz kernel defined as g(x, y) = exp( 1j * k * | x- y| ) / (4 pi | x- y| )
    Helmholtz(c64),
    /// The modified Helmholtz kernel defined as g(x, y) = exp( -omega * | x- y| ) / (4 * pi * | x- y |)
    ModifiedHelmholtz(f64),
}
