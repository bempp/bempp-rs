use bempp_field::fft::Fft;
use cauchy::Scalar;
use num::Float;
use rlst_blis::interface::gemm::Gemm;

/// Supertrait for scalars compatible with the FMM
pub trait FmmScalar: Scalar<Real = Self> + Float + Default + Send + Sync + Fft + Gemm {}

/// Blanket implemntation of FmmScalar
impl<U> FmmScalar for U where U: Scalar<Real = U> + Float + Default + Send + Sync + Fft + Gemm {}
