use bempp_field::fft::Fft;
use cauchy::Scalar;
use num::Float;
use rlst_blis::interface::gemm::Gemm;

pub trait FmmScalar: Scalar<Real = Self> + Float + Default + Send + Sync + Fft + Gemm {}

impl<U> FmmScalar for U where U: Scalar<Real = U> + Float + Default + Send + Sync + Fft + Gemm {}
