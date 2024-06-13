//! Types specific to bempp-rs

use rlst::{LinAlg, RlstScalar};

pub trait RealScalar: num::Float + LinAlg + RlstScalar<Real = Self> {}

impl<T: num::Float + LinAlg + RlstScalar<Real = T>> RealScalar for T {}
