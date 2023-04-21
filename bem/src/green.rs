use num::complex::Complex;
use num::Num;

pub enum GreenParameters {
    None,
    Wavenumber(f64),
}

pub trait Scalar: Num + std::ops::AddAssign {
    /// Get 1 over pi as this scalar type
    fn inv_pi() -> Self;
    /// Get the distance between x and y as this scalar type
    fn dist(x: &[f64], y: &[f64]) -> Self;
    /// Get the square of the distance between x and y as this scalar type
    fn dist_squared(x: &[f64], y: &[f64]) -> Self;
    /// Get the cube of the distance between x and y as this scalar type
    fn dist_cubed(x: &[f64], y: &[f64]) -> Self;
    /// Get 0.25 as this scalar type
    fn quarter() -> Self;
    /// Get the (x-y).n as this scalar type
    fn subdot(x: &[f64], y: &[f64], n: &[f64]) -> Self;
    /// Convert a f64 to this type
    fn from_f64(v: f64) -> Self;
    /// Get e^(i*x) as this scalar type
    fn eix(x: f64) -> Self;
    /// Get i*e^(i*x) as this scalar type
    fn ieix(x: f64) -> Self;
}

impl Scalar for f64 {
    fn inv_pi() -> Self {
        std::f64::consts::FRAC_1_PI
    }
    fn dist(x: &[f64], y: &[f64]) -> Self {
        f64::sqrt(
            (x[0] - y[0]) * (x[0] - y[0])
                + (x[1] - y[1]) * (x[1] - y[1])
                + (x[2] - y[2]) * (x[2] - y[2]),
        )
    }
    fn dist_squared(x: &[f64], y: &[f64]) -> Self {
        (x[0] - y[0]) * (x[0] - y[0])
            + (x[1] - y[1]) * (x[1] - y[1])
            + (x[2] - y[2]) * (x[2] - y[2])
    }
    fn dist_cubed(x: &[f64], y: &[f64]) -> Self {
        let sq = Self::dist_squared(x, y);
        sq * f64::sqrt(sq)
    }
    fn quarter() -> Self {
        0.25
    }
    fn subdot(x: &[f64], y: &[f64], n: &[f64]) -> Self {
        (x[0] - y[0]) * n[0] + (x[1] - y[1]) * n[1] + (x[2] - y[2]) * n[2]
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn eix(x: f64) -> Self {
        Complex::<f64>::eix(x).re
    }
    fn ieix(x: f64) -> Self {
        -Complex::<f64>::eix(x).im
    }
}

impl Scalar for Complex<f64> {
    fn inv_pi() -> Self {
        Complex::<f64>::new(f64::inv_pi(), 0.0)
    }
    fn dist(x: &[f64], y: &[f64]) -> Self {
        Complex::<f64>::new(f64::dist(x, y), 0.0)
    }
    fn dist_squared(x: &[f64], y: &[f64]) -> Self {
        Complex::<f64>::new(f64::dist_squared(x, y), 0.0)
    }
    fn dist_cubed(x: &[f64], y: &[f64]) -> Self {
        Complex::<f64>::new(f64::dist_cubed(x, y), 0.0)
    }
    fn quarter() -> Self {
        Complex::<f64>::new(f64::quarter(), 0.0)
    }
    fn subdot(x: &[f64], y: &[f64], n: &[f64]) -> Self {
        Complex::<f64>::new(f64::subdot(x, y, n), 0.0)
    }
    fn eix(x: f64) -> Self {
        Complex::<f64>::new(0.0, x).exp()
    }
    fn ieix(x: f64) -> Self {
        Complex::<f64>::new(0.0, 1.0) * Complex::<f64>::new(0.0, x).exp()
    }
    fn from_f64(v: f64) -> Self {
        Complex::<f64>::new(v, 0.0)
    }
}

pub fn laplace_green<T: Scalar>(
    x: &[f64],
    y: &[f64],
    _nx: &[f64],
    _ny: &[f64],
    _params: &GreenParameters,
) -> T {
    T::quarter() * T::inv_pi() / T::dist(x, y)
}

pub fn laplace_green_dx<T: Scalar>(
    x: &[f64],
    y: &[f64],
    nx: &[f64],
    _ny: &[f64],
    _params: &GreenParameters,
) -> T {
    T::quarter() * T::inv_pi() * T::subdot(y, x, nx) / T::dist_cubed(x, y)
}

pub fn laplace_green_dy<T: Scalar>(
    x: &[f64],
    y: &[f64],
    _nx: &[f64],
    ny: &[f64],
    _params: &GreenParameters,
) -> T {
    T::quarter() * T::inv_pi() * T::subdot(x, y, ny) / T::dist_cubed(x, y)
}

pub fn helmholtz_green<T: Scalar>(
    x: &[f64],
    y: &[f64],
    _nx: &[f64],
    _ny: &[f64],
    params: &GreenParameters,
) -> T {
    if let GreenParameters::Wavenumber(k) = params {
        let dist = f64::dist(x, y);
        T::quarter() * T::inv_pi() * T::eix(k * dist) / T::from_f64(dist)
    } else {
        panic!("Helmholtz Green's function needs a wavenumber");
    }
}

pub fn helmholtz_green_dx<T: Scalar>(
    x: &[f64],
    y: &[f64],
    nx: &[f64],
    _ny: &[f64],
    params: &GreenParameters,
) -> T {
    if let GreenParameters::Wavenumber(k) = params {
        let sq = f64::dist_squared(x, y);
        let dist = sq.sqrt();
        T::quarter()
            * T::inv_pi()
            * T::subdot(x, y, nx)
            * (T::from_f64(*k) * T::ieix(k * dist) - T::eix(k * dist) / T::from_f64(dist))
            / T::from_f64(sq)
    } else {
        panic!("Helmholtz Green's function needs a wavenumber");
    }
}

pub fn helmholtz_green_dy<T: Scalar>(
    x: &[f64],
    y: &[f64],
    _nx: &[f64],
    ny: &[f64],
    params: &GreenParameters,
) -> T {
    if let GreenParameters::Wavenumber(k) = params {
        let sq = f64::dist_squared(x, y);
        let dist = sq.sqrt();
        T::quarter()
            * T::inv_pi()
            * T::subdot(y, x, ny)
            * (T::from_f64(*k) * T::ieix(k * dist) - T::eix(k * dist) / T::from_f64(dist))
            / T::from_f64(sq)
    } else {
        panic!("Helmholtz Green's function needs a wavenumber");
    }
}
