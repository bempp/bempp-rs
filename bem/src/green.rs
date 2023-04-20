fn laplace_green(x: &[f64], y: &[f64], _nx: &[f64], _ny: &[f64]) -> f64 {
    let inv_dist = 1.0
        / f64::sqrt(
            (x[0] - y[0]) * (x[0] - y[0])
                + (x[1] - y[1]) * (x[1] - y[1])
                + (x[2] - y[2]) * (x[2] - y[2]),
        );

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist
}

fn laplace_green_dx(x: &[f64], y: &[f64], nx: &[f64], _ny: &[f64]) -> f64 {
    let inv_dist = 1.0
        / f64::sqrt(
            (x[0] - y[0]) * (x[0] - y[0])
                + (x[1] - y[1]) * (x[1] - y[1])
                + (x[2] - y[2]) * (x[2] - y[2]),
        );
    let sum = (y[0] - x[0]) * nx[0] + (y[1] - x[1]) * nx[1] + (y[2] - x[2]) * nx[2];

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist * inv_dist * inv_dist * sum
}

fn laplace_green_dy(x: &[f64], y: &[f64], _nx: &[f64], ny: &[f64]) -> f64 {
    let inv_dist = 1.0
        / f64::sqrt(
            (x[0] - y[0]) * (x[0] - y[0])
                + (x[1] - y[1]) * (x[1] - y[1])
                + (x[2] - y[2]) * (x[2] - y[2]),
        );
    let sum = (x[0] - y[0]) * ny[0] + (x[1] - y[1]) * ny[1] + (x[2] - y[2]) * ny[2];

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist * inv_dist * inv_dist * sum
}
