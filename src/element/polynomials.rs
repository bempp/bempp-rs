//! Orthonormal polynomials

use crate::traits::types::ReferenceCellType;
use rlst::RlstScalar;
use rlst::{RandomAccessByRef, RandomAccessMut, Shape};
/// Tabulate orthonormal polynomials on a interval
fn tabulate_legendre_polynomials_interval<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    assert_eq!(data.shape()[0], derivatives + 1);
    assert_eq!(data.shape()[1], degree + 1);
    assert_eq!(data.shape()[2], points.shape()[0]);
    assert_eq!(points.shape()[1], 1);

    for i in 0..data.shape()[2] {
        *data.get_mut([0, 0, i]).unwrap() = T::from(1.0).unwrap();
    }
    for k in 1..data.shape()[0] {
        for i in 0..data.shape()[2] {
            *data.get_mut([k, 0, i]).unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..derivatives + 1 {
        for p in 1..degree + 1 {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data.get([k, p - 1, i]).unwrap();
                *data.get_mut([k, p, i]).unwrap() =
                    (T::from(*points.get([i, 0]).unwrap()).unwrap() * T::from(2.0).unwrap()
                        - T::from(1.0).unwrap())
                        * d
                        * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data.get([k, p - 2, i]).unwrap();
                    *data.get_mut([k, p, i]).unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data.get([k - 1, p - 1, i]).unwrap();
                    *data.get_mut([k, p, i]).unwrap() +=
                        T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }
}

fn tri_index(i: usize, j: usize) -> usize {
    (i + j + 1) * (i + j) / 2 + j
}

fn quad_index(i: usize, j: usize, n: usize) -> usize {
    j * (n + 1) + i
}

/// Tabulate orthonormal polynomials on a quadrilateral
fn tabulate_legendre_polynomials_quadrilateral<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    assert_eq!(data.shape()[0], (derivatives + 1) * (derivatives + 2) / 2);
    assert_eq!(data.shape()[1], (degree + 1) * (degree + 1));
    assert_eq!(data.shape()[2], points.shape()[0]);
    assert_eq!(points.shape()[1], 2);

    for i in 0..data.shape()[2] {
        *data
            .get_mut([tri_index(0, 0), quad_index(0, 0, degree), i])
            .unwrap() = T::from(1.0).unwrap();
    }

    // Tabulate polynomials in x
    for k in 1..derivatives + 1 {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tri_index(k, 0), quad_index(0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..derivatives + 1 {
        for p in 1..degree + 1 {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data
                    .get([tri_index(k, 0), quad_index(p - 1, 0, degree), i])
                    .unwrap();
                *data
                    .get_mut([tri_index(k, 0), quad_index(p, 0, degree), i])
                    .unwrap() = (T::from(*points.get([i, 0]).unwrap()).unwrap()
                    * T::from(2.0).unwrap()
                    - T::from(1.0).unwrap())
                    * d
                    * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(k, 0), quad_index(p - 2, 0, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(k, 0), quad_index(p, 0, degree), i])
                        .unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(k - 1, 0), quad_index(p - 1, 0, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(k, 0), quad_index(p, 0, degree), i])
                        .unwrap() += T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }

    // Tabulate polynomials in y
    for k in 1..derivatives + 1 {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tri_index(0, k), quad_index(0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..derivatives + 1 {
        for p in 1..degree + 1 {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data
                    .get([tri_index(0, k), quad_index(0, p - 1, degree), i])
                    .unwrap();
                *data
                    .get_mut([tri_index(0, k), quad_index(0, p, degree), i])
                    .unwrap() = (T::from(*points.get([i, 1]).unwrap()).unwrap()
                    * T::from(2.0).unwrap()
                    - T::from(1.0).unwrap())
                    * d
                    * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(0, k), quad_index(0, p - 2, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(0, k), quad_index(0, p, degree), i])
                        .unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(0, k - 1), quad_index(0, p - 1, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(0, k), quad_index(0, p, degree), i])
                        .unwrap() += T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }

    // Fill in the rest of the values as products
    for kx in 0..derivatives + 1 {
        for ky in 0..derivatives + 1 - kx {
            for px in 1..degree + 1 {
                for py in 1..degree + 1 {
                    for i in 0..data.shape()[2] {
                        let d = *data
                            .get([tri_index(0, ky), quad_index(0, py, degree), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), quad_index(px, py, degree), i])
                            .unwrap() = *data
                            .get([tri_index(kx, 0), quad_index(px, 0, degree), i])
                            .unwrap()
                            * d;
                    }
                }
            }
        }
    }
}
/// Tabulate orthonormal polynomials on a triangle
fn tabulate_legendre_polynomials_triangle<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    assert_eq!(data.shape()[0], (derivatives + 1) * (derivatives + 2) / 2);
    assert_eq!(data.shape()[1], (degree + 1) * (degree + 2) / 2);
    assert_eq!(data.shape()[2], points.shape()[0]);
    assert_eq!(points.shape()[1], 2);

    for i in 0..data.shape()[2] {
        *data.get_mut([tri_index(0, 0), tri_index(0, 0), i]).unwrap() =
            T::sqrt(T::from(2.0).unwrap());
    }

    for k in 1..data.shape()[0] {
        for i in 0..data.shape()[2] {
            *data.get_mut([k, tri_index(0, 0), i]).unwrap() = T::from(0.0).unwrap();
        }
    }

    for kx in 0..derivatives + 1 {
        for ky in 0..derivatives + 1 - kx {
            for p in 1..degree + 1 {
                let a = T::from(2.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
                let scale1 = T::sqrt(
                    (T::from(p).unwrap() + T::from(0.5).unwrap())
                        * (T::from(p).unwrap() + T::from(1.0).unwrap())
                        / ((T::from(p).unwrap() - T::from(0.5).unwrap()) * T::from(p).unwrap()),
                );
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(kx, ky), tri_index(0, p - 1), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                        .unwrap() = (T::from(*points.get([i, 0]).unwrap()).unwrap()
                        * T::from(2.0).unwrap()
                        + T::from(*points.get([i, 1]).unwrap()).unwrap()
                        - T::from(1.0).unwrap())
                        * d
                        * a
                        * scale1;
                }
                if kx > 0 {
                    for i in 0..data.shape()[2] {
                        let d = *data
                            .get([tri_index(kx - 1, ky), tri_index(0, p - 1), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                            .unwrap() +=
                            T::from(2.0).unwrap() * T::from(kx).unwrap() * a * d * scale1;
                    }
                }
                if ky > 0 {
                    for i in 0..data.shape()[2] {
                        let d = *data
                            .get([tri_index(kx, ky - 1), tri_index(0, p - 1), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                            .unwrap() += T::from(ky).unwrap() * a * d * scale1;
                    }
                }
                if p > 1 {
                    let scale2 = T::sqrt(
                        (T::from(p).unwrap() + T::from(0.5).unwrap())
                            * (T::from(p).unwrap() + T::from(1.0).unwrap()),
                    ) / T::sqrt(
                        (T::from(p).unwrap() - T::from(1.5).unwrap())
                            * (T::from(p).unwrap() - T::from(1.0).unwrap()),
                    );

                    for i in 0..data.shape()[2] {
                        let b =
                            T::from(1.0).unwrap() - T::from(*points.get([i, 1]).unwrap()).unwrap();
                        let d = *data
                            .get([tri_index(kx, ky), tri_index(0, p - 2), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                            .unwrap() -= b * b * d * (a - T::from(1.0).unwrap()) * scale2;
                    }
                    if ky > 0 {
                        for i in 0..data.shape()[2] {
                            let d = *data
                                .get([tri_index(kx, ky - 1), tri_index(0, p - 2), i])
                                .unwrap();
                            *data
                                .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                                .unwrap() -= T::from(2.0).unwrap()
                                * T::from(ky).unwrap()
                                * (T::from(*points.get([i, 1]).unwrap()).unwrap()
                                    - T::from(1.0).unwrap())
                                * d
                                * scale2
                                * (a - T::from(1.0).unwrap());
                        }
                    }
                    if ky > 1 {
                        for i in 0..data.shape()[2] {
                            let d = *data
                                .get([tri_index(kx, ky - 2), tri_index(0, p - 2), i])
                                .unwrap();
                            *data
                                .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                                .unwrap() -= T::from(ky).unwrap()
                                * (T::from(ky).unwrap() - T::from(1.0).unwrap())
                                * d
                                * scale2
                                * (a - T::from(1.0).unwrap());
                        }
                    }
                }
            }
            for p in 0..degree {
                let scale3 = T::sqrt(
                    (T::from(p).unwrap() + T::from(2.0).unwrap())
                        / (T::from(p).unwrap() + T::from(1.0).unwrap()),
                );
                for i in 0..data.shape()[2] {
                    *data
                        .get_mut([tri_index(kx, ky), tri_index(1, p), i])
                        .unwrap() = *data.get([tri_index(kx, ky), tri_index(0, p), i]).unwrap()
                        * scale3
                        * ((T::from(*points.get([i, 1]).unwrap()).unwrap()
                            * T::from(2.0).unwrap()
                            - T::from(1.0).unwrap())
                            * (T::from(1.5).unwrap() + T::from(p).unwrap())
                            + T::from(0.5).unwrap()
                            + T::from(p).unwrap());
                }
                if ky > 0 {
                    for i in 0..data.shape()[2] {
                        let d = *data
                            .get([tri_index(kx, ky - 1), tri_index(0, p), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(1, p), i])
                            .unwrap() += T::from(2.0).unwrap()
                            * T::from(ky).unwrap()
                            * (T::from(1.5).unwrap() + T::from(p).unwrap())
                            * d
                            * scale3;
                    }
                }
                for q in 1..degree - p {
                    let scale4 = T::sqrt(
                        (T::from(p).unwrap() + T::from(q).unwrap() + T::from(2.0).unwrap())
                            / (T::from(p).unwrap() + T::from(q).unwrap() + T::from(1.0).unwrap()),
                    );
                    let scale5 = T::sqrt(
                        (T::from(p).unwrap() + T::from(q).unwrap() + T::from(2.0).unwrap())
                            / (T::from(p).unwrap() + T::from(q).unwrap()),
                    );
                    let a1 = T::from((p + q + 1) * (2 * p + 2 * q + 3)).unwrap()
                        / T::from((q + 1) * (2 * p + q + 2)).unwrap();
                    let a2 = T::from((2 * p + 1) * (2 * p + 1) * (p + q + 1)).unwrap()
                        / T::from((q + 1) * (2 * p + q + 2) * (2 * p + 2 * q + 1)).unwrap();
                    let a3 = T::from(q * (2 * p + q + 1) * (2 * p + 2 * q + 3)).unwrap()
                        / T::from((q + 1) * (2 * p + q + 2) * (2 * p + 2 * q + 1)).unwrap();

                    for i in 0..data.shape()[2] {
                        let d = *data.get([tri_index(kx, ky), tri_index(q, p), i]).unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(q + 1, p), i])
                            .unwrap() = d
                            * scale4
                            * ((T::from(*points.get([i, 1]).unwrap()).unwrap()
                                * T::from(T::from(2.0).unwrap()).unwrap()
                                - T::from(T::from(1.0).unwrap()).unwrap())
                                * a1
                                + a2)
                            - *data
                                .get([tri_index(kx, ky), tri_index(q - 1, p), i])
                                .unwrap()
                                * scale5
                                * a3;
                    }
                    if ky > 0 {
                        for i in 0..data.shape()[2] {
                            let d = *data
                                .get([tri_index(kx, ky - 1), tri_index(q, p), i])
                                .unwrap();
                            *data
                                .get_mut([tri_index(kx, ky), tri_index(q + 1, p), i])
                                .unwrap() += T::from(T::from(2.0).unwrap() * T::from(ky).unwrap())
                                .unwrap()
                                * a1
                                * d
                                * scale4;
                        }
                    }
                }
            }
        }
    }
}

/// The number of polynomials
pub fn polynomial_count(cell_type: ReferenceCellType, degree: usize) -> usize {
    match cell_type {
        ReferenceCellType::Interval => degree + 1,
        ReferenceCellType::Triangle => (degree + 1) * (degree + 2) / 2,
        ReferenceCellType::Quadrilateral => (degree + 1) * (degree + 1),
        _ => {
            panic!("Unsupported cell type");
        }
    }
}

/// The total number of partial derivatives up to a give degree
pub fn derivative_count(cell_type: ReferenceCellType, derivatives: usize) -> usize {
    match cell_type {
        ReferenceCellType::Interval => derivatives + 1,
        ReferenceCellType::Triangle => (derivatives + 1) * (derivatives + 2) / 2,
        ReferenceCellType::Quadrilateral => (derivatives + 1) * (derivatives + 2) / 2,
        _ => {
            panic!("Unsupported cell type");
        }
    }
}

/// The shape of a table containing the values of Legendre polynomials
pub fn legendre_shape<T, Array2: RandomAccessByRef<2, Item = T> + Shape<2>>(
    cell_type: ReferenceCellType,
    points: &Array2,
    degree: usize,
    derivatives: usize,
) -> [usize; 3] {
    [
        derivative_count(cell_type, derivatives),
        polynomial_count(cell_type, degree),
        points.shape()[0],
    ]
}

/// Tabulate orthonormal polynomials
pub fn tabulate_legendre_polynomials<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    cell_type: ReferenceCellType,
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    match cell_type {
        ReferenceCellType::Interval => {
            tabulate_legendre_polynomials_interval(points, degree, derivatives, data)
        }
        ReferenceCellType::Triangle => {
            tabulate_legendre_polynomials_triangle(points, degree, derivatives, data)
        }
        ReferenceCellType::Quadrilateral => {
            tabulate_legendre_polynomials_quadrilateral(points, degree, derivatives, data)
        }
        _ => {
            panic!("Unsupported cell type");
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::quadrature::simplex_rules::simplex_rule;
    use approx::*;
    use rlst::{rlst_dynamic_array2, rlst_dynamic_array3, RandomAccessMut};

    #[test]
    fn test_legendre_interval() {
        let degree = 6;

        let rule = simplex_rule(ReferenceCellType::Interval, degree + 1).unwrap();

        let mut points = rlst_dynamic_array2!(f64, [rule.npoints, 1]);
        for (i, j) in rule.points.iter().enumerate() {
            *points.get_mut([i, 0]).unwrap() = *j;
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Interval, &points, degree, 0,)
        );
        tabulate_legendre_polynomials(ReferenceCellType::Interval, &points, degree, 0, &mut data);

        for i in 0..degree + 1 {
            for j in 0..degree + 1 {
                let mut product = 0.0;
                for k in 0..rule.npoints {
                    product += data.get([0, i, k]).unwrap()
                        * data.get([0, j, k]).unwrap()
                        * rule.weights[k];
                }
                if i == j {
                    assert_relative_eq!(product, 1.0, epsilon = 1e-12);
                } else {
                    assert_relative_eq!(product, 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_legendre_triangle() {
        let degree = 5;

        let rule = simplex_rule(ReferenceCellType::Triangle, 79).unwrap();
        let mut points = rlst_dynamic_array2!(f64, [rule.npoints, 2]);
        for i in 0..rule.npoints {
            for j in 0..2 {
                *points.get_mut([i, j]).unwrap() = rule.points[i * 2 + j];
            }
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Triangle, &points, degree, 0,)
        );
        tabulate_legendre_polynomials(ReferenceCellType::Triangle, &points, degree, 0, &mut data);

        for i in 0..data.shape()[1] {
            for j in 0..data.shape()[1] {
                let mut product = 0.0;
                for k in 0..rule.npoints {
                    product += data.get([0, i, k]).unwrap()
                        * data.get([0, j, k]).unwrap()
                        * rule.weights[k];
                }
                if i == j {
                    assert_relative_eq!(product, 1.0, epsilon = 1e-12);
                } else {
                    assert_relative_eq!(product, 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_legendre_quadrilateral() {
        let degree = 5;

        let rule = simplex_rule(ReferenceCellType::Quadrilateral, 85).unwrap();
        let mut points = rlst_dynamic_array2!(f64, [rule.npoints, 2]);
        for i in 0..rule.npoints {
            for j in 0..2 {
                *points.get_mut([i, j]).unwrap() = rule.points[i * 2 + j];
            }
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Quadrilateral, &points, degree, 0,)
        );
        tabulate_legendre_polynomials(
            ReferenceCellType::Quadrilateral,
            &points,
            degree,
            0,
            &mut data,
        );

        for i in 0..data.shape()[1] {
            for j in 0..data.shape()[1] {
                let mut product = 0.0;
                for k in 0..rule.npoints {
                    product += data.get([0, i, k]).unwrap()
                        * data.get([0, j, k]).unwrap()
                        * rule.weights[k];
                }
                if i == j {
                    assert_relative_eq!(product, 1.0, epsilon = 1e-12);
                } else {
                    assert_relative_eq!(product, 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_legendre_interval_derivative() {
        let degree = 6;

        let epsilon = 1e-10;
        let mut points = rlst_dynamic_array2!(f64, [20, 1]);
        for i in 0..10 {
            *points.get_mut([2 * i, 0]).unwrap() = i as f64 / 10.0;
            *points.get_mut([2 * i + 1, 0]).unwrap() = points.get([2 * i, 0]).unwrap() + epsilon;
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Interval, &points, degree, 1,)
        );
        tabulate_legendre_polynomials(ReferenceCellType::Interval, &points, degree, 1, &mut data);

        for i in 0..degree + 1 {
            for k in 0..points.shape()[0] / 2 {
                assert_relative_eq!(
                    *data.get([1, i, 2 * k]).unwrap(),
                    (data.get([0, i, 2 * k + 1]).unwrap() - data.get([0, i, 2 * k]).unwrap())
                        / epsilon,
                    epsilon = 1e-4
                );
            }
        }
    }

    #[test]
    fn test_legendre_triangle_derivative() {
        let degree = 6;

        let epsilon = 1e-10;
        let mut points = rlst_dynamic_array2!(f64, [165, 2]);
        let mut index = 0;
        for i in 0..10 {
            for j in 0..10 - i {
                *points.get_mut([3 * index, 0]).unwrap() = i as f64 / 10.0;
                *points.get_mut([3 * index, 1]).unwrap() = j as f64 / 10.0;
                *points.get_mut([3 * index + 1, 0]).unwrap() =
                    *points.get([3 * index, 0]).unwrap() + epsilon;
                *points.get_mut([3 * index + 1, 1]).unwrap() = *points.get([3 * index, 1]).unwrap();
                *points.get_mut([3 * index + 2, 0]).unwrap() = *points.get([3 * index, 0]).unwrap();
                *points.get_mut([3 * index + 2, 1]).unwrap() =
                    *points.get([3 * index, 1]).unwrap() + epsilon;
                index += 1;
            }
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Triangle, &points, degree, 1,)
        );
        tabulate_legendre_polynomials(ReferenceCellType::Triangle, &points, degree, 1, &mut data);

        for i in 0..degree + 1 {
            for k in 0..points.shape()[0] / 3 {
                assert_relative_eq!(
                    *data.get([1, i, 3 * k]).unwrap(),
                    (data.get([0, i, 3 * k + 1]).unwrap() - data.get([0, i, 3 * k]).unwrap())
                        / epsilon,
                    epsilon = 1e-4
                );
                assert_relative_eq!(
                    *data.get([2, i, 3 * k]).unwrap(),
                    (data.get([0, i, 3 * k + 2]).unwrap() - data.get([0, i, 3 * k]).unwrap())
                        / epsilon,
                    epsilon = 1e-4
                );
            }
        }
    }

    #[test]
    fn test_legendre_quadrilateral_derivative() {
        let degree = 6;

        let epsilon = 1e-10;
        let mut points = rlst_dynamic_array2!(f64, [300, 2]);
        for i in 0..10 {
            for j in 0..10 {
                let index = 10 * i + j;
                *points.get_mut([3 * index, 0]).unwrap() = i as f64 / 10.0;
                *points.get_mut([3 * index, 1]).unwrap() = j as f64 / 10.0;
                *points.get_mut([3 * index + 1, 0]).unwrap() =
                    *points.get([3 * index, 0]).unwrap() + epsilon;
                *points.get_mut([3 * index + 1, 1]).unwrap() = *points.get([3 * index, 1]).unwrap();
                *points.get_mut([3 * index + 2, 0]).unwrap() = *points.get([3 * index, 0]).unwrap();
                *points.get_mut([3 * index + 2, 1]).unwrap() =
                    *points.get([3 * index, 1]).unwrap() + epsilon;
            }
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Quadrilateral, &points, degree, 1,)
        );
        tabulate_legendre_polynomials(
            ReferenceCellType::Quadrilateral,
            &points,
            degree,
            1,
            &mut data,
        );

        for i in 0..degree + 1 {
            for k in 0..points.shape()[0] / 3 {
                assert_relative_eq!(
                    *data.get([1, i, 3 * k]).unwrap(),
                    (data.get([0, i, 3 * k + 1]).unwrap() - data.get([0, i, 3 * k]).unwrap())
                        / epsilon,
                    epsilon = 1e-4
                );
                assert_relative_eq!(
                    *data.get([2, i, 3 * k]).unwrap(),
                    (data.get([0, i, 3 * k + 2]).unwrap() - data.get([0, i, 3 * k]).unwrap())
                        / epsilon,
                    epsilon = 1e-4
                );
            }
        }
    }

    #[test]
    fn test_legendre_interval_against_known_polynomials() {
        let degree = 3;

        let mut points = rlst_dynamic_array2!(f64, [11, 1]);
        for i in 0..11 {
            *points.get_mut([i, 0]).unwrap() = i as f64 / 10.0;
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Interval, &points, degree, 3,)
        );
        tabulate_legendre_polynomials(ReferenceCellType::Interval, &points, degree, 3, &mut data);

        for k in 0..points.shape()[0] {
            let x = *points.get([k, 0]).unwrap();

            // 0 => 1
            assert_relative_eq!(*data.get([0, 0, k]).unwrap(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([1, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([2, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([3, 0, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 1 => sqrt(3)*(2x - 1)
            assert_relative_eq!(
                *data.get([0, 1, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 1, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 1, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([3, 1, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 2 => sqrt(5)*(6x^2 - 6x + 1)
            assert_relative_eq!(
                *data.get([0, 2, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 2, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * x - 6.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 2, k]).unwrap(),
                f64::sqrt(5.0) * 12.0,
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([3, 2, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 3 => sqrt(7)*(20x^3 - 30x^2 + 12x - 1)
            assert_relative_eq!(
                *data.get([0, 3, k]).unwrap(),
                f64::sqrt(7.0) * (20.0 * x * x * x - 30.0 * x * x + 12.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 3, k]).unwrap(),
                f64::sqrt(7.0) * (60.0 * x * x - 60.0 * x + 12.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 3, k]).unwrap(),
                f64::sqrt(7.0) * (120.0 * x - 60.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([3, 3, k]).unwrap(),
                f64::sqrt(7.0) * 120.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_legendre_quadrilateral_against_known_polynomials() {
        let degree = 2;

        let mut points = rlst_dynamic_array2!(f64, [121, 2]);
        for i in 0..11 {
            for j in 0..11 {
                *points.get_mut([11 * i + j, 0]).unwrap() = i as f64 / 10.0;
                *points.get_mut([11 * i + j, 1]).unwrap() = j as f64 / 10.0;
            }
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Quadrilateral, &points, degree, 1,)
        );
        tabulate_legendre_polynomials(
            ReferenceCellType::Quadrilateral,
            &points,
            degree,
            1,
            &mut data,
        );

        for k in 0..points.shape()[0] {
            let x = *points.get([k, 0]).unwrap();
            let y = *points.get([k, 1]).unwrap();

            // 0 => 1
            assert_relative_eq!(*data.get([0, 0, k]).unwrap(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([1, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([2, 0, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 1 => sqrt(3)*(2x - 1)
            assert_relative_eq!(
                *data.get([0, 1, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 1, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 1, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 2 => sqrt(5)*(6x^2 - 6x + 1)
            assert_relative_eq!(
                *data.get([0, 2, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 2, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * x - 6.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 2, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 3 => sqrt(3)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 3, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );

            assert_relative_eq!(*data.get([1, 3, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(
                *data.get([2, 3, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );

            // 4 => 3*(2x - 1)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 4, k]).unwrap(),
                3.0 * (2.0 * x - 1.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 4, k]).unwrap(),
                6.0 * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 4, k]).unwrap(),
                6.0 * (2.0 * x - 1.0),
                epsilon = 1e-12
            );

            // 5 => sqrt(15)*(6x^2 - 6x + 1)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 5, k]).unwrap(),
                f64::sqrt(15.0) * (6.0 * x * x - 6.0 * x + 1.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 5, k]).unwrap(),
                f64::sqrt(15.0) * (12.0 * x - 6.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 5, k]).unwrap(),
                2.0 * f64::sqrt(15.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );

            // 6 => sqrt(5)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 6, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([1, 6, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(
                *data.get([2, 6, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * y - 6.0),
                epsilon = 1e-12
            );

            // 7 => sqrt(15)*(2x - 1)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 7, k]).unwrap(),
                f64::sqrt(15.0) * (2.0 * x - 1.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 7, k]).unwrap(),
                2.0 * f64::sqrt(15.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 7, k]).unwrap(),
                f64::sqrt(15.0) * (2.0 * x - 1.0) * (12.0 * y - 6.0),
                epsilon = 1e-12
            );

            // 8 => 5*(6x^2 - 6x + 1)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 8, k]).unwrap(),
                5.0 * (6.0 * x * x - 6.0 * x + 1.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 8, k]).unwrap(),
                5.0 * (12.0 * x - 6.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 8, k]).unwrap(),
                5.0 * (12.0 * y - 6.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
        }
    }
}
