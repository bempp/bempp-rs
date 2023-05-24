//! Orthonormal polynomials
use bempp_traits::arrays::{Array2DAccess, Array3DAccess};
use bempp_traits::cell::ReferenceCellType;

/// Tabulate orthonormal polynomials on a interval
fn tabulate_legendre_polynomials_interval<'a>(
    points: &impl Array2DAccess<'a, f64>,
    degree: usize,
    derivatives: usize,
    data: &mut impl Array3DAccess<f64>,
) {
    assert_eq!(data.shape().0, derivatives + 1);
    assert_eq!(data.shape().1, degree + 1);
    assert_eq!(data.shape().2, points.shape().0);
    assert_eq!(points.shape().1, 1);

    for i in 0..data.shape().2 {
        *data.get_mut(0, 0, i).unwrap() = 1.0;
    }

    if degree == 0 {
        return;
    }

    let root3 = f64::sqrt(3.0);
    for i in 0..data.shape().2 {
        *data.get_mut(0, 1, i).unwrap() = (points.get(i, 0).unwrap() * 2.0 - 1.0) * root3;
    }

    for p in 2..degree + 1 {
        let a = 1.0 - 1.0 / p as f64;
        let b = (a + 1.0) * ((2.0 * p as f64 + 1.0) / (2.0 * p as f64 - 1.0)).sqrt();
        let c = a * ((2.0 * p as f64 + 1.0) / (2.0 * p as f64 - 3.0)).sqrt();
        for i in 0..data.shape().2 {
            *data.get_mut(0, p, i).unwrap() =
                (points.get(i, 0).unwrap() * 2.0 - 1.0) * data.get(0, p - 1, i).unwrap() * b
                    - data.get(0, p - 2, i).unwrap() * c;
        }
    }

    for k in 1..derivatives + 1 {
        for i in 0..data.shape().2 {
            *data.get_mut(k, 1, i).unwrap() = ((points.get(i, 0).unwrap() * 2.0 - 1.0)
                * data.get(k, 0, i).unwrap()
                + 2.0 * k as f64 * data.get(k - 1, 0, i).unwrap())
                * root3;
        }
        for p in 2..degree + 1 {
            let a = 1.0 - 1.0 / p as f64;
            let b = (a + 1.0) * ((2.0 * p as f64 + 1.0) / (2.0 * p as f64 - 1.0)).sqrt();
            let c = a * ((2.0 * p as f64 + 1.0) / (2.0 * p as f64 - 3.0)).sqrt();
            for i in 0..data.shape().2 {
                *data.get_mut(k, p, i).unwrap() =
                    (points.get(i, 0).unwrap() * 2.0 - 1.0) * data.get(k, p - 1, i).unwrap() * b
                        + 2.0 * k as f64 * data.get(k - 1, p - 1, i).unwrap() * b
                        - data.get(k, p - 2, i).unwrap() * c;
            }
        }
    }
}
/// Tabulate orthonormal polynomials on a quadrilateral
fn tabulate_legendre_polynomials_quadrilateral<'a>(
    points: &impl Array2DAccess<'a, f64>,
    degree: usize,
    derivatives: usize,
    data: &mut impl Array3DAccess<f64>,
) {
}
/// Tabulate orthonormal polynomials on a triangle
fn tabulate_legendre_polynomials_triangle<'a>(
    points: &impl Array2DAccess<'a, f64>,
    degree: usize,
    derivatives: usize,
    data: &mut impl Array3DAccess<f64>,
) {
}

/// Tabulate orthonormal polynomials
pub fn tabulate_legendre_polynomials<'a>(
    cell_type: ReferenceCellType,
    points: &impl Array2DAccess<'a, f64>,
    degree: usize,
    derivatives: usize,
    data: &mut impl Array3DAccess<f64>,
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
    use crate::polynomials::*;
    use approx::*;
    use bempp_quadrature::simplex_rules::simplex_rule;
    use bempp_tools::arrays::{Array2D, Array3D};

    #[test]
    fn test_legendre_interval() {
        let degree = 6;

        let rule = simplex_rule(ReferenceCellType::Interval, degree + 1).unwrap();
        let points = Array2D::from_data(rule.points, (rule.npoints, 1));

        let mut data = Array3D::<f64>::new((1, degree + 1, rule.npoints));
        tabulate_legendre_polynomials(ReferenceCellType::Interval, &points, degree, 0, &mut data);

        for i in 0..degree + 1 {
            for j in 0..degree + 1 {
                let mut product = 0.0;
                for k in 0..rule.npoints {
                    product +=
                        data.get(0, i, k).unwrap() * data.get(0, j, k).unwrap() * rule.weights[k];
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
    fn test_legendre_interval_against_known_polynomials() {
        let degree = 3;

        let points = Array2D::from_data(
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            (11, 1),
        );

        let mut data = Array3D::<f64>::new((4, degree + 1, points.shape().0));
        tabulate_legendre_polynomials(ReferenceCellType::Interval, &points, degree, 3, &mut data);

        for k in 0..points.shape().0 {
            let x = *points.get(k, 0).unwrap();
            assert_relative_eq!(*data.get(0, 0, k).unwrap(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get(1, 0, k).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get(2, 0, k).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get(3, 0, k).unwrap(), 0.0, epsilon = 1e-12);

            assert_relative_eq!(
                *data.get(0, 1, k).unwrap(),
                f64::sqrt(3.0) * (2.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get(1, 1, k).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get(2, 1, k).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get(3, 1, k).unwrap(), 0.0, epsilon = 1e-12);

            assert_relative_eq!(
                *data.get(0, 2, k).unwrap(),
                f64::sqrt(5.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get(1, 2, k).unwrap(),
                f64::sqrt(5.0) * (12.0 * x - 6.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get(2, 2, k).unwrap(),
                f64::sqrt(5.0) * 12.0,
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get(3, 2, k).unwrap(), 0.0, epsilon = 1e-12);

            assert_relative_eq!(
                *data.get(0, 3, k).unwrap(),
                f64::sqrt(7.0) * (20.0 * x * x * x - 30.0 * x * x + 12.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get(1, 3, k).unwrap(),
                f64::sqrt(7.0) * (60.0 * x * x - 60.0 * x + 12.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get(2, 3, k).unwrap(),
                f64::sqrt(7.0) * (120.0 * x - 60.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get(3, 3, k).unwrap(),
                f64::sqrt(7.0) * 120.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_legendre_interval_derivative() {
        let degree = 6;

        let rule = simplex_rule(ReferenceCellType::Interval, degree + 1).unwrap();
        let epsilon = 1e-10;
        let points = Array2D::from_data(
            vec![
                0.0,
                0.0 + epsilon,
                0.1,
                0.1 + epsilon,
                0.2,
                0.2 + epsilon,
                0.3,
                0.3 + epsilon,
                0.4,
                0.4 + epsilon,
                0.5,
                0.5 + epsilon,
                0.6,
                0.6 + epsilon,
                0.7,
                0.7 + epsilon,
                0.8,
                0.8 + epsilon,
                0.9,
                0.9 + epsilon,
            ],
            (20, 1),
        );

        let mut data = Array3D::<f64>::new((2, degree + 1, points.shape().0));
        tabulate_legendre_polynomials(ReferenceCellType::Interval, &points, degree, 1, &mut data);

        for i in 0..degree + 1 {
            for k in 0..points.shape().0 / 2 {
                assert_relative_eq!(
                    *data.get(1, i, 2 * k).unwrap(),
                    (data.get(0, i, 2 * k + 1).unwrap() - data.get(0, i, 2 * k).unwrap()) / epsilon,
                    epsilon = 1e-5
                );
            }
        }
    }
}
