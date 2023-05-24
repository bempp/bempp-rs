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
    for i in 1..data.shape().0 {
        for j in 0..data.shape().2 {
            *data.get_mut(i, 0, j).unwrap() = 0.0;
        }
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

fn tri_index(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

fn quad_index(i: usize, j: usize, n: usize) -> usize {
    (n + 1) * i + j
}

/// Tabulate orthonormal polynomials on a quadrilateral
fn tabulate_legendre_polynomials_quadrilateral<'a>(
    points: &impl Array2DAccess<'a, f64>,
    degree: usize,
    derivatives: usize,
    data: &mut impl Array3DAccess<f64>,
) {
    assert_eq!(data.shape().0, (derivatives + 1) * (derivatives + 2) / 2);
    assert_eq!(data.shape().1, (degree + 1) * (degree + 1));
    assert_eq!(data.shape().2, points.shape().0);
    assert_eq!(points.shape().1, 2);

    for i in 0..data.shape().2 {
        *data
            .get_mut(tri_index(0, 0), quad_index(0, 0, degree), i)
            .unwrap() = 1.0;
    }
    for i0 in 0..derivatives + 1 {
        for i1 in 0..derivatives + 1 - i0 {
            if i0 + i1 > 0 {
                for j in 0..data.shape().2 {
                    *data
                        .get_mut(tri_index(i0, i1), quad_index(0, 0, degree), j)
                        .unwrap() = 0.0;
                }
            }
        }
    }

    if degree == 0 {
        return;
    }

    for i in 0..data.shape().2 {
        *data
            .get_mut(tri_index(0, 0), quad_index(0, 1, degree), i)
            .unwrap() = *points.get(i, 1).unwrap() * 2.0 - 1.0;
    }
    for py in 2..degree + 1 {
        let a = 1.0 - 1.0 / py as f64;
        for i in 0..data.shape().2 {
            *data
                .get_mut(tri_index(0, 0), quad_index(0, py, degree), i)
                .unwrap() = (*points.get(i, 1).unwrap() * 2.0 - 1.0)
                * *data
                    .get(tri_index(0, 0), quad_index(0, py - 1, degree), i)
                    .unwrap()
                * (a + 1.0)
                - *data
                    .get(tri_index(0, 0), quad_index(0, py - 2, degree), i)
                    .unwrap()
                    * a;
        }
    }

    /*

      { // scope
      }
      for (std::size_t ky = 1; ky <= nderiv; ++ky)
      {
        // Get reference to this derivative
        auto result = stdex::submdspan(P, idx(0, ky), stdex::full_extent,
                                       stdex::full_extent);
        auto result0 = stdex::submdspan(P, idx(0, ky - 1), stdex::full_extent,
                                        stdex::full_extent);
        for (std::size_t i = 0; i < result.extent(1); ++i)
        {
          result(quad_idx(0, 1), i)
              = (x1[i] * 2.0 - 1.0) * result(quad_idx(0, 0), i)
                + 2 * ky * result0(quad_idx(0, 0), i);
        }

        for (std::size_t py = 2; py <= n; ++py)
        {
          const T a = 1.0 - 1.0 / static_cast<T>(py);
          for (std::size_t i = 0; i < result.extent(1); ++i)
          {
            result(quad_idx(0, py), i)
                = (x1[i] * 2.0 - 1.0) * result(quad_idx(0, py - 1), i) * (a + 1.0)
                  + 2 * ky * result0(quad_idx(0, py - 1), i) * (a + 1.0)
                  - result(quad_idx(0, py - 2), i) * a;
          }
        }
      }

      // Take tensor product with another interval
      for (std::size_t ky = 0; ky <= nderiv; ++ky)
      {
        auto result = stdex::submdspan(P, idx(0, ky), stdex::full_extent,
                                       stdex::full_extent);
        for (std::size_t py = 0; py <= n; ++py)
        {
          for (std::size_t i = 0; i < result.extent(1); ++i)
          {
            result(quad_idx(1, py), i)
                = (x0[i] * 2.0 - 1.0) * result(quad_idx(0, py), i);
          }
        }
      }

      for (std::size_t px = 2; px <= n; ++px)
      {
        const T a = 1.0 - 1.0 / static_cast<T>(px);
        for (std::size_t ky = 0; ky <= nderiv; ++ky)
        {
          auto result = stdex::submdspan(P, idx(0, ky), stdex::full_extent,
                                         stdex::full_extent);
          for (std::size_t py = 0; py <= n; ++py)
          {
            for (std::size_t i = 0; i < result.extent(1); ++i)
            {
              result(quad_idx(px, py), i) = (x0[i] * 2.0 - 1.0)
                                                * result(quad_idx(px - 1, py), i)
                                                * (a + 1.0)
                                            - result(quad_idx(px - 2, py), i) * a;
            }
          }
        }
      }

      for (std::size_t kx = 1; kx <= nderiv; ++kx)
      {
        for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
        {
          auto result = stdex::submdspan(P, idx(kx, ky), stdex::full_extent,
                                         stdex::full_extent);
          auto result0 = stdex::submdspan(P, idx(kx - 1, ky), stdex::full_extent,
                                          stdex::full_extent);
          for (std::size_t py = 0; py <= n; ++py)
          {
            for (std::size_t i = 0; i < result.extent(1); ++i)
            {
              result(quad_idx(1, py), i)
                  = (x0[i] * 2.0 - 1.0) * result(quad_idx(0, py), i)
                    + 2 * kx * result0(quad_idx(0, py), i);
            }
          }
        }

        for (std::size_t px = 2; px <= n; ++px)
        {
          const T a = 1.0 - 1.0 / static_cast<T>(px);
          for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
          {
            auto result = stdex::submdspan(P, idx(kx, ky), stdex::full_extent,
                                           stdex::full_extent);
            auto result0 = stdex::submdspan(P, idx(kx - 1, ky), stdex::full_extent,
                                            stdex::full_extent);
            for (std::size_t py = 0; py <= n; ++py)
            {
              for (std::size_t i = 0; i < result.extent(1); ++i)
              {
                result(quad_idx(px, py), i)
                    = (x0[i] * 2.0 - 1.0) * result(quad_idx(px - 1, py), i)
                          * (a + 1.0)
                      + 2 * kx * result0(quad_idx(px - 1, py), i) * (a + 1.0)
                      - result(quad_idx(px - 2, py), i) * a;
              }
            }
          }
        }
      }

    */

    // Normalise
    for px in 0..degree + 1 {
        for py in 0..degree + 1 {
            for i in 0..data.shape().0 {
                for j in 0..data.shape().2 {
                    *data.get_mut(i, quad_index(px, py, degree), j).unwrap() *=
                        f64::sqrt((2.0 * px as f64 + 1.0) * (2.0 * py as f64 + 1.0))
                }
            }
        }
    }
}
/// Tabulate orthonormal polynomials on a triangle
fn tabulate_legendre_polynomials_triangle<'a>(
    _points: &impl Array2DAccess<'a, f64>,
    _degree: usize,
    _derivatives: usize,
    _data: &mut impl Array3DAccess<f64>,
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

    #[test]
    fn test_legendre_quadrilataral() {
        let degree = 4;

        let rule = simplex_rule(ReferenceCellType::Quadrilateral, 28).unwrap();
        let points = Array2D::from_data(rule.points, (rule.npoints, 2));

        let mut data = Array3D::<f64>::new((1, (degree + 1) * (degree + 1), rule.npoints));
        tabulate_legendre_polynomials(
            ReferenceCellType::Quadrilateral,
            &points,
            degree,
            0,
            &mut data,
        );

        for i in 0..data.shape().1 {
            for j in 0..data.shape().1 {
                let mut product = 0.0;
                for k in 0..rule.npoints {
                    product +=
                        data.get(0, i, k).unwrap() * data.get(0, j, k).unwrap() * rule.weights[k];
                }
                println!("{i} {j} {product}");
                if i == j {
                    assert_relative_eq!(product, 1.0, epsilon = 1e-12);
                } else {
                    assert_relative_eq!(product, 0.0, epsilon = 1e-12);
                }
            }
        }
    }
}
