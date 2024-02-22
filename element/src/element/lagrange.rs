//! Lagrange elements

use crate::element::{create_cell, CiarletElement};
use crate::polynomials::polynomial_count;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily, MapType};
use rlst_dense::{rlst_dynamic_array2, rlst_dynamic_array3, traits::RandomAccessMut};

/// Create a Lagrange element
pub fn create(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement {
    let cell = create_cell(cell_type);
    let dim = polynomial_count(cell_type, degree);
    let tdim = cell.dim();
    let mut wcoeffs = rlst_dynamic_array3!(f64, [dim, 1, dim]);
    for i in 0..dim {
        *wcoeffs.get_mut([i, 0, i]).unwrap() = 1.0;
    }

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];
    if degree == 0 {
        if continuity == Continuity::Continuous {
            panic!("Cannot create continuous degree 0 Lagrange element");
        }
        for d in 0..tdim {
            for _e in 0..cell.entity_count(d) {
                x[d].push(rlst_dynamic_array2!(f64, [0, tdim]));
                m[d].push(rlst_dynamic_array3!(f64, [0, 1, 0]));
            }
        }
        let mut midp = rlst_dynamic_array2!(f64, [1, tdim]);
        let nvertices = cell.vertex_count();
        let vertices = cell.vertices();
        for i in 0..tdim {
            for j in 0..nvertices {
                *midp.get_mut([0, i]).unwrap() += vertices[j * tdim + i];
            }
            *midp.get_mut([0, i]).unwrap() /= nvertices as f64;
        }
        x[tdim].push(midp);
        let mut mentry = rlst_dynamic_array3!(f64, [1, 1, 1]);
        *mentry.get_mut([0, 0, 0]).unwrap() = 1.0;
        m[tdim].push(mentry);
    } else {
        // TODO: GLL points
        for e in 0..cell.entity_count(0) {
            let mut pts = rlst_dynamic_array2!(f64, [1, tdim]);
            for i in 0..tdim {
                *pts.get_mut([0, i]).unwrap() = cell.vertices()[e * tdim + i];
            }
            x[0].push(pts);
            let mut mentry = rlst_dynamic_array3!(f64, [1, 1, 1]);
            *mentry.get_mut([0, 0, 0]).unwrap() = 1.0;
            m[0].push(mentry);
        }
        for e in 0..cell.entity_count(1) {
            let mut pts = rlst_dynamic_array2!(f64, [degree - 1, tdim]);
            let vn0 = cell.edges()[2 * e];
            let vn1 = cell.edges()[2 * e + 1];
            let v0 = &cell.vertices()[vn0 * tdim..(vn0 + 1) * tdim];
            let v1 = &cell.vertices()[vn1 * tdim..(vn1 + 1) * tdim];
            let mut ident = rlst_dynamic_array3!(f64, [degree - 1, 1, degree - 1]);

            for i in 1..degree {
                *ident.get_mut([i - 1, 0, i - 1]).unwrap() = 1.0;
                for j in 0..tdim {
                    *pts.get_mut([i - 1, j]).unwrap() =
                        v0[j] + i as f64 / degree as f64 * (v1[j] - v0[j]);
                }
            }
            x[1].push(pts);
            m[1].push(ident);
        }
        let mut start = 0;
        for e in 0..cell.entity_count(2) {
            let nvertices = cell.faces_nvertices()[e];
            let npts = if nvertices == 3 {
                if degree > 2 {
                    (degree - 1) * (degree - 2) / 2
                } else {
                    0
                }
            } else if nvertices == 4 {
                (degree - 1).pow(2)
            } else {
                panic!("Unsupported face type");
            };
            let mut pts = rlst_dynamic_array2!(f64, [npts, tdim]);

            let vn0 = cell.faces()[start];
            let vn1 = cell.faces()[start + 1];
            let vn2 = cell.faces()[start + 2];
            let v0 = &cell.vertices()[vn0 * tdim..(vn0 + 1) * tdim];
            let v1 = &cell.vertices()[vn1 * tdim..(vn1 + 1) * tdim];
            let v2 = &cell.vertices()[vn2 * tdim..(vn2 + 1) * tdim];

            if nvertices == 3 {
                // Triangle
                let mut n = 0;
                for i0 in 1..degree {
                    for i1 in 1..degree - i0 {
                        for j in 0..tdim {
                            *pts.get_mut([n, j]).unwrap() = v0[j]
                                + i0 as f64 / degree as f64 * (v1[j] - v0[j])
                                + i1 as f64 / degree as f64 * (v2[j] - v0[j]);
                        }
                        n += 1;
                    }
                }
            } else if nvertices == 4 {
                // Quadrilateral
                let mut n = 0;
                for i0 in 1..degree {
                    for i1 in 1..degree {
                        for j in 0..tdim {
                            *pts.get_mut([n, j]).unwrap() = v0[j]
                                + i0 as f64 / degree as f64 * (v1[j] - v0[j])
                                + i1 as f64 / degree as f64 * (v2[j] - v0[j]);
                        }
                        n += 1;
                    }
                }
            } else {
                panic!("Unsupported face type.");
            }

            let mut ident = rlst_dynamic_array3!(f64, [npts, 1, npts]);
            for i in 0..npts {
                *ident.get_mut([i, 0, i]).unwrap() = 1.0;
            }
            x[2].push(pts);
            m[2].push(ident);
            start += nvertices;
        }
    }
    CiarletElement::create(
        ElementFamily::Lagrange,
        cell_type,
        degree,
        vec![],
        wcoeffs,
        x,
        m,
        MapType::Identity,
        continuity,
        degree,
    )
}

#[cfg(test)]
mod test {
    use crate::cell::*;
    use crate::element::lagrange::*;
    use approx::*;
    use bempp_traits::element::FiniteElement;
    use rlst_dense::rlst_dynamic_array4;
    use rlst_dense::traits::RandomAccessByRef;

    fn check_dofs(e: impl FiniteElement) {
        let cell_dim = match e.cell_type() {
            ReferenceCellType::Point => Point {}.dim(),
            ReferenceCellType::Interval => Interval {}.dim(),
            ReferenceCellType::Triangle => Triangle {}.dim(),
            ReferenceCellType::Quadrilateral => Quadrilateral {}.dim(),
            ReferenceCellType::Tetrahedron => Tetrahedron {}.dim(),
            ReferenceCellType::Hexahedron => Hexahedron {}.dim(),
            ReferenceCellType::Prism => Prism {}.dim(),
            ReferenceCellType::Pyramid => Pyramid {}.dim(),
        };
        let mut ndofs = 0;
        for dim in 0..cell_dim + 1 {
            let entity_count = match e.cell_type() {
                ReferenceCellType::Point => Point {}.entity_count(dim),
                ReferenceCellType::Interval => Interval {}.entity_count(dim),
                ReferenceCellType::Triangle => Triangle {}.entity_count(dim),
                ReferenceCellType::Quadrilateral => Quadrilateral {}.entity_count(dim),
                ReferenceCellType::Tetrahedron => Tetrahedron {}.entity_count(dim),
                ReferenceCellType::Hexahedron => Hexahedron {}.entity_count(dim),
                ReferenceCellType::Prism => Prism {}.entity_count(dim),
                ReferenceCellType::Pyramid => Pyramid {}.entity_count(dim),
            };
            for entity in 0..entity_count {
                ndofs += e.entity_dofs(dim, entity).unwrap().len();
            }
        }
        assert_eq!(ndofs, e.dim());
    }

    #[test]
    fn test_lagrange_0_interval() {
        let e = create(ReferenceCellType::Interval, 0, Continuity::Discontinuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 4));
        let mut points = rlst_dynamic_array2!(f64, [4, 1]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.2;
        *points.get_mut([2, 0]).unwrap() = 0.4;
        *points.get_mut([3, 0]).unwrap() = 1.0;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_interval() {
        let e = create(ReferenceCellType::Interval, 1, Continuity::Continuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 4));
        let mut points = rlst_dynamic_array2!(f64, [4, 1]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.2;
        *points.get_mut([2, 0]).unwrap() = 0.4;
        *points.get_mut([3, 0]).unwrap() = 1.0;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                1.0 - *points.get([pt, 0]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([pt, 0]).unwrap()
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_triangle() {
        let e = create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));

        let mut points = rlst_dynamic_array2!(f64, [6, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 1.0;
        *points.get_mut([3, 0]).unwrap() = 0.5;
        *points.get_mut([3, 1]).unwrap() = 0.0;
        *points.get_mut([4, 0]).unwrap() = 0.0;
        *points.get_mut([4, 1]).unwrap() = 0.5;
        *points.get_mut([5, 0]).unwrap() = 0.5;
        *points.get_mut([5, 1]).unwrap() = 0.5;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_triangle() {
        let e = create(ReferenceCellType::Triangle, 1, Continuity::Continuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [6, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 1.0;
        *points.get_mut([3, 0]).unwrap() = 0.5;
        *points.get_mut([3, 1]).unwrap() = 0.0;
        *points.get_mut([4, 0]).unwrap() = 0.0;
        *points.get_mut([4, 1]).unwrap() = 0.5;
        *points.get_mut([5, 0]).unwrap() = 0.5;
        *points.get_mut([5, 1]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                1.0 - *points.get([pt, 0]).unwrap() - *points.get([pt, 1]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([pt, 0]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                *points.get([pt, 1]).unwrap()
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_higher_degree_triangle() {
        create(ReferenceCellType::Triangle, 2, Continuity::Continuous);
        create(ReferenceCellType::Triangle, 3, Continuity::Continuous);
        create(ReferenceCellType::Triangle, 4, Continuity::Continuous);
        create(ReferenceCellType::Triangle, 5, Continuity::Continuous);

        create(ReferenceCellType::Triangle, 2, Continuity::Discontinuous);
        create(ReferenceCellType::Triangle, 3, Continuity::Discontinuous);
        create(ReferenceCellType::Triangle, 4, Continuity::Discontinuous);
        create(ReferenceCellType::Triangle, 5, Continuity::Discontinuous);
    }

    #[test]
    fn test_lagrange_higher_degree_interval() {
        create(ReferenceCellType::Interval, 2, Continuity::Continuous);
        create(ReferenceCellType::Interval, 3, Continuity::Continuous);
        create(ReferenceCellType::Interval, 4, Continuity::Continuous);
        create(ReferenceCellType::Interval, 5, Continuity::Continuous);

        create(ReferenceCellType::Interval, 2, Continuity::Discontinuous);
        create(ReferenceCellType::Interval, 3, Continuity::Discontinuous);
        create(ReferenceCellType::Interval, 4, Continuity::Discontinuous);
        create(ReferenceCellType::Interval, 5, Continuity::Discontinuous);
    }

    #[test]
    fn test_lagrange_higher_degree_quadrilateral() {
        create(ReferenceCellType::Quadrilateral, 2, Continuity::Continuous);
        create(ReferenceCellType::Quadrilateral, 3, Continuity::Continuous);
        create(ReferenceCellType::Quadrilateral, 4, Continuity::Continuous);
        create(ReferenceCellType::Quadrilateral, 5, Continuity::Continuous);

        create(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Discontinuous,
        );
        create(
            ReferenceCellType::Quadrilateral,
            3,
            Continuity::Discontinuous,
        );
        create(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Discontinuous,
        );
        create(
            ReferenceCellType::Quadrilateral,
            5,
            Continuity::Discontinuous,
        );
    }

    #[test]
    fn test_lagrange_0_quadrilateral() {
        let e = create(
            ReferenceCellType::Quadrilateral,
            0,
            Continuity::Discontinuous,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [6, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 1.0;
        *points.get_mut([3, 0]).unwrap() = 0.5;
        *points.get_mut([3, 1]).unwrap() = 0.0;
        *points.get_mut([4, 0]).unwrap() = 0.0;
        *points.get_mut([4, 1]).unwrap() = 0.5;
        *points.get_mut([5, 0]).unwrap() = 0.5;
        *points.get_mut([5, 1]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_quadrilateral() {
        let e = create(ReferenceCellType::Quadrilateral, 1, Continuity::Continuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [6, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 1.0;
        *points.get_mut([3, 0]).unwrap() = 1.0;
        *points.get_mut([3, 1]).unwrap() = 1.0;
        *points.get_mut([4, 0]).unwrap() = 0.25;
        *points.get_mut([4, 1]).unwrap() = 0.5;
        *points.get_mut([5, 0]).unwrap() = 0.3;
        *points.get_mut([5, 1]).unwrap() = 0.2;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                (1.0 - *points.get([pt, 0]).unwrap()) * (1.0 - *points.get([pt, 1]).unwrap())
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([pt, 0]).unwrap() * (1.0 - *points.get([pt, 1]).unwrap())
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                (1.0 - *points.get([pt, 0]).unwrap()) * *points.get([pt, 1]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                *points.get([pt, 0]).unwrap() * *points.get([pt, 1]).unwrap()
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_2_quadrilateral() {
        let e = create(ReferenceCellType::Quadrilateral, 2, Continuity::Continuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [6, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 1.0;
        *points.get_mut([3, 0]).unwrap() = 1.0;
        *points.get_mut([3, 1]).unwrap() = 1.0;
        *points.get_mut([4, 0]).unwrap() = 0.25;
        *points.get_mut([4, 1]).unwrap() = 0.5;
        *points.get_mut([5, 0]).unwrap() = 0.3;
        *points.get_mut([5, 1]).unwrap() = 0.2;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([pt, 0]).unwrap();
            let y = *points.get([pt, 1]).unwrap();
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                x * (2.0 * x - 1.0) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                x * (2.0 * x - 1.0) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 4, 0]).unwrap(),
                4.0 * x * (1.0 - x) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 5, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 6, 0]).unwrap(),
                x * (2.0 * x - 1.0) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 7, 0]).unwrap(),
                4.0 * x * (1.0 - x) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 8, 0]).unwrap(),
                4.0 * x * (1.0 - x) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
        }
        check_dofs(e);
    }
}
