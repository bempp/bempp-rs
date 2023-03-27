//! Raviart-Thomas elements

use crate::cell::*;
use crate::element::*;
use crate::map::*;
use bempp_tools::arrays::{Array2D, Array4D};
use bempp_traits::element::ElementFamily;

/// Degree 1 Raviart-Thomas element on a triangle
pub struct RaviartThomasElementTriangleDegree1 {}

impl FiniteElement for RaviartThomasElementTriangleDegree1 {
    fn value_size(&self) -> usize {
        2
    }
    fn map_type(&self) -> MapType {
        MapType::ContravariantPiola
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Triangle
    }
    fn degree(&self) -> usize {
        1
    }
    fn highest_degree(&self) -> usize {
        1
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::RaviartThomas
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        3
    }
    fn tabulate(&self, points: &Array2D<f64>, nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..(nderivs + 1) * (nderivs + 2) / 2 {
            for pt in 0..data.shape().1 {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = -*points.get(pt, 0).unwrap();
                    *data.get_mut(deriv, pt, 0, 1).unwrap() = -*points.get(pt, 1).unwrap();
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = *points.get(pt, 0).unwrap() - 1.0;
                    *data.get_mut(deriv, pt, 1, 1).unwrap() = *points.get(pt, 1).unwrap();
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = -*points.get(pt, 0).unwrap();
                    *data.get_mut(deriv, pt, 2, 1).unwrap() = 1.0 - *points.get(pt, 1).unwrap();
                } else if deriv == 1 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = -1.0;
                    *data.get_mut(deriv, pt, 0, 1).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 1.0;
                    *data.get_mut(deriv, pt, 1, 1).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = -1.0;
                    *data.get_mut(deriv, pt, 2, 1).unwrap() = 0.0;
                } else if deriv == 2 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 0, 1).unwrap() = -1.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 1, 1).unwrap() = 1.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 2, 1).unwrap() = -1.0;
                } else {
                    for fun in 0..3 {
                        *data.get_mut(deriv, pt, fun, 0).unwrap() = 0.;
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 1 {
            vec![entity_number]
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod test {
    use crate::element::raviart_thomas::*;
    use approx::*;

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
                ReferenceCellType::Point => Point {}.entity_count(dim).unwrap(),
                ReferenceCellType::Interval => Interval {}.entity_count(dim).unwrap(),
                ReferenceCellType::Triangle => Triangle {}.entity_count(dim).unwrap(),
                ReferenceCellType::Quadrilateral => Quadrilateral {}.entity_count(dim).unwrap(),
                ReferenceCellType::Tetrahedron => Tetrahedron {}.entity_count(dim).unwrap(),
                ReferenceCellType::Hexahedron => Hexahedron {}.entity_count(dim).unwrap(),
                ReferenceCellType::Prism => Prism {}.entity_count(dim).unwrap(),
                ReferenceCellType::Pyramid => Pyramid {}.entity_count(dim).unwrap(),
            };
            for entity in 0..entity_count {
                ndofs += e.entity_dofs(dim, entity).len();
            }
        }
        assert_eq!(ndofs, e.dim());
    }

    #[test]
    fn test_raviart_thomas_1_triangle() {
        let e = RaviartThomasElementTriangleDegree1 {};
        assert_eq!(e.value_size(), 2);
        let mut data = e.create_tabulate_array(0, 6);
        let points = Array2D::from_data(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
            (6, 2),
        );
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get(0, pt, 0, 0).unwrap(),
                -*points.get(pt, 0).unwrap()
            );
            assert_relative_eq!(
                *data.get(0, pt, 0, 1).unwrap(),
                -*points.get(pt, 1).unwrap()
            );
            assert_relative_eq!(
                *data.get(0, pt, 1, 0).unwrap(),
                *points.get(pt, 0).unwrap() - 1.0
            );
            assert_relative_eq!(*data.get(0, pt, 1, 1).unwrap(), *points.get(pt, 1).unwrap());
            assert_relative_eq!(
                *data.get(0, pt, 2, 0).unwrap(),
                -*points.get(pt, 0).unwrap()
            );
            assert_relative_eq!(
                *data.get(0, pt, 2, 1).unwrap(),
                1.0 - *points.get(pt, 1).unwrap()
            );
        }
        check_dofs(e);
    }
}
