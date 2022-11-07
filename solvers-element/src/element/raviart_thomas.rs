//! Lagrange elements

use crate::element::*;

/// Degree 1 Raviart-Thomas element on a triangle
pub struct RaviartThomasElementTriangleDegree1 {}

impl FiniteElement for RaviartThomasElementTriangleDegree1 {
    const VALUE_SIZE: usize = 2;
    const MAP_TYPE: MapType = MapType::ContravariantPiola;

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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..data.deriv_count() {
            for pt in 0..data.point_count() {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0) = -points[2 * pt];
                    *data.get_mut(deriv, pt, 0, 1) = -points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 1, 0) = points[2 * pt] - 1.0;
                    *data.get_mut(deriv, pt, 1, 1) = points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 2, 0) = -points[2 * pt];
                    *data.get_mut(deriv, pt, 2, 1) = 1.0 - points[2 * pt + 1];
                } else if deriv == 1 {
                    *data.get_mut(deriv, pt, 0, 0) = -1.0;
                    *data.get_mut(deriv, pt, 0, 1) = 0.0;
                    *data.get_mut(deriv, pt, 1, 0) = 1.0;
                    *data.get_mut(deriv, pt, 1, 1) = 0.0;
                    *data.get_mut(deriv, pt, 2, 0) = -1.0;
                    *data.get_mut(deriv, pt, 2, 1) = 0.0;
                } else if deriv == 2 {
                    *data.get_mut(deriv, pt, 0, 0) = 0.0;
                    *data.get_mut(deriv, pt, 0, 1) = -1.0;
                    *data.get_mut(deriv, pt, 1, 0) = 0.0;
                    *data.get_mut(deriv, pt, 1, 1) = 1.0;
                    *data.get_mut(deriv, pt, 2, 0) = 0.0;
                    *data.get_mut(deriv, pt, 2, 1) = -1.0;
                } else {
                    for fun in 0..3 {
                        *data.get_mut(deriv, pt, fun, 0) = 0.;
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
    use crate::element::*;
    use approx::*;

    fn check_dofs(e: impl FiniteElement) {
        let cell_dim = match e.cell_type() {
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
        let mut data = TabulatedData::new(&e, 0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get(0, pt, 0, 0), -points[2 * pt]);
            assert_relative_eq!(*data.get(0, pt, 0, 1), -points[2 * pt + 1]);
            assert_relative_eq!(*data.get(0, pt, 1, 0), points[2 * pt] - 1.0);
            assert_relative_eq!(*data.get(0, pt, 1, 1), points[2 * pt + 1]);
            assert_relative_eq!(*data.get(0, pt, 2, 0), -points[2 * pt]);
            assert_relative_eq!(*data.get(0, pt, 2, 1), 1.0 - points[2 * pt + 1]);
        }
        check_dofs(e);
    }
}
