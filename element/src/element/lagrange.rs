//! Lagrange elements

use crate::element::OldCiarletElement;
use bempp_tools::arrays::{AdjacencyList, Array3D};
use bempp_traits::arrays::Array3DAccess;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{ElementFamily, MapType};

/// Create a Lagrange element
pub fn create(
    cell_type: ReferenceCellType,
    degree: usize,
    discontinuous: bool,
) -> OldCiarletElement {
    if degree == 0 && !discontinuous {
        panic!("Cannot create continuous degree 0 element");
    }
    let coefficients = match cell_type {
        ReferenceCellType::Interval => match degree {
            // Basis = {1}
            0 => Array3D::from_data(vec![1.0], (1, 1, 1)),
            // Basis = {1 - x, x}
            1 => Array3D::from_data(vec![1.0, -1.0, 0.0, 1.0], (2, 1, 2)),
            _ => {
                panic!("Degree not supported");
            }
        },
        ReferenceCellType::Triangle => match degree {
            // Basis = {1}
            0 => Array3D::from_data(vec![1.0], (1, 1, 1)),
            // Basis = {1-x-y, x, y}
            1 => Array3D::from_data(
                vec![1.0, -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                (3, 1, 3),
            ),
            // Basis = {(1-x-y)(1-2x-2y), x(2x-1), y(2y - 1), 4xy, 4y(1-x-y), 4x(1-x-y)}
            2 => Array3D::from_data(
                vec![
                    1.0, -3.0, -3.0, 2.0, 4.0, 2.0, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0, -4.0, -4.0,
                    0.0, 4.0, 0.0, -4.0, -4.0, 0.0,
                ],
                (6, 1, 6),
            ),
            _ => {
                panic!("Degree not supported");
            }
        },
        ReferenceCellType::Quadrilateral => match degree {
            // Basis = {0}
            0 => Array3D::from_data(vec![1.0], (1, 1, 1)),
            // Basis = {(1-x)(1-y), x(1-y), (1-x)y, xy}
            1 => Array3D::from_data(
                vec![
                    1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
                    1.0,
                ],
                (4, 1, 4),
            ),
            // Basis = {(1-x)*(1-2*x)*(1-y)*(1-2*y), x*(2*x-1)*(1-y)*(1-2*y), (1-x)*(1-2*x)*y*(2*y-1), x*(2*x-1)*y*(2*y-1), 4*x*(1-x)*(1-y)*(1-2*y), (1-x)*(1-2*x)*4*y*(1-y), x*(2*x-1)*4*y*(1-y), 4*x*(1-x)*y*(2*y-1), 4*x*(1-x)*4*y*(1-y)}
            2 => Array3D::from_data(
                vec![
                    1.0, -3.0, 2.0, -3.0, 9.0, -6.0, 2.0, -6.0, 4.0, 0.0, -1.0, 2.0, 0.0, 3.0,
                    -6.0, 0.0, -2.0, 4.0, 0.0, 0.0, 0.0, -1.0, 3.0, -2.0, 2.0, -6.0, 4.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, -2.0, 0.0, -2.0, 4.0, 0.0, 4.0, -4.0, 0.0, -12.0, 12.0, 0.0,
                    8.0, -8.0, 0.0, 0.0, 0.0, 4.0, -12.0, 8.0, -4.0, 12.0, -8.0, 0.0, 0.0, 0.0,
                    0.0, -4.0, 8.0, 0.0, 4.0, -8.0, 0.0, 0.0, 0.0, 0.0, -4.0, 4.0, 0.0, 8.0, -8.0,
                    0.0, 0.0, 0.0, 0.0, 16.0, -16.0, 0.0, -16.0, 16.0,
                ],
                (9, 1, 9),
            ),
            _ => {
                panic!("Degree not supported");
            }
        },
        _ => {
            panic!("Cell type not supported");
        }
    };
    let entity_dofs = if discontinuous {
        let dofs = AdjacencyList::<usize>::from_data(
            (0..coefficients.shape().0).collect(),
            vec![0, coefficients.shape().0],
        );
        match cell_type {
            ReferenceCellType::Interval => [
                AdjacencyList::<usize>::from_data(vec![], vec![0, 0, 0]),
                dofs,
                AdjacencyList::<usize>::new(),
                AdjacencyList::<usize>::new(),
            ],
            ReferenceCellType::Triangle => [
                AdjacencyList::<usize>::from_data(vec![], vec![0, 0, 0, 0]),
                AdjacencyList::<usize>::from_data(vec![], vec![0, 0, 0, 0]),
                dofs,
                AdjacencyList::<usize>::new(),
            ],
            ReferenceCellType::Quadrilateral => [
                AdjacencyList::<usize>::from_data(vec![], vec![0, 0, 0, 0, 0]),
                AdjacencyList::<usize>::from_data(vec![], vec![0, 0, 0, 0, 0]),
                dofs,
                AdjacencyList::<usize>::new(),
            ],
            _ => {
                panic!("Cell type not supported");
            }
        }
    } else {
        match cell_type {
            ReferenceCellType::Interval => match degree {
                1 => [
                    AdjacencyList::<usize>::from_data(vec![0, 1], vec![0, 1, 2]),
                    AdjacencyList::<usize>::from_data(vec![], vec![0, 0]),
                    AdjacencyList::<usize>::new(),
                    AdjacencyList::<usize>::new(),
                ],
                _ => {
                    panic!("Degree not supported");
                }
            },
            ReferenceCellType::Triangle => match degree {
                1 => [
                    AdjacencyList::<usize>::from_data(vec![0, 1, 2], vec![0, 1, 2, 3]),
                    AdjacencyList::<usize>::from_data(vec![], vec![0, 0, 0, 0]),
                    AdjacencyList::<usize>::from_data(vec![], vec![0, 0]),
                    AdjacencyList::<usize>::new(),
                ],
                2 => [
                    AdjacencyList::<usize>::from_data(vec![0, 1, 2], vec![0, 1, 2, 3]),
                    AdjacencyList::<usize>::from_data(vec![3, 4, 5], vec![0, 1, 2, 3]),
                    AdjacencyList::<usize>::from_data(vec![], vec![0, 0]),
                    AdjacencyList::<usize>::new(),
                ],
                _ => {
                    panic!("Degree not supported");
                }
            },
            ReferenceCellType::Quadrilateral => match degree {
                1 => [
                    AdjacencyList::<usize>::from_data(vec![0, 1, 2, 3], vec![0, 1, 2, 3, 4]),
                    AdjacencyList::<usize>::from_data(vec![], vec![0, 0, 0, 0, 0]),
                    AdjacencyList::<usize>::from_data(vec![], vec![0, 0]),
                    AdjacencyList::<usize>::new(),
                ],
                2 => [
                    AdjacencyList::<usize>::from_data(vec![0, 1, 2, 3], vec![0, 1, 2, 3, 4]),
                    AdjacencyList::<usize>::from_data(vec![4, 5, 6, 7], vec![0, 1, 2, 3, 4]),
                    AdjacencyList::<usize>::from_data(vec![8], vec![0, 1]),
                    AdjacencyList::<usize>::new(),
                ],
                _ => {
                    panic!("Degree not supported");
                }
            },
            _ => {
                panic!("Cell type not supported");
            }
        }
    };
    OldCiarletElement {
        cell_type,
        degree,
        highest_degree: degree,
        map_type: MapType::Identity,
        value_size: 1,
        family: ElementFamily::Lagrange,
        discontinuous,
        dim: coefficients.shape().0,
        coefficients,
        entity_dofs,
    }
}

#[cfg(test)]
mod test {
    use crate::cell::*;
    use crate::element::lagrange::*;
    use approx::*;
    use bempp_tools::arrays::{Array2D, Array4D};
    use bempp_traits::arrays::{Array2DAccess, Array4DAccess};
    use bempp_traits::element::FiniteElement;

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
        let e = create(ReferenceCellType::Interval, 0, true);
        assert_eq!(e.value_size(), 1);
        let mut data = Array4D::<f64>::new(e.tabulate_array_shape(0, 4));
        let points = Array2D::from_data(vec![0.0, 0.2, 0.4, 1.0], (4, 1));
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(*data.get(0, pt, 0, 0).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_interval() {
        let e = create(ReferenceCellType::Interval, 1, false);
        assert_eq!(e.value_size(), 1);
        let mut data = Array4D::<f64>::new(e.tabulate_array_shape(0, 4));
        let points = Array2D::from_data(vec![0.0, 0.2, 0.4, 1.0], (4, 1));
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(
                *data.get(0, pt, 0, 0).unwrap(),
                1.0 - *points.get(pt, 0).unwrap()
            );
            assert_relative_eq!(*data.get(0, pt, 1, 0).unwrap(), *points.get(pt, 0).unwrap());
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_triangle() {
        let e = create(ReferenceCellType::Triangle, 0, true);
        assert_eq!(e.value_size(), 1);
        let mut data = Array4D::<f64>::new(e.tabulate_array_shape(0, 6));
        let points = Array2D::from_data(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
            (6, 2),
        );
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get(0, pt, 0, 0).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_triangle() {
        let e = create(ReferenceCellType::Triangle, 1, false);
        assert_eq!(e.value_size(), 1);
        let mut data = Array4D::<f64>::new(e.tabulate_array_shape(0, 6));
        let points = Array2D::from_data(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
            (6, 2),
        );
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get(0, pt, 0, 0).unwrap(),
                1.0 - *points.get(pt, 0).unwrap() - *points.get(pt, 1).unwrap()
            );
            assert_relative_eq!(*data.get(0, pt, 1, 0).unwrap(), *points.get(pt, 0).unwrap());
            assert_relative_eq!(*data.get(0, pt, 2, 0).unwrap(), *points.get(pt, 1).unwrap());
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_quadrilateral() {
        let e = create(ReferenceCellType::Quadrilateral, 0, true);
        assert_eq!(e.value_size(), 1);
        let mut data = Array4D::<f64>::new(e.tabulate_array_shape(0, 6));
        let points = Array2D::from_data(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.3, 0.2],
            (6, 2),
        );
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get(0, pt, 0, 0).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_quadrilateral() {
        let e = create(ReferenceCellType::Quadrilateral, 1, false);
        assert_eq!(e.value_size(), 1);
        let mut data = Array4D::<f64>::new(e.tabulate_array_shape(0, 6));
        let points = Array2D::from_data(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.3, 0.2],
            (6, 2),
        );
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get(0, pt, 0, 0).unwrap(),
                (1.0 - *points.get(pt, 0).unwrap()) * (1.0 - *points.get(pt, 1).unwrap())
            );
            assert_relative_eq!(
                *data.get(0, pt, 1, 0).unwrap(),
                *points.get(pt, 0).unwrap() * (1.0 - *points.get(pt, 1).unwrap())
            );
            assert_relative_eq!(
                *data.get(0, pt, 2, 0).unwrap(),
                (1.0 - *points.get(pt, 0).unwrap()) * *points.get(pt, 1).unwrap()
            );
            assert_relative_eq!(
                *data.get(0, pt, 3, 0).unwrap(),
                *points.get(pt, 0).unwrap() * *points.get(pt, 1).unwrap()
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_2_quadrilateral() {
        let e = create(ReferenceCellType::Quadrilateral, 2, false);
        assert_eq!(e.value_size(), 1);
        let mut data = Array4D::<f64>::new(e.tabulate_array_shape(0, 6));
        let points = Array2D::from_data(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.3, 0.2],
            (6, 2),
        );
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get(pt, 0).unwrap();
            let y = *points.get(pt, 1).unwrap();
            assert_relative_eq!(
                *data.get(0, pt, 0, 0).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * (1.0 - y) * (1.0 - 2.0 * y)
            );
            assert_relative_eq!(
                *data.get(0, pt, 1, 0).unwrap(),
                x * (2.0 * x - 1.0) * (1.0 - y) * (1.0 - 2.0 * y)
            );
            assert_relative_eq!(
                *data.get(0, pt, 2, 0).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * y * (2.0 * y - 1.0)
            );
            assert_relative_eq!(
                *data.get(0, pt, 3, 0).unwrap(),
                x * (2.0 * x - 1.0) * y * (2.0 * y - 1.0)
            );
            assert_relative_eq!(
                *data.get(0, pt, 4, 0).unwrap(),
                4.0 * x * (1.0 - x) * (1.0 - y) * (1.0 - 2.0 * y)
            );
            assert_relative_eq!(
                *data.get(0, pt, 5, 0).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * 4.0 * y * (1.0 - y)
            );
            assert_relative_eq!(
                *data.get(0, pt, 6, 0).unwrap(),
                x * (2.0 * x - 1.0) * 4.0 * y * (1.0 - y)
            );
            assert_relative_eq!(
                *data.get(0, pt, 7, 0).unwrap(),
                4.0 * x * (1.0 - x) * y * (2.0 * y - 1.0)
            );
            assert_relative_eq!(
                *data.get(0, pt, 8, 0).unwrap(),
                4.0 * x * (1.0 - x) * 4.0 * y * (1.0 - y)
            );
        }
        check_dofs(e);
    }
}
