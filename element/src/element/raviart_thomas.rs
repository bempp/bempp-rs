//! Raviart-Thomas elements

use crate::element::OldCiarletElement;
use bempp_tools::arrays::{AdjacencyList, Array3D};
use bempp_traits::arrays::Array3DAccess;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily, MapType};

/// Create a Raviart-Thomas element
pub fn create(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> OldCiarletElement {
    let coefficients = match cell_type {
        ReferenceCellType::Triangle => match degree {
            // Basis = {(-x, -y), (x-1,y), (-x, 1-y)}
            1 => Array3D::from_data(
                vec![
                    0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0,
                    1.0, 0.0, -1.0,
                ],
                (3, 2, 3),
            ),
            _ => {
                panic!("Degree not supported");
            }
        },
        _ => {
            panic!("Cell type not supported");
        }
    };
    let entity_dofs = if continuity == Continuity::Discontinuous {
        let dofs = AdjacencyList::<usize>::from_data(
            (0..coefficients.shape().0).collect(),
            vec![0, coefficients.shape().0],
        );
        match cell_type {
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
            ReferenceCellType::Triangle => match degree {
                1 => [
                    AdjacencyList::<usize>::from_data(vec![], vec![0, 0, 0, 0]),
                    AdjacencyList::<usize>::from_data(vec![0, 1, 2], vec![0, 1, 2, 3]),
                    AdjacencyList::<usize>::from_data(vec![], vec![0, 0]),
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
        map_type: MapType::ContravariantPiola,
        value_size: 2,
        family: ElementFamily::RaviartThomas,
        continuity,
        dim: coefficients.shape().0,
        coefficients,
        entity_dofs,
    }
}

#[cfg(test)]
mod test {
    use crate::cell::*;
    use crate::element::raviart_thomas::*;
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
    fn test_raviart_thomas_1_triangle() {
        let e = create(ReferenceCellType::Triangle, 1, Continuity::Continuous);
        assert_eq!(e.value_size(), 2);
        let mut data = Array4D::<f64>::new(e.tabulate_array_shape(0, 6));
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
