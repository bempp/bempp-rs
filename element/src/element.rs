//! Finite Element definitions

use bempp_tools::arrays::{AdjacencyList, Array3D};
use bempp_traits::arrays::{AdjacencyListAccess, Array2DAccess, Array3DAccess, Array4DAccess};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{ElementFamily, FiniteElement, MapType};
pub mod lagrange;
pub mod raviart_thomas;

pub struct CiarletElement {
    cell_type: ReferenceCellType,
    degree: usize,
    highest_degree: usize,
    map_type: MapType,
    value_size: usize,
    family: ElementFamily,
    discontinuous: bool,
    dim: usize,
    coefficients: Array3D<f64>,
    entity_dofs: [AdjacencyList<usize>; 4],
}

impl FiniteElement for CiarletElement {
    fn value_size(&self) -> usize {
        self.value_size
    }
    fn map_type(&self) -> MapType {
        self.map_type
    }

    fn cell_type(&self) -> ReferenceCellType {
        self.cell_type
    }
    fn degree(&self) -> usize {
        self.degree
    }
    fn highest_degree(&self) -> usize {
        self.highest_degree
    }
    fn family(&self) -> ElementFamily {
        self.family
    }
    fn discontinuous(&self) -> bool {
        self.discontinuous
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn tabulate<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        nderivs: usize,
        data: &mut impl Array4DAccess<f64>,
    ) {
        for deriv in 0..nderivs + 1 {
            for pt in 0..data.shape().1 {
                let evals = match self.cell_type {
                    ReferenceCellType::Interval => {
                        let x = *points.get(pt, 0).unwrap();
                        match self.highest_degree {
                            0 => match deriv {
                                0 => vec![1.0],
                                _ => vec![0.0],
                            },
                            1 => match deriv {
                                0 => vec![1.0, x],
                                1 => vec![0.0, 1.0],
                                _ => vec![0.0, 0.0, 0.0],
                            },
                            2 => match deriv {
                                0 => vec![1.0, x, x * x],
                                1 => vec![0.0, 1.0, 2.0 * x],
                                2 => vec![0.0, 0.0, 2.0],
                                _ => vec![0.0, 0.0, 0.0],
                            },
                            3 => match deriv {
                                0 => vec![1.0, x, x * x, x * x * x],
                                1 => vec![0.0, 1.0, 2.0 * x, 3.0 * x * x],
                                2 => vec![0.0, 0.0, 2.0, 6.0 * x],
                                3 => vec![0.0, 0.0, 0.0, 6.0],
                                _ => vec![0.0, 0.0, 0.0, 0.0],
                            },
                            _ => {
                                panic!("Degree currently unsupported");
                            }
                        }
                    }
                    ReferenceCellType::Triangle => {
                        let x = *points.get(pt, 0).unwrap();
                        let y = *points.get(pt, 1).unwrap();
                        match self.highest_degree {
                            0 => match deriv {
                                0 => vec![1.0],
                                _ => vec![0.0],
                            },
                            1 => match deriv {
                                0 => vec![1.0, x, y],
                                1 => vec![0.0, 1.0, 0.0],
                                2 => vec![0.0, 0.0, 1.0],
                                _ => vec![0.0, 0.0, 0.0],
                            },
                            2 => match deriv {
                                0 => vec![1.0, x, y, x * x, x * y, y * y],
                                1 => vec![0.0, 1.0, 0.0, 2.0 * x, y, 0.0],
                                2 => vec![0.0, 0.0, 1.0, 0.0, x, 2.0 * y],
                                3 => vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                                4 => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                5 => vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                                _ => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            },
                            _ => {
                                panic!("Degree currently unsupported");
                            }
                        }
                    }
                    ReferenceCellType::Quadrilateral => {
                        let x = *points.get(pt, 0).unwrap();
                        let y = *points.get(pt, 1).unwrap();
                        match self.highest_degree {
                            0 => match deriv {
                                0 => vec![1.0],
                                _ => vec![0.0],
                            },
                            1 => match deriv {
                                0 => vec![1.0, x, y, x * y],
                                1 => vec![0.0, 1.0, 0.0, y],
                                2 => vec![0.0, 0.0, 1.0, x],
                                4 => vec![0.0, 0.0, 0.0, 1.0],
                                _ => vec![0.0, 0.0, 0.0, 0.0],
                            },
                            2 => match deriv {
                                0 => vec![
                                    1.0,
                                    x,
                                    x * x,
                                    y,
                                    x * y,
                                    x * x * y,
                                    y * y,
                                    x * y * y,
                                    x * x * y * y,
                                ],
                                1 => vec![
                                    1.0,
                                    1.0,
                                    2.0 * x,
                                    0.0,
                                    y,
                                    2.0 * x * y,
                                    0.0,
                                    y * y,
                                    2.0 * x * y * y,
                                ],
                                2 => vec![
                                    1.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                    x,
                                    x * x,
                                    2.0 * y,
                                    2.0 * x * y,
                                    2.0 * x * x * y,
                                ],
                                3 => vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0 * y, 0.0, 0.0, 2.0 * y * y],
                                4 => vec![
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                    2.0 * x,
                                    0.0,
                                    2.0 * y,
                                    4.0 * x * y,
                                ],
                                5 => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0 * x, 2.0 * x * x],
                                7 => vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0 * y],
                                8 => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0 * x],
                                12 => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
                                _ => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            },
                            _ => {
                                panic!("Degree currently unsupported");
                            }
                        }
                    }
                    _ => {
                        panic!("Tabulation not implemented for this cell type");
                    }
                };

                for i in 0..self.coefficients.shape().0 {
                    for j in 0..self.coefficients.shape().1 {
                        *data.get_mut(deriv, pt, i, j).unwrap() = 0.0;
                        for k in 0..self.coefficients.shape().2 {
                            *data.get_mut(deriv, pt, i, j).unwrap() +=
                                unsafe { *self.coefficients.get_unchecked(i, j, k) } * evals[k];
                        }
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]> {
        self.entity_dofs[entity_dim].row(entity_number)
    }
}

pub fn create_element(
    family: ElementFamily,
    cell_type: ReferenceCellType,
    degree: usize,
    discontinuous: bool,
) -> CiarletElement {
    match family {
        ElementFamily::Lagrange => lagrange::create(cell_type, degree, discontinuous),
        ElementFamily::RaviartThomas => raviart_thomas::create(cell_type, degree, discontinuous),
    }
}

#[cfg(test)]
mod test {
    use crate::element::*;
    use bempp_traits::cell::ReferenceCellType;

    #[test]
    fn test_lagrange_1() {
        let e = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            false,
        );
        assert_eq!(e.value_size(), 1);
    }
}
