//! Lagrange elements

use crate::cell::*;
use crate::element::*;
use crate::map::*;
use bempp_tools::arrays::Array4D;

/// Lagrange element
pub struct LagrangeElement {
    celltype: ReferenceCellType,
    degree: usize,
}

impl LagrangeElement {
    pub fn new(celltype: ReferenceCellType, degree: usize) -> Self {
        Self {
            celltype: celltype,
            degree: degree,
        }
    }
}

impl FiniteElement for LagrangeElement {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
    }

    fn cell_type(&self) -> ReferenceCellType {
        self.celltype
    }
    fn degree(&self) -> usize {
        self.degree
    }
    fn highest_degree(&self) -> usize {
        self.degree
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        unimplemented!("dim not yet implemented for this element");
    }
    fn tabulate(&self, _points: &[f64], _nderivs: usize, _data: &mut Array4D<f64>) {
        unimplemented!("tabulate not yet implemented for this element");
    }
    fn entity_dofs(&self, _entity_dim: usize, _entity_number: usize) -> Vec<usize> {
        unimplemented!("entity_dofs not yet implemented for this element");
    }
}

/// Degree 0 Lagrange element on an interval
pub struct LagrangeElementIntervalDegree0 {}

impl FiniteElement for LagrangeElementIntervalDegree0 {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Interval
    }
    fn degree(&self) -> usize {
        0
    }
    fn highest_degree(&self) -> usize {
        0
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        1
    }
    fn tabulate(&self, _points: &[f64], nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..nderivs + 1 {
            for pt in 0..data.shape().1 {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 1.0;
                } else {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 0.;
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 1 && entity_number == 0 {
            vec![0]
        } else {
            vec![]
        }
    }
}

/// Degree 1 Lagrange element on an interval
pub struct LagrangeElementIntervalDegree1 {}

impl FiniteElement for LagrangeElementIntervalDegree1 {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Interval
    }
    fn degree(&self) -> usize {
        1
    }
    fn highest_degree(&self) -> usize {
        1
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        2
    }
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..(nderivs + 1) * (nderivs + 2) / 2 {
            for pt in 0..data.shape().1 {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 1.0 - points[pt];
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = points[pt];
                } else if deriv == 1 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = -1.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 1.0;
                } else {
                    for fun in 0..2 {
                        *data.get_mut(deriv, pt, fun, 0).unwrap() = 0.;
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 0 {
            vec![entity_number]
        } else {
            vec![]
        }
    }
}

/// Degree 0 Lagrange element on a triangle
pub struct LagrangeElementTriangleDegree0 {}

impl FiniteElement for LagrangeElementTriangleDegree0 {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Triangle
    }
    fn degree(&self) -> usize {
        0
    }
    fn highest_degree(&self) -> usize {
        0
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        1
    }
    fn tabulate(&self, _points: &[f64], nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..(nderivs + 1) * (nderivs + 2) / 2 {
            for pt in 0..data.shape().1 {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 1.0;
                } else {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 0.;
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 2 && entity_number == 0 {
            vec![0]
        } else {
            vec![]
        }
    }
}

/// Degree 1 Lagrange element on a triangle
pub struct LagrangeElementTriangleDegree1 {}

impl FiniteElement for LagrangeElementTriangleDegree1 {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
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
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        3
    }
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..(nderivs + 1) * (nderivs + 2) / 2 {
            for pt in 0..data.shape().1 {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() =
                        1.0 - points[2 * pt] - points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = points[2 * pt];
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = points[2 * pt + 1];
                } else if deriv == 1 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = -1.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 1.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 0.0;
                } else if deriv == 2 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = -1.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 1.0;
                } else {
                    for fun in 0..3 {
                        *data.get_mut(deriv, pt, fun, 0).unwrap() = 0.;
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 0 {
            vec![entity_number]
        } else {
            vec![]
        }
    }
}

/// Degree 2 Lagrange element on a triangle
pub struct LagrangeElementTriangleDegree2 {}

impl FiniteElement for LagrangeElementTriangleDegree2 {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Triangle
    }
    fn degree(&self) -> usize {
        2
    }
    fn highest_degree(&self) -> usize {
        2
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        6
    }
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are:
        // * (1-x-y)(1-2x-2y)
        // * x(2x-1)
        // * y(2y - 1)
        // * 4xy
        // * 4y(1-x-y)
        // * 4x(1-x-y)
        for deriv in 0..(nderivs + 1) * (nderivs + 2) / 2 {
            for pt in 0..data.shape().1 {
                let x = points[2 * pt];
                let y = points[2 * pt + 1];
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 2.0 * (1.0 - x - y) * (1.0 - x - y);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = x * (2.0 * x - 1.0);
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 4.0 * x * y;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = 4.0 * y * (1.0 - x - y);
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = 4.0 * x * (1.0 - x - y);
                } else if deriv == 1 {
                    // d/dx
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = -4.0 * (1.0 - x - y);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 4.0 * x - 1.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 4.0 * y;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = -4.0 * y;
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = 4.0 - 8.0 * x - 4.0 * y;
                } else if deriv == 2 {
                    // d/dx
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = -4.0 * (1.0 - x - y);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 4.0 * y - 1.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 4.0 * x;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = 4.0 - 4.0 * x - 8.0 * y;
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = -4.0 * x;
                } else if deriv == 3 {
                    // d2/dx2
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 4.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 4.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = -8.0;
                } else if deriv == 4 {
                    // d2/dxdy
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 4.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 4.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = -8.0;
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = 0.0;
                } else if deriv == 5 {
                    // d2/dy2
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 4.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 4.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 0.0;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = -8.0;
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = 0.0;
                } else {
                    for fun in 0..6 {
                        *data.get_mut(deriv, pt, fun, 0).unwrap() = 0.;
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 0 {
            vec![entity_number]
        } else if entity_dim == 1 {
            vec![3 + entity_number]
        } else {
            vec![]
        }
    }
}

/// Degree 0 Lagrange element on a quadrilateral
pub struct LagrangeElementQuadrilateralDegree0 {}

impl FiniteElement for LagrangeElementQuadrilateralDegree0 {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Quadrilateral
    }
    fn degree(&self) -> usize {
        0
    }
    fn highest_degree(&self) -> usize {
        0
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        1
    }
    fn tabulate(&self, _points: &[f64], nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are (1-x)(1-y), x(1-y), (1-x)y, xy
        for deriv in 0..(nderivs + 1) * (nderivs + 2) / 2 {
            for pt in 0..data.shape().1 {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 1.0;
                } else {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 0.;
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 2 && entity_number == 0 {
            vec![0]
        } else {
            vec![]
        }
    }
}

/// Degree 1 Lagrange element on a quadrilateral
pub struct LagrangeElementQuadrilateralDegree1 {}

impl FiniteElement for LagrangeElementQuadrilateralDegree1 {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Quadrilateral
    }
    fn degree(&self) -> usize {
        1
    }
    fn highest_degree(&self) -> usize {
        1
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        4
    }
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are (1-x)(1-y), x(1-y), (1-x)y, xy
        for deriv in 0..(nderivs + 1) * (nderivs + 2) / 2 {
            for pt in 0..data.shape().1 {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() =
                        (1.0 - points[2 * pt]) * (1.0 - points[2 * pt + 1]);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() =
                        points[2 * pt] * (1.0 - points[2 * pt + 1]);
                    *data.get_mut(deriv, pt, 2, 0).unwrap() =
                        (1.0 - points[2 * pt]) * points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = points[2 * pt] * points[2 * pt + 1];
                } else if deriv == 1 {
                    // d/dx
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = points[2 * pt + 1] - 1.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 1.0 - points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = -points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = points[2 * pt + 1];
                } else if deriv == 2 {
                    // d/dy
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = points[2 * pt] - 1.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = -points[2 * pt];
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 1.0 - points[2 * pt];
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = points[2 * pt];
                } else if deriv == 4 {
                    // d2/dxdy
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 1.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = -1.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = -1.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 1.0;
                } else {
                    for fun in 0..3 {
                        *data.get_mut(deriv, pt, fun, 0).unwrap() = 0.;
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 0 {
            vec![entity_number]
        } else {
            vec![]
        }
    }
}

/// Degree 2 Lagrange element on a quadrilateral
pub struct LagrangeElementQuadrilateralDegree2 {}

impl FiniteElement for LagrangeElementQuadrilateralDegree2 {
    fn value_size(&self) -> usize {
        1
    }
    fn map_type(&self) -> MapType {
        MapType::Identity
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Quadrilateral
    }
    fn degree(&self) -> usize {
        2
    }
    fn highest_degree(&self) -> usize {
        2
    }
    fn family(&self) -> ElementFamily {
        ElementFamily::Lagrange
    }
    fn discontinuous(&self) -> bool {
        false
    }
    fn dim(&self) -> usize {
        9
    }
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut Array4D<f64>) {
        // Basis functions are (1-x)(1-y), x(1-y), (1-x)y, xy
        for deriv in 0..(nderivs + 1) * (nderivs + 2) / 2 {
            for pt in 0..data.shape().1 {
                let x = points[2 * pt];
                let y = points[2 * pt + 1];
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0).unwrap() =
                        (1.0 - x) * (1.0 - 2.0 * x) * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() =
                        x * (2.0 * x - 1.0) * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 2, 0).unwrap() =
                        (1.0 - x) * (1.0 - 2.0 * x) * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 3, 0).unwrap() =
                        x * (2.0 * x - 1.0) * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 4, 0).unwrap() =
                        4.0 * x * (1.0 - x) * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 5, 0).unwrap() =
                        (1.0 - x) * (1.0 - 2.0 * x) * 4.0 * y * (1.0 - y);
                    *data.get_mut(deriv, pt, 6, 0).unwrap() =
                        x * (2.0 * x - 1.0) * 4.0 * y * (1.0 - y);
                    *data.get_mut(deriv, pt, 7, 0).unwrap() =
                        4.0 * x * (1.0 - x) * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 8, 0).unwrap() =
                        4.0 * x * (1.0 - x) * 4.0 * y * (1.0 - y);
                } else if deriv == 1 {
                    // d/dx
                    *data.get_mut(deriv, pt, 0, 0).unwrap() =
                        (4.0 * x - 3.0) * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() =
                        (4.0 * x - 1.0) * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = (4.0 * x - 3.0) * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = (4.0 * x - 1.0) * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 4, 0).unwrap() =
                        4.0 * (1.0 - 2.0 * x) * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = (4.0 * x - 3.0) * 4.0 * y * (1.0 - y);
                    *data.get_mut(deriv, pt, 6, 0).unwrap() = (4.0 * x - 1.0) * 4.0 * y * (1.0 - y);
                    *data.get_mut(deriv, pt, 7, 0).unwrap() =
                        4.0 * (1.0 - 2.0 * x) * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 8, 0).unwrap() =
                        4.0 * (1.0 - 2.0 * x) * 4.0 * y * (1.0 - y);
                } else if deriv == 2 {
                    // d/dy
                    *data.get_mut(deriv, pt, 0, 0).unwrap() =
                        (1.0 - x) * (1.0 - 2.0 * x) * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = x * (2.0 * x - 1.0) * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 2, 0).unwrap() =
                        (1.0 - x) * (1.0 - 2.0 * x) * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = x * (2.0 * x - 1.0) * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = 4.0 * x * (1.0 - x) * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 5, 0).unwrap() =
                        (1.0 - x) * (1.0 - 2.0 * x) * 4.0 * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 6, 0).unwrap() =
                        x * (2.0 * x - 1.0) * 4.0 * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 7, 0).unwrap() = 4.0 * x * (1.0 - x) * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 8, 0).unwrap() =
                        4.0 * x * (1.0 - x) * 4.0 * (1.0 - 2.0 * y);
                } else if deriv == 3 {
                    // d2/dx2
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 4.0 * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 4.0 * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 4.0 * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 4.0 * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = -8.0 * (1.0 - y) * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = 4.0 * 4.0 * y * (1.0 - y);
                    *data.get_mut(deriv, pt, 6, 0).unwrap() = 4.0 * 4.0 * y * (1.0 - y);
                    *data.get_mut(deriv, pt, 7, 0).unwrap() = -8.0 * y * (2.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 8, 0).unwrap() = -8.0 * 4.0 * y * (1.0 - y);
                } else if deriv == 4 {
                    // d2/dxdy
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = (4.0 * x - 3.0) * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = (4.0 * x - 1.0) * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = (4.0 * x - 3.0) * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = (4.0 * x - 1.0) * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 4, 0).unwrap() =
                        4.0 * (1.0 - 2.0 * x) * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 5, 0).unwrap() =
                        (4.0 * x - 3.0) * 4.0 * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 6, 0).unwrap() =
                        (4.0 * x - 1.0) * 4.0 * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 7, 0).unwrap() =
                        4.0 * (1.0 - 2.0 * x) * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 8, 0).unwrap() =
                        4.0 * (1.0 - 2.0 * x) * 4.0 * (1.0 - 2.0 * y);
                } else if deriv == 5 {
                    // d2/dy2
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = (1.0 - x) * (1.0 - 2.0 * x) * 4.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = x * (2.0 * x - 1.0) * 4.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = (1.0 - x) * (1.0 - 2.0 * x) * 4.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = x * (2.0 * x - 1.0) * 4.0;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = 4.0 * x * (1.0 - x) * 4.0;
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = (1.0 - x) * (1.0 - 2.0 * x) * -8.0;
                    *data.get_mut(deriv, pt, 6, 0).unwrap() = x * (2.0 * x - 1.0) * -8.0;
                    *data.get_mut(deriv, pt, 7, 0).unwrap() = 4.0 * x * (1.0 - x) * 4.0;
                    *data.get_mut(deriv, pt, 8, 0).unwrap() = 4.0 * x * (1.0 - x) * -8.0;
                } else if deriv == 7 {
                    // d3/dx2dy
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 4.0 * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 4.0 * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 4.0 * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 4.0 * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = -8.0 * (4.0 * y - 3.0);
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = 4.0 * 4.0 * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 6, 0).unwrap() = 4.0 * 4.0 * (1.0 - 2.0 * y);
                    *data.get_mut(deriv, pt, 7, 0).unwrap() = -8.0 * (4.0 * y - 1.0);
                    *data.get_mut(deriv, pt, 8, 0).unwrap() = -8.0 * 4.0 * (1.0 - 2.0 * y);
                } else if deriv == 8 {
                    // d3/dxdy2
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = (4.0 * x - 3.0) * 4.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = (4.0 * x - 1.0) * 4.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = (4.0 * x - 3.0) * 4.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = (4.0 * x - 1.0) * 4.0;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = 4.0 * (1.0 - 2.0 * x) * 4.0;
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = (4.0 * x - 3.0) * -8.0;
                    *data.get_mut(deriv, pt, 6, 0).unwrap() = (4.0 * x - 1.0) * -8.0;
                    *data.get_mut(deriv, pt, 7, 0).unwrap() = 4.0 * (1.0 - 2.0 * x) * 4.0;
                    *data.get_mut(deriv, pt, 8, 0).unwrap() = 4.0 * (1.0 - 2.0 * x) * -8.0;
                } else if deriv == 12 {
                    // d3/dx2dy
                    *data.get_mut(deriv, pt, 0, 0).unwrap() = 4.0 * 4.0;
                    *data.get_mut(deriv, pt, 1, 0).unwrap() = 4.0 * 4.0;
                    *data.get_mut(deriv, pt, 2, 0).unwrap() = 4.0 * 4.0;
                    *data.get_mut(deriv, pt, 3, 0).unwrap() = 4.0 * 4.0;
                    *data.get_mut(deriv, pt, 4, 0).unwrap() = -8.0 * 4.0;
                    *data.get_mut(deriv, pt, 5, 0).unwrap() = 4.0 * -8.0;
                    *data.get_mut(deriv, pt, 6, 0).unwrap() = 4.0 * -8.0;
                    *data.get_mut(deriv, pt, 7, 0).unwrap() = -8.0 * 4.0;
                    *data.get_mut(deriv, pt, 8, 0).unwrap() = -8.0 * -8.0;
                } else {
                    for fun in 0..3 {
                        *data.get_mut(deriv, pt, fun, 0).unwrap() = 0.;
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        if entity_dim == 0 {
            vec![entity_number]
        } else if entity_dim == 1 {
            vec![4 + entity_number]
        } else if entity_dim == 2 {
            vec![8]
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod test {
    use crate::cell::*;
    use crate::element::*;
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
    fn test_lagrange_0_interval() {
        let e = LagrangeElementIntervalDegree0 {};
        assert_eq!(e.value_size(), 1);
        let mut data = e.create_tabulate_array(0, 4);
        let points = vec![0.0, 0.2, 0.4, 1.0];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(*data.get(0, pt, 0, 0).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_interval() {
        let e = LagrangeElementIntervalDegree1 {};
        assert_eq!(e.value_size(), 1);
        let mut data = e.create_tabulate_array(0, 4);
        let points = vec![0.0, 0.2, 0.4, 1.0];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(*data.get(0, pt, 0, 0).unwrap(), 1.0 - points[pt]);
            assert_relative_eq!(*data.get(0, pt, 1, 0).unwrap(), points[pt]);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_triangle() {
        let e = LagrangeElementTriangleDegree0 {};
        assert_eq!(e.value_size(), 1);
        let mut data = e.create_tabulate_array(0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get(0, pt, 0, 0).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_triangle() {
        let e = LagrangeElementTriangleDegree1 {};
        assert_eq!(e.value_size(), 1);
        let mut data = e.create_tabulate_array(0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get(0, pt, 0, 0).unwrap(),
                1.0 - points[2 * pt] - points[2 * pt + 1]
            );
            assert_relative_eq!(*data.get(0, pt, 1, 0).unwrap(), points[2 * pt]);
            assert_relative_eq!(*data.get(0, pt, 2, 0).unwrap(), points[2 * pt + 1]);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_quadrilateral() {
        let e = LagrangeElementQuadrilateralDegree0 {};
        assert_eq!(e.value_size(), 1);
        let mut data = e.create_tabulate_array(0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.3, 0.2];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get(0, pt, 0, 0).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_quadrilateral() {
        let e = LagrangeElementQuadrilateralDegree1 {};
        assert_eq!(e.value_size(), 1);
        let mut data = e.create_tabulate_array(0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.3, 0.2];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get(0, pt, 0, 0).unwrap(),
                (1.0 - points[2 * pt]) * (1.0 - points[2 * pt + 1])
            );
            assert_relative_eq!(
                *data.get(0, pt, 1, 0).unwrap(),
                points[2 * pt] * (1.0 - points[2 * pt + 1])
            );
            assert_relative_eq!(
                *data.get(0, pt, 2, 0).unwrap(),
                (1.0 - points[2 * pt]) * points[2 * pt + 1]
            );
            assert_relative_eq!(
                *data.get(0, pt, 3, 0).unwrap(),
                points[2 * pt] * points[2 * pt + 1]
            );
        }
        check_dofs(e);
    }
}
