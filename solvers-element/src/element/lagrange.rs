//! Lagrange elements

use crate::element::*;

/// Lagrange element
pub struct LagrangeElement {
    pub celltype: ReferenceCellType,
    pub degree: usize,
}

impl FiniteElement for LagrangeElement {
    const VALUE_SIZE: usize = 1;
    const MAP_TYPE: MapType = MapType::Identity;

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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>) {
        unimplemented!("tabulate not yet implemented for this element");
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize> {
        unimplemented!("entity_dofs not yet implemented for this element");
    }
}

/// Degree 0 Lagrange element on an interval
pub struct LagrangeElementIntervalDegree0 {}

impl FiniteElement for LagrangeElementIntervalDegree0 {
    const VALUE_SIZE: usize = 1;
    const MAP_TYPE: MapType = MapType::Identity;

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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..data.deriv_count() {
            for pt in 0..data.point_count() {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0) = 1.0;
                } else {
                    *data.get_mut(deriv, pt, 0, 0) = 0.;
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
    const VALUE_SIZE: usize = 1;
    const MAP_TYPE: MapType = MapType::Identity;

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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..data.deriv_count() {
            for pt in 0..data.point_count() {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0) = 1.0 - points[pt];
                    *data.get_mut(deriv, pt, 1, 0) = points[pt];
                } else if deriv == 1 {
                    *data.get_mut(deriv, pt, 0, 0) = -1.0;
                    *data.get_mut(deriv, pt, 1, 0) = 1.0;
                } else {
                    for fun in 0..2 {
                        *data.get_mut(deriv, pt, fun, 0) = 0.;
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
    const VALUE_SIZE: usize = 1;
    const MAP_TYPE: MapType = MapType::Identity;

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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..data.deriv_count() {
            for pt in 0..data.point_count() {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0) = 1.0;
                } else {
                    *data.get_mut(deriv, pt, 0, 0) = 0.;
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
    const VALUE_SIZE: usize = 1;
    const MAP_TYPE: MapType = MapType::Identity;

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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>) {
        // Basis functions are 1-x-y, x, y
        for deriv in 0..data.deriv_count() {
            for pt in 0..data.point_count() {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0) = 1.0 - points[2 * pt] - points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 1, 0) = points[2 * pt];
                    *data.get_mut(deriv, pt, 2, 0) = points[2 * pt + 1];
                } else if deriv == 1 {
                    *data.get_mut(deriv, pt, 0, 0) = -1.0;
                    *data.get_mut(deriv, pt, 1, 0) = 1.0;
                    *data.get_mut(deriv, pt, 2, 0) = 0.0;
                } else if deriv == 2 {
                    *data.get_mut(deriv, pt, 0, 0) = -1.0;
                    *data.get_mut(deriv, pt, 1, 0) = 0.0;
                    *data.get_mut(deriv, pt, 2, 0) = 1.0;
                } else {
                    for fun in 0..3 {
                        *data.get_mut(deriv, pt, fun, 0) = 0.;
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

/// Degree 0 Lagrange element on a quadrilateral
pub struct LagrangeElementQuadrilateralDegree0 {}

impl FiniteElement for LagrangeElementQuadrilateralDegree0 {
    const VALUE_SIZE: usize = 1;
    const MAP_TYPE: MapType = MapType::Identity;

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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>) {
        // Basis functions are (1-x)(1-y), x(1-y), (1-x)y, xy
        for deriv in 0..data.deriv_count() {
            for pt in 0..data.point_count() {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0) = 1.0;
                } else {
                    *data.get_mut(deriv, pt, 0, 0) = 0.;
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
    const VALUE_SIZE: usize = 1;
    const MAP_TYPE: MapType = MapType::Identity;

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
    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>) {
        // Basis functions are (1-x)(1-y), x(1-y), (1-x)y, xy
        for deriv in 0..data.deriv_count() {
            for pt in 0..data.point_count() {
                if deriv == 0 {
                    *data.get_mut(deriv, pt, 0, 0) =
                        (1.0 - points[2 * pt]) * (1.0 - points[2 * pt + 1]);
                    *data.get_mut(deriv, pt, 1, 0) = points[2 * pt] * (1.0 - points[2 * pt + 1]);
                    *data.get_mut(deriv, pt, 2, 0) = (1.0 - points[2 * pt]) * points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 3, 0) = points[2 * pt] * points[2 * pt + 1];
                } else if deriv == 1 {
                    // d/dx
                    *data.get_mut(deriv, pt, 0, 0) = points[2 * pt + 1] - 1.0;
                    *data.get_mut(deriv, pt, 1, 0) = 1.0 - points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 2, 0) = -points[2 * pt + 1];
                    *data.get_mut(deriv, pt, 3, 0) = points[2 * pt + 1];
                } else if deriv == 2 {
                    // d/dy
                    *data.get_mut(deriv, pt, 0, 0) = points[2 * pt] - 1.0;
                    *data.get_mut(deriv, pt, 1, 0) = -points[2 * pt];
                    *data.get_mut(deriv, pt, 2, 0) = 1.0 - points[2 * pt];
                    *data.get_mut(deriv, pt, 3, 0) = points[2 * pt];
                } else if deriv == 4 {
                    // d2/dxdy
                    *data.get_mut(deriv, pt, 0, 0) = 1.0;
                    *data.get_mut(deriv, pt, 1, 0) = -1.0;
                    *data.get_mut(deriv, pt, 2, 0) = -1.0;
                    *data.get_mut(deriv, pt, 3, 0) = 1.0;
                } else {
                    for fun in 0..3 {
                        *data.get_mut(deriv, pt, fun, 0) = 0.;
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
    fn test_lagrange_0_interval() {
        let e = LagrangeElementIntervalDegree0 {};
        assert_eq!(e.value_size(), 1);
        let mut data = TabulatedData::new(&e, 0, 4);
        let points = vec![0.0, 0.2, 0.4, 1.0];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(*data.get(0, pt, 0, 0), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_interval() {
        let e = LagrangeElementIntervalDegree1 {};
        assert_eq!(e.value_size(), 1);
        let mut data = TabulatedData::new(&e, 0, 4);
        let points = vec![0.0, 0.2, 0.4, 1.0];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(*data.get(0, pt, 0, 0), 1.0 - points[pt]);
            assert_relative_eq!(*data.get(0, pt, 1, 0), points[pt]);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_triangle() {
        let e = LagrangeElementTriangleDegree0 {};
        assert_eq!(e.value_size(), 1);
        let mut data = TabulatedData::new(&e, 0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get(0, pt, 0, 0), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_triangle() {
        let e = LagrangeElementTriangleDegree1 {};
        assert_eq!(e.value_size(), 1);
        let mut data = TabulatedData::new(&e, 0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get(0, pt, 0, 0),
                1.0 - points[2 * pt] - points[2 * pt + 1]
            );
            assert_relative_eq!(*data.get(0, pt, 1, 0), points[2 * pt]);
            assert_relative_eq!(*data.get(0, pt, 2, 0), points[2 * pt + 1]);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_quadrilateral() {
        let e = LagrangeElementQuadrilateralDegree0 {};
        assert_eq!(e.value_size(), 1);
        let mut data = TabulatedData::new(&e, 0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.3, 0.2];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get(0, pt, 0, 0), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_quadrilateral() {
        let e = LagrangeElementQuadrilateralDegree1 {};
        assert_eq!(e.value_size(), 1);
        let mut data = TabulatedData::new(&e, 0, 6);
        let points = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.3, 0.2];
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(
                *data.get(0, pt, 0, 0),
                (1.0 - points[2 * pt]) * (1.0 - points[2 * pt + 1])
            );
            assert_relative_eq!(
                *data.get(0, pt, 1, 0),
                points[2 * pt] * (1.0 - points[2 * pt + 1])
            );
            assert_relative_eq!(
                *data.get(0, pt, 2, 0),
                (1.0 - points[2 * pt]) * points[2 * pt + 1]
            );
            assert_relative_eq!(*data.get(0, pt, 3, 0), points[2 * pt] * points[2 * pt + 1]);
        }
        check_dofs(e);
    }
}
