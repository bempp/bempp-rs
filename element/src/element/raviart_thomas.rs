//! Raviart-Thomas elements

use crate::element::{create_cell, CiarletElement};
use crate::polynomials::polynomial_count;
use bempp_tools::arrays::{to_matrix, Array3D};
use bempp_traits::arrays::Array3DAccess;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily, MapType};

/// Create a Raviart-Thomas element
pub fn create(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement {
    if cell_type != ReferenceCellType::Triangle && cell_type != ReferenceCellType::Quadrilateral {
        panic!("Unsupported cell type");
    }

    if cell_type != ReferenceCellType::Triangle {
        panic!("RT elements on quadrilaterals not implemented yet");
    }
    if degree != 1 {
        panic!("Degree > 1 RT elements not implemented yet");
    }

    let cell = create_cell(cell_type);
    let pdim = polynomial_count(cell_type, degree);
    let tdim = cell.dim();
    let edim = tdim * polynomial_count(cell_type, degree - 1) + degree;

    let mut wcoeffs = Array3D::<f64>::new((edim, tdim, pdim));

    // [sqrt(2), 6*y - 2, 4*sqrt(3)*(x + y/2 - 1/2)]

    // norm(x**2 + y**2)
    // sqrt(70)/30

    *wcoeffs.get_mut(0, 0, 0).unwrap() = 1.0;
    *wcoeffs.get_mut(1, 1, 0).unwrap() = 1.0;
    *wcoeffs.get_mut(2, 0, 1).unwrap() = -0.5 / f64::sqrt(2.0);
    *wcoeffs.get_mut(2, 0, 2).unwrap() = 0.5 * f64::sqrt(1.5);
    *wcoeffs.get_mut(2, 1, 1).unwrap() = 1.0 / f64::sqrt(2.0);

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];
    for _e in 0..cell.entity_count(0) {
        x[0].push(to_matrix(&[], (0, tdim)));
        m[0].push(Array3D::<f64>::new((0, 2, 0)));
    }

    for e in 0..cell.entity_count(1) {
        let mut pts = vec![0.0; tdim];
        let mut mat = vec![0.0; 2];
        let vn0 = cell.edges()[2 * e];
        let vn1 = cell.edges()[2 * e + 1];
        let v0 = &cell.vertices()[vn0 * tdim..(vn0 + 1) * tdim];
        let v1 = &cell.vertices()[vn1 * tdim..(vn1 + 1) * tdim];
        for i in 0..tdim {
            pts[i] = (v0[i] + v1[i]) / 2.0;
        }
        mat[0] = v0[1] - v1[1];
        mat[1] = v1[0] - v0[0];
        x[1].push(to_matrix(&pts, (1, tdim)));
        m[1].push(Array3D::<f64>::from_data(mat, (1, 2, 1)));
    }

    for _e in 0..cell.entity_count(2) {
        x[2].push(to_matrix(&[], (0, tdim)));
        m[2].push(Array3D::<f64>::new((0, 2, 0)));
    }

    CiarletElement::create(
        ElementFamily::RaviartThomas,
        cell_type,
        degree,
        vec![2],
        wcoeffs,
        x,
        m,
        MapType::ContravariantPiola,
        continuity,
        degree,
    )
}

#[cfg(test)]
mod test {
    use crate::cell::*;
    use crate::element::raviart_thomas::*;
    use approx::*;
    use bempp_tools::arrays::Array4D;
    use bempp_traits::arrays::Array4DAccess;
    use bempp_traits::element::FiniteElement;
    use rlst_dense::RandomAccessByRef;

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
        let points = to_matrix(
            &vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
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
