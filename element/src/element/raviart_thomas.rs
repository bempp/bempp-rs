//! Raviart-Thomas elements

use crate::element::{create_cell, CiarletElement, ElementFamily};
use crate::polynomials::polynomial_count;
use bempp_traits::element::{Continuity, MapType};
use bempp_traits::types::ReferenceCellType;
use rlst_dense::linalg::inverse::MatrixInverse;
use rlst_dense::types::RlstScalar;
use rlst_dense::{
    array::views::ArrayViewMut, array::Array, base_array::BaseArray,
    data_container::VectorContainer, rlst_dynamic_array2, rlst_dynamic_array3,
    traits::RandomAccessMut,
};

/// Create a Raviart-Thomas element
pub fn create<T: RlstScalar>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
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

    let mut wcoeffs = rlst_dynamic_array3!(T, [edim, tdim, pdim]);

    // [sqrt(2), 6*y - 2, 4*sqrt(3)*(x + y/2 - 1/2)]

    // norm(x**2 + y**2)
    // sqrt(70)/30

    *wcoeffs.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
    *wcoeffs.get_mut([1, 1, 0]).unwrap() = T::from(1.0).unwrap();
    *wcoeffs.get_mut([2, 0, 1]).unwrap() = T::from(-0.5).unwrap() / T::sqrt(T::from(2.0).unwrap());
    *wcoeffs.get_mut([2, 0, 2]).unwrap() = T::from(0.5).unwrap() * T::sqrt(T::from(1.5).unwrap());
    *wcoeffs.get_mut([2, 1, 1]).unwrap() = T::from(1.0).unwrap() / T::sqrt(T::from(2.0).unwrap());

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];
    for _e in 0..cell.entity_count(0) {
        x[0].push(rlst_dynamic_array2!(T, [0, tdim]));
        m[0].push(rlst_dynamic_array3!(T, [0, 2, 0]));
    }

    for e in 0..cell.entity_count(1) {
        let mut pts = rlst_dynamic_array2!(T, [1, tdim]);
        let mut mat = rlst_dynamic_array3!(T, [1, 2, 1]);
        let vn0 = cell.edges()[2 * e];
        let vn1 = cell.edges()[2 * e + 1];
        let v0 = &cell.vertices()[vn0 * tdim..(vn0 + 1) * tdim];
        let v1 = &cell.vertices()[vn1 * tdim..(vn1 + 1) * tdim];
        for i in 0..tdim {
            *pts.get_mut([0, i]).unwrap() = T::from(v0[i] + v1[i]).unwrap() / T::from(2.0).unwrap();
        }
        *mat.get_mut([0, 0, 0]).unwrap() = T::from(v0[1] - v1[1]).unwrap();
        *mat.get_mut([0, 1, 0]).unwrap() = T::from(v1[0] - v0[0]).unwrap();
        x[1].push(pts);
        m[1].push(mat);
    }

    for _e in 0..cell.entity_count(2) {
        x[2].push(rlst_dynamic_array2!(T, [0, tdim]));
        m[2].push(rlst_dynamic_array3!(T, [0, 2, 0]))
    }

    CiarletElement::create(
        cell_type,
        ElementFamily::RaviartThomas,
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
    use bempp_traits::element::FiniteElement;
    use rlst_dense::rlst_dynamic_array4;
    use rlst_dense::traits::RandomAccessByRef;

    fn check_dofs(e: impl FiniteElement) {
        let cell_dim = match e.cell_type() {
            ReferenceCellType::Point => 0,
            ReferenceCellType::Interval => 1,
            ReferenceCellType::Triangle => 2,
            ReferenceCellType::Quadrilateral => 2,
            ReferenceCellType::Tetrahedron => 3,
            ReferenceCellType::Hexahedron => 3,
            ReferenceCellType::Prism => 3,
            ReferenceCellType::Pyramid => 3,
        };
        let mut ndofs = 0;
        for (dim, entity_count) in match e.cell_type() {
            ReferenceCellType::Point => vec![1],
            ReferenceCellType::Interval => vec![2, 1],
            ReferenceCellType::Triangle => vec![3, 3, 1],
            ReferenceCellType::Quadrilateral => vec![4, 4, 1],
            ReferenceCellType::Tetrahedron => vec![4, 6, 4, 1],
            ReferenceCellType::Hexahedron => vec![8, 12, 6, 1],
            ReferenceCellType::Prism => vec![6, 9, 5, 1],
            ReferenceCellType::Pyramid => vec![5, 8, 5, 1],
        }
        .iter()
        .enumerate()
        {
            for entity in 0..*entity_count {
                ndofs += e.entity_dofs(dim, entity).unwrap().len();
            }
        }
        assert_eq!(ndofs, e.dim());
    }

    #[test]
    fn test_raviart_thomas_1_triangle() {
        let e = create(ReferenceCellType::Triangle, 1, Continuity::Continuous);
        assert_eq!(e.value_size(), 2);
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
                -*points.get([pt, 0]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 0, 1]).unwrap(),
                -*points.get([pt, 1]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([pt, 0]).unwrap() - 1.0
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 1]).unwrap(),
                *points.get([pt, 1]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                -*points.get([pt, 0]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 1]).unwrap(),
                1.0 - *points.get([pt, 1]).unwrap()
            );
        }
        check_dofs(e);
    }
}
