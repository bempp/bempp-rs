//! Finite element definitions

use crate::element::polynomials::{
    legendre_shape, polynomial_count, tabulate_legendre_polynomials,
};
use crate::element::reference_cell;
use crate::traits::element::{Continuity, FiniteElement, MapType};
use crate::traits::types::ReferenceCell;
use rlst::LinAlg;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, Array, BaseArray, RandomAccessByRef, RandomAccessMut,
    RlstScalar, Shape, UnsafeRandomAccessMut, VectorContainer,
};

pub mod lagrange;
pub mod raviart_thomas;
pub use lagrange::LagrangeElementFamily;
pub use raviart_thomas::RaviartThomasElementFamily;

type EntityPoints<T> = [Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>; 4];
type EntityWeights<T> = [Vec<Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>>; 4];

/// A Ciarlet element
pub struct CiarletElement<T: RlstScalar> {
    cell_type: ReferenceCell,
    degree: usize,
    embedded_superdegree: usize,
    map_type: MapType,
    value_shape: Vec<usize>,
    value_size: usize,
    continuity: Continuity,
    dim: usize,
    coefficients: Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
    entity_dofs: [Vec<Vec<usize>>; 4],
    // interpolation_points: EntityPoints,
    // interpolation_weights: EntityWeights,
}

impl<T: RlstScalar> CiarletElement<T>
where
    T: LinAlg,
{
    /// Create a Ciarlet element
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        cell_type: ReferenceCell,
        degree: usize,
        value_shape: Vec<usize>,
        polynomial_coeffs: Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
        interpolation_points: EntityPoints<T::Real>,
        interpolation_weights: EntityWeights<T>,
        map_type: MapType,
        continuity: Continuity,
        embedded_superdegree: usize,
    ) -> Self {
        let mut dim = 0;
        let mut npts = 0;

        for emats in &interpolation_weights {
            for mat in emats {
                dim += mat.shape()[0];
                npts += mat.shape()[2];
            }
        }
        let tdim = reference_cell::dim(cell_type);

        let mut value_size = 1;
        for i in &value_shape {
            value_size *= *i;
        }

        for matrices in &interpolation_weights {
            for mat in matrices {
                if mat.shape()[1] != value_size {
                    panic!("Incompatible value size");
                }
            }
        }

        let new_pts = if continuity == Continuity::Discontinuous {
            let mut new_pts: EntityPoints<T::Real> = [vec![], vec![], vec![], vec![]];
            let mut pn = 0;
            let mut all_pts = rlst_dynamic_array2![T::Real, [npts, tdim]];
            for (i, pts_i) in interpolation_points.iter().take(tdim).enumerate() {
                for _pts in pts_i {
                    new_pts[i].push(rlst_dynamic_array2![T::Real, [0, tdim]]);
                }
            }
            for pts_i in interpolation_points.iter() {
                for pts in pts_i {
                    for j in 0..pts.shape()[0] {
                        for k in 0..tdim {
                            *all_pts.get_mut([pn + j, k]).unwrap() = *pts.get([j, k]).unwrap();
                        }
                    }
                    pn += pts.shape()[0];
                }
            }
            new_pts[tdim].push(all_pts);
            new_pts
        } else {
            interpolation_points
        };
        let new_wts = if continuity == Continuity::Discontinuous {
            let mut new_wts = [vec![], vec![], vec![], vec![]];
            let mut pn = 0;
            let mut dn = 0;
            let mut all_mat = rlst_dynamic_array3!(T, [dim, value_size, npts]);
            for (i, mi) in interpolation_weights.iter().take(tdim).enumerate() {
                for _mat in mi {
                    new_wts[i].push(rlst_dynamic_array3!(T, [0, value_size, 0]));
                }
            }
            for mi in interpolation_weights.iter() {
                for mat in mi {
                    for j in 0..mat.shape()[0] {
                        for k in 0..value_size {
                            for l in 0..mat.shape()[2] {
                                *all_mat.get_mut([dn + j, k, pn + l]).unwrap() =
                                    *mat.get([j, k, l]).unwrap();
                            }
                        }
                    }
                    dn += mat.shape()[0];
                    pn += mat.shape()[2];
                }
            }
            new_wts[tdim].push(all_mat);
            new_wts
        } else {
            interpolation_weights
        };

        // Compute the dual matrix
        let pdim = polynomial_count(cell_type, embedded_superdegree);
        let mut d_matrix = rlst_dynamic_array3!(T, [value_size, pdim, dim]);

        let mut dof = 0;
        for d in 0..4 {
            for (e, pts) in new_pts[d].iter().enumerate() {
                if pts.shape()[0] > 0 {
                    let mut table = rlst_dynamic_array3!(T, [1, pdim, pts.shape()[0]]);
                    tabulate_legendre_polynomials(
                        cell_type,
                        pts,
                        embedded_superdegree,
                        0,
                        &mut table,
                    );
                    let mat = &new_wts[d][e];
                    for i in 0..mat.shape()[0] {
                        for j in 0..value_size {
                            for l in 0..pdim {
                                let value = d_matrix.get_mut([j, l, dof + i]).unwrap();
                                *value = T::from(0.0).unwrap();
                                for k in 0..pts.shape()[0] {
                                    *value += *mat.get([i, j, k]).unwrap()
                                        * *table.get([0, l, k]).unwrap();
                                }
                            }
                        }
                    }
                    dof += mat.shape()[0];
                }
            }
        }

        let mut inverse = rlst::rlst_dynamic_array2!(T, [dim, dim]);

        for i in 0..dim {
            for j in 0..dim {
                let entry = inverse.get_mut([i, j]).unwrap();
                *entry = T::from(0.0).unwrap();
                for k in 0..value_size {
                    for l in 0..pdim {
                        *entry += *polynomial_coeffs.get([i, k, l]).unwrap()
                            * *d_matrix.get([k, l, j]).unwrap();
                    }
                }
            }
        }

        let mut ident = rlst::rlst_dynamic_array2!(T, [dim, dim]);
        for i in 0..dim {
            unsafe {
                *ident.get_unchecked_mut([i, i]) = T::from(1.0).unwrap();
            }
        }
        inverse.view_mut().into_inverse_alloc().unwrap();

        let mut coefficients = rlst_dynamic_array3!(T, [dim, value_size, pdim]);
        for i in 0..dim {
            for l in 0..pdim {
                for j in 0..value_size {
                    for k in 0..pdim {
                        *coefficients.get_mut([i, j, k]).unwrap() += *inverse.get([i, l]).unwrap()
                            * *polynomial_coeffs.get([l, j, k]).unwrap()
                    }
                }
            }
        }

        let mut entity_dofs = [vec![], vec![], vec![], vec![]];
        let mut dof = 0;
        for i in 0..4 {
            for pts in &new_pts[i] {
                let dofs = (dof..dof + pts.shape()[0]).collect::<Vec<_>>();
                entity_dofs[i].push(dofs);
                dof += pts.shape()[0];
            }
        }
        CiarletElement::<T> {
            cell_type,
            degree,
            embedded_superdegree,
            map_type,
            value_shape,
            value_size,
            continuity,
            dim,
            coefficients,
            entity_dofs,
            // interpolation_points: new_pts,
            // interpolation_weights: new_wts,
        }
    }

    /// The polynomial degree
    pub fn degree(&self) -> usize {
        self.degree
    }
}

impl<T: RlstScalar> FiniteElement for CiarletElement<T> {
    type T = T;
    fn value_shape(&self) -> &[usize] {
        &self.value_shape
    }
    fn value_size(&self) -> usize {
        self.value_size
    }
    fn map_type(&self) -> MapType {
        self.map_type
    }

    fn cell_type(&self) -> ReferenceCell {
        self.cell_type
    }
    fn embedded_superdegree(&self) -> usize {
        self.embedded_superdegree
    }
    fn continuity(&self) -> Continuity {
        self.continuity
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn tabulate<Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>>(
        &self,
        points: &Array2,
        nderivs: usize,
        data: &mut impl RandomAccessMut<4, Item = T>,
    ) {
        let mut table = rlst_dynamic_array3!(
            T,
            legendre_shape(self.cell_type, points, self.embedded_superdegree, nderivs,)
        );
        tabulate_legendre_polynomials(
            self.cell_type,
            points,
            self.embedded_superdegree,
            nderivs,
            &mut table,
        );

        for d in 0..table.shape()[0] {
            for p in 0..points.shape()[0] {
                for j in 0..self.value_size {
                    for b in 0..self.dim {
                        let value = data.get_mut([d, p, b, j]).unwrap();
                        *value = T::from(0.0).unwrap();
                        for i in 0..table.shape()[1] {
                            *value += *self.coefficients.get([b, j, i]).unwrap()
                                * *table.get_mut([d, i, p]).unwrap();
                        }
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]> {
        if entity_dim < 4 && entity_number < self.entity_dofs[entity_dim].len() {
            Some(&self.entity_dofs[entity_dim][entity_number])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use rlst::rlst_dynamic_array4;

    fn check_dofs(e: impl FiniteElement) {
        let mut ndofs = 0;
        for (dim, entity_count) in match e.cell_type() {
            ReferenceCell::Point => vec![1],
            ReferenceCell::Interval => vec![2, 1],
            ReferenceCell::Triangle => vec![3, 3, 1],
            ReferenceCell::Quadrilateral => vec![4, 4, 1],
            ReferenceCell::Tetrahedron => vec![4, 6, 4, 1],
            ReferenceCell::Hexahedron => vec![8, 12, 6, 1],
            ReferenceCell::Prism => vec![6, 9, 5, 1],
            ReferenceCell::Pyramid => vec![5, 8, 5, 1],
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
    fn test_lagrange_1() {
        let e = lagrange::create::<f64>(ReferenceCell::Triangle, 1, Continuity::Continuous);
        assert_eq!(e.value_size(), 1);
    }

    #[test]
    fn test_lagrange_0_interval() {
        let e = lagrange::create::<f64>(ReferenceCell::Interval, 0, Continuity::Discontinuous);
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
        let e = lagrange::create::<f64>(ReferenceCell::Interval, 1, Continuity::Continuous);
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
        let e = lagrange::create::<f64>(ReferenceCell::Triangle, 0, Continuity::Discontinuous);
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
        let e = lagrange::create::<f64>(ReferenceCell::Triangle, 1, Continuity::Continuous);
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
        lagrange::create::<f64>(ReferenceCell::Triangle, 2, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Triangle, 3, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Triangle, 4, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Triangle, 5, Continuity::Continuous);

        lagrange::create::<f64>(ReferenceCell::Triangle, 2, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Triangle, 3, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Triangle, 4, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Triangle, 5, Continuity::Discontinuous);
    }

    #[test]
    fn test_lagrange_higher_degree_interval() {
        lagrange::create::<f64>(ReferenceCell::Interval, 2, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Interval, 3, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Interval, 4, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Interval, 5, Continuity::Continuous);

        lagrange::create::<f64>(ReferenceCell::Interval, 2, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Interval, 3, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Interval, 4, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Interval, 5, Continuity::Discontinuous);
    }

    #[test]
    fn test_lagrange_higher_degree_quadrilateral() {
        lagrange::create::<f64>(ReferenceCell::Quadrilateral, 2, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Quadrilateral, 3, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Quadrilateral, 4, Continuity::Continuous);
        lagrange::create::<f64>(ReferenceCell::Quadrilateral, 5, Continuity::Continuous);

        lagrange::create::<f64>(ReferenceCell::Quadrilateral, 2, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Quadrilateral, 3, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Quadrilateral, 4, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCell::Quadrilateral, 5, Continuity::Discontinuous);
    }

    #[test]
    fn test_lagrange_0_quadrilateral() {
        let e = lagrange::create::<f64>(ReferenceCell::Quadrilateral, 0, Continuity::Discontinuous);
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
        let e = lagrange::create::<f64>(ReferenceCell::Quadrilateral, 1, Continuity::Continuous);
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
        let e = lagrange::create::<f64>(ReferenceCell::Quadrilateral, 2, Continuity::Continuous);
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

    #[test]
    fn test_raviart_thomas_1_triangle() {
        let e = raviart_thomas::create(ReferenceCell::Triangle, 1, Continuity::Continuous);
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
