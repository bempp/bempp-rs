//! Finite Element definitions

use crate::cell::create_cell;
use crate::polynomials::{legendre_shape, polynomial_count, tabulate_legendre_polynomials};
use bempp_tools::arrays::AdjacencyList;
use bempp_traits::arrays::AdjacencyListAccess;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, FiniteElement, MapType};
use rlst_dense::linalg::inverse::MatrixInverse;
use rlst_dense::{
    array::Array,
    base_array::BaseArray,
    data_container::VectorContainer,
    traits::{RandomAccessByRef, RandomAccessMut, Shape, UnsafeRandomAccessMut},
};
use rlst_dense::{rlst_dynamic_array2, rlst_dynamic_array3};
pub mod lagrange;
pub mod raviart_thomas;

type EntityPoints = [Vec<Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>>; 4];
type EntityWeights = [Vec<Array<f64, BaseArray<f64, VectorContainer<f64>, 3>, 3>>; 4];

/// The family of an element
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ElementFamily {
    Lagrange = 0,
    RaviartThomas = 1,
}

pub struct CiarletElement {
    cell_type: ReferenceCellType,
    family: ElementFamily,
    degree: usize,
    embedded_superdegree: usize,
    map_type: MapType,
    value_shape: Vec<usize>,
    value_size: usize,
    continuity: Continuity,
    dim: usize,
    coefficients: Array<f64, BaseArray<f64, VectorContainer<f64>, 3>, 3>,
    entity_dofs: [AdjacencyList<usize>; 4],
    // interpolation_points: EntityPoints,
    // interpolation_weights: EntityWeights,
}

impl CiarletElement {
    /// Create a Ciarlet element
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        cell_type: ReferenceCellType,
        family: ElementFamily,
        degree: usize,
        value_shape: Vec<usize>,
        polynomial_coeffs: Array<f64, BaseArray<f64, VectorContainer<f64>, 3>, 3>,
        interpolation_points: EntityPoints,
        interpolation_weights: EntityWeights,
        map_type: MapType,
        continuity: Continuity,
        embedded_superdegree: usize,
    ) -> CiarletElement {
        let mut dim = 0;
        let mut npts = 0;

        for emats in &interpolation_weights {
            for mat in emats {
                dim += mat.shape()[0];
                npts += mat.shape()[2];
            }
        }
        let cell = create_cell(cell_type);
        let tdim = cell.dim();

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
            let mut new_pts: EntityPoints = [vec![], vec![], vec![], vec![]];
            let mut pn = 0;
            let mut all_pts = rlst_dynamic_array2![f64, [npts, tdim]];
            for (i, pts_i) in interpolation_points.iter().take(tdim).enumerate() {
                for _pts in pts_i {
                    new_pts[i].push(rlst_dynamic_array2![f64, [0, tdim]]);
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
            let mut all_mat = rlst_dynamic_array3!(f64, [dim, value_size, npts]);
            for (i, mi) in interpolation_weights.iter().take(tdim).enumerate() {
                for _mat in mi {
                    new_wts[i].push(rlst_dynamic_array3!(f64, [0, value_size, 0]));
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
        let mut d_matrix = rlst_dynamic_array3!(f64, [value_size, pdim, dim]);

        let mut dof = 0;
        for d in 0..4 {
            for (e, pts) in new_pts[d].iter().enumerate() {
                if pts.shape()[0] > 0 {
                    let mut table = rlst_dynamic_array3!(f64, [1, pdim, pts.shape()[0]]);
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
                                *value = 0.0;
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

        let mut inverse = rlst_dense::rlst_dynamic_array2!(f64, [dim, dim]);

        for i in 0..dim {
            for j in 0..dim {
                let entry = inverse.get_mut([i, j]).unwrap();
                *entry = 0.0;
                for k in 0..value_size {
                    for l in 0..pdim {
                        *entry += *polynomial_coeffs.get([i, k, l]).unwrap()
                            * *d_matrix.get([k, l, j]).unwrap();
                    }
                }
            }
        }

        let mut ident = rlst_dense::rlst_dynamic_array2!(f64, [dim, dim]);
        for i in 0..dim {
            unsafe {
                *ident.get_unchecked_mut([i, i]) = 1.0;
            }
        }
        inverse.view_mut().into_inverse_alloc().unwrap();

        let mut coefficients = rlst_dynamic_array3!(f64, [dim, value_size, pdim]);
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

        let mut entity_dofs = [
            AdjacencyList::<usize>::new(),
            AdjacencyList::<usize>::new(),
            AdjacencyList::<usize>::new(),
            AdjacencyList::<usize>::new(),
        ];
        let mut dof = 0;
        for i in 0..4 {
            for pts in &new_pts[i] {
                let dofs: Vec<usize> = (dof..dof + pts.shape()[0]).collect();
                entity_dofs[i].add_row(&dofs);
                dof += pts.shape()[0];
            }
        }
        CiarletElement {
            cell_type,
            family,
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

    // The element family
    pub fn family(&self) -> ElementFamily {
        self.family
    }

    // The polynomial degree
    pub fn degree(&self) -> usize {
        self.degree
    }
}

impl FiniteElement for CiarletElement {
    fn value_shape(&self) -> &[usize] {
        &self.value_shape
    }
    fn value_size(&self) -> usize {
        self.value_size
    }
    fn map_type(&self) -> MapType {
        self.map_type
    }

    fn cell_type(&self) -> ReferenceCellType {
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
    fn is_lagrange(&self) -> bool {
        self.family == ElementFamily::Lagrange
    }
    fn tabulate<
        T: RandomAccessByRef<2, Item = f64> + Shape<2>,
        T4Mut: RandomAccessMut<4, Item = f64>,
    >(
        &self,
        points: &T,
        nderivs: usize,
        data: &mut T4Mut,
    ) {
        let mut table = rlst_dynamic_array3!(
            f64,
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
                        *value = 0.0;
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
        self.entity_dofs[entity_dim].row(entity_number)
    }
}

pub fn create_element(
    family: ElementFamily,
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement {
    match family {
        ElementFamily::Lagrange => lagrange::create(cell_type, degree, continuity),
        ElementFamily::RaviartThomas => raviart_thomas::create(cell_type, degree, continuity),
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
            Continuity::Continuous,
        );
        assert_eq!(e.value_size(), 1);
    }
}
