//! Finite Element definitions

use crate::cell::create_cell;
use crate::polynomials::{legendre_shape, polynomial_count, tabulate_legendre_polynomials};
use bempp_tools::arrays::{AdjacencyList, Array2D, Array3D};
use bempp_traits::arrays::{AdjacencyListAccess, Array2DAccess, Array3DAccess, Array4DAccess};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily, FiniteElement, MapType};
use rlst_algorithms::linalg::LinAlg;
use rlst_algorithms::traits::inverse::Inverse;
use rlst_dense::{RandomAccessByRef, RandomAccessMut};
pub mod lagrange;
pub mod raviart_thomas;

pub struct CiarletElement {
    cell_type: ReferenceCellType,
    degree: usize,
    highest_degree: usize,
    map_type: MapType,
    value_shape: Vec<usize>,
    value_size: usize,
    family: ElementFamily,
    continuity: Continuity,
    dim: usize,
    coefficients: Array3D<f64>,
    entity_dofs: [AdjacencyList<usize>; 4],
    interpolation_points: [Vec<Array2D<f64>>; 4],
    interpolation_weights: [Vec<Array3D<f64>>; 4],
}

impl CiarletElement {
    /// Create a Ciarlet element
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        family: ElementFamily,
        cell_type: ReferenceCellType,
        degree: usize,
        value_shape: Vec<usize>,
        polynomial_coeffs: Array3D<f64>,
        interpolation_points: [Vec<Array2D<f64>>; 4],
        interpolation_weights: [Vec<Array3D<f64>>; 4],
        map_type: MapType,
        continuity: Continuity,
        highest_degree: usize,
    ) -> CiarletElement {
        let mut dim = 0;
        let mut npts = 0;

        for emats in &interpolation_weights {
            for mat in emats {
                dim += mat.shape().0;
                npts += mat.shape().2;
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
                if mat.shape().1 != value_size {
                    panic!("Incompatible value size");
                }
            }
        }

        let new_pts = if continuity == Continuity::Discontinuous {
            let mut new_pts = [vec![], vec![], vec![], vec![]];
            let mut pn = 0;
            let mut all_pts = Array2D::<f64>::new((npts, tdim));
            for (i, pts_i) in interpolation_points.iter().take(tdim).enumerate() {
                for _pts in pts_i {
                    new_pts[i].push(Array2D::<f64>::new((0, tdim)));
                }
            }
            for pts_i in interpolation_points.iter() {
                for pts in pts_i {
                    for j in 0..pts.shape().0 {
                        for k in 0..tdim {
                            *all_pts.get_mut(pn + j, k).unwrap() = *pts.get(j, k).unwrap();
                        }
                    }
                    pn += pts.shape().0;
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
            let mut all_mat = Array3D::<f64>::new((dim, value_size, npts));
            for (i, mi) in interpolation_weights.iter().take(tdim).enumerate() {
                for _mat in mi {
                    new_wts[i].push(Array3D::<f64>::new((0, value_size, 0)));
                }
            }
            for mi in interpolation_weights.iter() {
                for mat in mi {
                    for j in 0..mat.shape().0 {
                        for k in 0..value_size {
                            for l in 0..mat.shape().2 {
                                *all_mat.get_mut(dn + j, k, pn + l).unwrap() =
                                    *mat.get(j, k, l).unwrap();
                            }
                        }
                    }
                    dn += mat.shape().0;
                    pn += mat.shape().2;
                }
            }
            new_wts[tdim].push(all_mat);
            new_wts
        } else {
            interpolation_weights
        };

        // Compute the dual matrix
        let pdim = polynomial_count(cell_type, highest_degree);
        let mut d_matrix = Array3D::<f64>::new((value_size, pdim, dim));

        let mut dof = 0;
        for d in 0..4 {
            for (e, pts) in new_pts[d].iter().enumerate() {
                if pts.shape().0 > 0 {
                    let mut table = Array3D::<f64>::new((1, pdim, pts.shape().0));
                    tabulate_legendre_polynomials(cell_type, pts, highest_degree, 0, &mut table);
                    let mat = &new_wts[d][e];
                    for i in 0..mat.shape().0 {
                        for j in 0..value_size {
                            for l in 0..pdim {
                                let value = d_matrix.get_mut(j, l, dof + i).unwrap();
                                *value = 0.0;
                                for k in 0..pts.shape().0 {
                                    *value +=
                                        *mat.get(i, j, k).unwrap() * *table.get(0, l, k).unwrap();
                                }
                            }
                        }
                    }
                    dof += mat.shape().0;
                }
            }
        }

        let mut dual_matrix = rlst_dense::rlst_mat![f64, (dim, dim)];

        for i in 0..dim {
            for j in 0..dim {
                let entry = dual_matrix.get_mut(i, j).unwrap();
                *entry = 0.0;
                for k in 0..value_size {
                    for l in 0..pdim {
                        *entry += *polynomial_coeffs.get(i, k, l).unwrap()
                            * *d_matrix.get(k, l, j).unwrap();
                    }
                }
            }
        }

        let inverse = dual_matrix.linalg().inverse().unwrap();

        let mut coefficients = Array3D::<f64>::new((dim, value_size, pdim));
        for i in 0..dim {
            for l in 0..pdim {
                for j in 0..value_size {
                    for k in 0..pdim {
                        *coefficients.get_mut(i, j, k).unwrap() +=
                            *inverse.get(i, l).unwrap() * *polynomial_coeffs.get(l, j, k).unwrap()
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
                let dofs: Vec<usize> = (dof..dof + pts.shape().0).collect();
                entity_dofs[i].add_row(&dofs);
                dof += pts.shape().0;
            }
        }
        CiarletElement {
            cell_type,
            degree,
            highest_degree,
            map_type,
            value_shape,
            value_size,
            family,
            continuity,
            dim,
            coefficients,
            entity_dofs,
            interpolation_points: new_pts,
            interpolation_weights: new_wts,
        }
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
    fn degree(&self) -> usize {
        self.degree
    }
    fn highest_degree(&self) -> usize {
        self.highest_degree
    }
    fn family(&self) -> ElementFamily {
        self.family
    }
    fn continuity(&self) -> Continuity {
        self.continuity
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
        let mut table = Array3D::<f64>::new(legendre_shape(
            self.cell_type,
            points,
            self.highest_degree,
            nderivs,
        ));
        tabulate_legendre_polynomials(
            self.cell_type,
            points,
            self.highest_degree,
            nderivs,
            &mut table,
        );

        for d in 0..table.shape().0 {
            for p in 0..points.shape().0 {
                for j in 0..self.value_size {
                    for b in 0..self.dim {
                        let value = data.get_mut(d, p, b, j).unwrap();
                        *value = 0.0;
                        for i in 0..table.shape().1 {
                            *value += *self.coefficients.get(b, j, i).unwrap()
                                * *table.get_mut(d, i, p).unwrap();
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
