//! Finite Element definitions

use crate::cell::create_cell;
use crate::polynomials::{legendre_shape, polynomial_count, tabulate_legendre_polynomials};
use bempp_tools::arrays::{AdjacencyList, Array2D, Array3D};
use bempp_traits::arrays::{AdjacencyListAccess, Array2DAccess, Array3DAccess, Array4DAccess};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{ElementFamily, FiniteElement, MapType};
use rlst_algorithms::linalg::LinAlg;
use rlst_algorithms::traits::inverse::Inverse;
use rlst_dense::{RandomAccessMut, RandomAccessByRef};
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
    discontinuous: bool,
    dim: usize,
    coefficients: Array3D<f64>,
    entity_dofs: [AdjacencyList<usize>; 4],
}

impl CiarletElement {
    pub fn create(
        family: ElementFamily,
        cell_type: ReferenceCellType,
        degree: usize,
        value_shape: Vec<usize>,
        wcoeffs: Array3D<f64>,
        x: [Vec<Array2D<f64>>; 4],
        m: [Vec<Array3D<f64>>; 4],
        map_type: MapType,
        discontinuous: bool,
        highest_degree: usize,
    ) -> CiarletElement {
        let mut dim = 0;
        let mut npts = 0;
        for emats in &m {
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

        for matrices in &m {
            for mat in matrices {
                if mat.shape().1 != value_size {
                    panic!("Incompatible value size");
                }
            }
        }

        let new_x = if discontinuous {
            let mut new_x = [vec![], vec![], vec![], vec![]];
            let mut pn = 0;
            let mut all_pts = Array2D::<f64>::new((npts, tdim));
            for i in 0..tdim + 1 {
                for pts in &x[i] {
                    new_x[i].push(Array2D::<f64>::new((0, tdim)));
                    for j in 0..pts.shape().0 {
                        for k in 0..tdim {
                            *all_pts.get_mut(pn + j, k).unwrap() = *pts.get(j, k).unwrap();
                        }
                    }
                    pn += pts.shape().0;
                }
            }
            new_x[tdim].push(all_pts);
            new_x
        } else {
            x
        };
        let new_m = if discontinuous {
            let mut new_m = [vec![], vec![], vec![], vec![]];
            let mut pn = 0;
            let mut dn = 0;
            let mut all_mat = Array3D::<f64>::new((dim, value_size, npts));
            for i in 0..tdim + 1 {
                for mat in &m[i] {
                    new_m[i].push(Array3D::<f64>::new((0, value_size, 0)));
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
            new_m[tdim].push(all_mat);
            new_m
        } else {
            m
        };

        // Compute the dual matrix
        let pdim = polynomial_count(cell_type, highest_degree);
        let mut d_matrix = Array3D::<f64>::new((value_size, pdim, dim));

        let mut dof = 0;
        for d in 0..4 {
            for (e, pts) in new_x[d].iter().enumerate() {
                if pts.shape().0 > 0 {
                    let mut table = Array3D::<f64>::new((1, pdim, pts.shape().0));
                    tabulate_legendre_polynomials(cell_type, pts, highest_degree, 0, &mut table);
                    let mat = &new_m[d][e];
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
                        *entry += *wcoeffs.get(i, k, l).unwrap() * *d_matrix.get(k, l, j).unwrap();
                    }
                }
            }
        }

        let inverse = dual_matrix.linalg().inverse().unwrap();
        let mut coefficients = Array3D::<f64>::new((dim, value_size, dim / value_size));
        for i in 0..dim {
            for j in 0..value_size {
                for k in 0..dim / value_size {
                    *coefficients.get_mut(i, j, k).unwrap() = *inverse.get(i,  j * dim / value_size + k).unwrap()
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
            for pts in &new_x[i] {
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
            discontinuous,
            dim,
            coefficients,
            entity_dofs,
        }
    }

    // TODO: move this to the FiniteElement trait
    pub fn value_shape(&self) -> &Vec<usize> {
        &self.value_shape
    }

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
                            *value +=
                                *self.coefficients.get(b, j, i).unwrap()
                                    * *table.get_mut(d, i, p).unwrap();
                            println!("data({d}, {p}, {b}, {j}) += {} * {}", *self.coefficients.get(b, j, i).unwrap(), *table.get_mut(d, i, p).unwrap());
                            println!("value = {}", *value);
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

pub struct OldCiarletElement {
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

impl FiniteElement for OldCiarletElement {
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
        _nderivs: usize,
        data: &mut impl Array4DAccess<f64>,
    ) {
        for deriv in 0..data.shape().0 {
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
                                    0.0,
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
                                    0.0,
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
                        for (k, e) in evals.iter().enumerate() {
                            *data.get_mut(deriv, pt, i, j).unwrap() +=
                                unsafe { *self.coefficients.get_unchecked(i, j, k) } * *e;
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
) -> OldCiarletElement {
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
