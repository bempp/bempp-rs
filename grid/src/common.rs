//! Functionality common to multiple grid implementations

use crate::traits::Geometry;
use num::Float;
use rlst_dense::types::RlstScalar;
use rlst_dense::{
    array::Array,
    traits::{Shape, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue},
};

/// Compute a physical point
pub fn compute_point<
    T: Float + RlstScalar<Real = T>,
    Table: UnsafeRandomAccessByRef<4, Item = T> + Shape<4>,
>(
    geometry: &impl Geometry<T = T>,
    table: Table,
    cell_index: usize,
    point_index: usize,
    point: &mut [T],
) {
    assert_eq!(geometry.dim(), point.len());

    let cell = geometry.index_map()[cell_index];

    for component in point.iter_mut() {
        *component = T::from(0.0).unwrap();
    }
    for (i, v) in geometry.cell_points(cell).unwrap().iter().enumerate() {
        let t = unsafe { *table.get_unchecked([0, point_index, i, 0]) };
        for (j, component) in point.iter_mut().enumerate() {
            *component += *geometry.coordinate(*v, j).unwrap() * t;
        }
    }
}

/// Compute a Jacobian
pub fn compute_jacobian<
    T: Float + RlstScalar<Real = T>,
    Table: UnsafeRandomAccessByRef<4, Item = T> + Shape<4>,
>(
    geometry: &impl Geometry<T = T>,
    table: Table,
    tdim: usize,
    cell_index: usize,
    point_index: usize,
    jacobian: &mut [T],
) {
    let gdim = geometry.dim();
    assert_eq!(jacobian.len(), gdim * tdim);

    let cell = geometry.index_map()[cell_index];

    for component in jacobian.iter_mut() {
        *component = T::from(0.0).unwrap();
    }
    for (i, v) in geometry.cell_points(cell).unwrap().iter().enumerate() {
        for gd in 0..gdim {
            for td in 0..tdim {
                jacobian[td * gdim + gd] += *geometry.coordinate(*v, gd).unwrap()
                    * unsafe { *table.get_unchecked([1 + td, point_index, i, 0]) };
            }
        }
    }
}

/// Compute a normal from a Jacobian of a cell with topological dimension 2 and geometric dimension 3
pub fn compute_normal_from_jacobian23<T: Float + RlstScalar<Real = T>>(
    jacobian: &[T],
    normal: &mut [T],
) {
    assert_eq!(jacobian.len(), 6);
    assert_eq!(normal.len(), 3);

    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
        normal[i] = jacobian[j] * jacobian[3 + k] - jacobian[k] * jacobian[3 + j];
    }
    let size = RlstScalar::sqrt(normal.iter().map(|&x| RlstScalar::powi(x, 2)).sum::<T>());
    for i in normal.iter_mut() {
        *i /= size;
    }
}

/// Compute a normal from a Jacobian
pub fn compute_normal_from_jacobian<T: Float + RlstScalar<Real = T>>(
    jacobian: &[T],
    normal: &mut [T],
    tdim: usize,
    gdim: usize,
) {
    assert_eq!(jacobian.len(), tdim * gdim);
    assert_eq!(normal.len(), gdim);

    match tdim {
        2 => match gdim {
            3 => compute_normal_from_jacobian23(jacobian, normal),
            _ => {
                unimplemented!("compute_normal_from_jacobian() not implemented for topological dimension {tdim} and geometric dimension: {gdim}");
            }
        },
        _ => {
            unimplemented!("compute_normal_from_jacobian() not implemented for topological dimension {tdim}");
        }
    }
}

/// Compute the determinant of a 1 by 1 matrix
pub fn compute_det11<T: RlstScalar<Real = T>>(jacobian: &[T]) -> T {
    T::abs(jacobian[0])
}
/// Compute the determinant of a 1 by 2 matrix
pub fn compute_det12<T: RlstScalar<Real = T>>(jacobian: &[T]) -> T {
    T::sqrt(jacobian.iter().map(|x| x.powi(2)).sum())
}
/// Compute the determinant of a 1 by 3 matrix
pub fn compute_det13<T: RlstScalar<Real = T>>(jacobian: &[T]) -> T {
    T::sqrt(jacobian.iter().map(|x| x.powi(2)).sum())
}
/// Compute the determinant of a 2 by 2 matrix
pub fn compute_det22<T: RlstScalar<Real = T>>(jacobian: &[T]) -> T {
    T::abs(jacobian[0] * jacobian[3] - jacobian[1] * jacobian[2])
}
/// Compute the determinant of a 2 by 3 matrix
pub fn compute_det23<T: RlstScalar<Real = T>>(jacobian: &[T]) -> T {
    T::sqrt(
        [(1, 2), (2, 0), (0, 1)]
            .iter()
            .map(|(j, k)| {
                (jacobian[*j] * jacobian[3 + *k] - jacobian[*k] * jacobian[3 + *j]).powi(2)
            })
            .sum(),
    )
}
/// Compute the determinant of a 3 by 3 matrix
pub fn compute_det33<T: RlstScalar<Real = T>>(jacobian: &[T]) -> T {
    T::abs(
        [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
            .iter()
            .map(|(i, j, k)| {
                jacobian[*i]
                    * (jacobian[3 + *j] * jacobian[6 + *k] - jacobian[3 + *k] * jacobian[6 + *j])
            })
            .sum(),
    )
}

/// Compute the determinant of a matrix
pub fn compute_det<T: RlstScalar<Real = T>>(jacobian: &[T], tdim: usize, gdim: usize) -> T {
    assert_eq!(jacobian.len(), tdim * gdim);
    match tdim {
        1 => match gdim {
            1 => compute_det11(jacobian),
            2 => compute_det12(jacobian),
            3 => compute_det13(jacobian),
            _ => {
                unimplemented!("compute_det() not implemented for topological dimension {tdim} and geometric dimension: {gdim}");
            }
        },
        2 => match gdim {
            2 => compute_det22(jacobian),
            3 => compute_det23(jacobian),
            _ => {
                unimplemented!("compute_det() not implemented for topological dimension {tdim} and geometric dimension: {gdim}");
            }
        },
        3 => match gdim {
            3 => compute_det33(jacobian),
            _ => {
                unimplemented!("compute_det() not implemented for topological dimension {tdim} and geometric dimension: {gdim}");
            }
        },
        _ => {
            unimplemented!("compute_det() not implemented for topological dimension {tdim}");
        }
    }
}












/// Compute physical points
pub fn compute_points<
    T: Float + RlstScalar<Real = T>,
    Table: UnsafeRandomAccessByRef<4, Item = T> + Shape<4>,
>(
    geometry: &impl Geometry<T = T>,
    table: Table,
    cell_index: usize,
    points: &mut [T],
) {
    let gdim = geometry.dim();
    let npts = table.shape()[1];
    assert_eq!(points.len(), geometry.dim() * npts);

    let cell = geometry.index_map()[cell_index];

    for component in points.iter_mut() {
        *component = T::from(0.0).unwrap();
    }
    for (i, v) in geometry.cell_points(cell).unwrap().iter().enumerate() {
        for point_index in 0..npts {
            let t = unsafe { *table.get_unchecked([0, point_index, i, 0]) };
            for j in 0..gdim {
                points[j * npts + point_index] += *geometry.coordinate(*v, j).unwrap() * t;
            }
        }
    }
}

/// Compute Jacobians
pub fn compute_jacobians<
    T: Float + RlstScalar<Real = T>,
    Table: UnsafeRandomAccessByRef<4, Item = T> + Shape<4>,
>(
    geometry: &impl Geometry<T = T>,
    table: Table,
    tdim: usize,
    cell_index: usize,
    jacobians: &mut [T],
) {
    let gdim = geometry.dim();
    let npts = table.shape()[1];
    assert_eq!(jacobians.len(), gdim * tdim * npts);

    let cell = geometry.index_map()[cell_index];

    for component in jacobians.iter_mut() {
        *component = T::from(0.0).unwrap();
    }
    for (i, v) in geometry.cell_points(cell).unwrap().iter().enumerate() {
        for point_index in 0..npts {
            for gd in 0..gdim {
                for td in 0..tdim {
                    jacobians[(td * gdim + gd) * npts + point_index] += *geometry.coordinate(*v, gd).unwrap()
                        * unsafe { *table.get_unchecked([1 + td, point_index, i, 0]) };
                }
            }
        }
    }
}

/// Compute normals from a Jacobians of a cell with topological dimension 2 and geometric dimension 3
pub fn compute_normals_from_jacobians23<T: Float + RlstScalar<Real = T>>(
    jacobians: &[T],
    normals: &mut [T],
) {
    let npts = normals.len() / 3;
    assert_eq!(jacobians.len(), 6 * npts);
    assert_eq!(normals.len(), 3 * npts);

    for point_index in 0..npts {
        for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            normals[i * npts + point_index] = jacobians[j * npts + point_index] * jacobians[(3 + k) * npts + point_index] - jacobians[k * npts + point_index] * jacobians[(3 + j) * npts + point_index];
        }
        let size = RlstScalar::sqrt((0..3).map(|i| RlstScalar::powi(normals[i*npts + point_index], 2)).sum::<T>());
        for i in 0..3 {
            normals[i * npts + point_index] /= size;
        }
    }
}

/// Compute normals from Jacobians
pub fn compute_normals_from_jacobians<T: Float + RlstScalar<Real = T>>(
    jacobians: &[T],
    normals: &mut [T],
    tdim: usize,
    gdim: usize,
) {
    let npts = normals.len() / gdim;
    assert_eq!(jacobians.len(), tdim * gdim * npts);
    assert_eq!(normals.len(), gdim * npts);

    match tdim {
        2 => match gdim {
            3 => compute_normals_from_jacobians23(jacobians, normals),
            _ => {
                unimplemented!("compute_normals_from_jacobians() not implemented for topological dimension {tdim} and geometric dimension: {gdim}");
            }
        },
        _ => {
            unimplemented!("compute_normals_from_jacobians() not implemented for topological dimension {tdim}");
        }
    }
}

/// Compute determinants of 1 by 1 matrices
pub fn compute_dets11<T: RlstScalar<Real = T>>(jacobian: &[T], jdets: &mut [T]) {
    let npts = jdets.len();
    for (i, jdet) in jdets.iter_mut().enumerate() {
        *jdet = T::abs(jacobian[i]);
    }
}
/// Compute determinants of 1 by 2 matrices
pub fn compute_dets12<T: RlstScalar<Real = T>>(jacobian: &[T], jdets: &mut [T]) {
    let npts = jdets.len();
    for (i, jdet) in jdets.iter_mut().enumerate() {
        *jdet = T::sqrt((0..2).map(|j| jacobian[j*npts + i].powi(2)).sum());
    }
}
/// Compute determinants of 1 by 3 matrices
pub fn compute_dets13<T: RlstScalar<Real = T>>(jacobian: &[T], jdets: &mut [T]) {
    let npts = jdets.len();
    for (i, jdet) in jdets.iter_mut().enumerate() {
        *jdet = T::sqrt((0..3).map(|j| jacobian[j*npts + i].powi(2)).sum());
    }
}
/// Compute determinants of 2 by 2 matrices
pub fn compute_dets22<T: RlstScalar<Real = T>>(jacobian: &[T], jdets: &mut [T]) {
    let npts = jdets.len();
    for (i, jdet) in jdets.iter_mut().enumerate() {
        *jdet = T::abs(jacobian[i] * jacobian[3 * npts + i] - jacobian[npts + i] * jacobian[2 * npts + i]);
    }
}
/// Compute determinants of 2 by 3 matrices
pub fn compute_dets23<T: RlstScalar<Real = T>>(jacobian: &[T], jdets: &mut [T]) {
    let npts = jdets.len();
    for (i, jdet) in jdets.iter_mut().enumerate() {
        *jdet = T::sqrt(
            [(1, 2), (2, 0), (0, 1)]
                .iter()
                .map(|(j, k)| {
                    (jacobian[*j * npts + i] * jacobian[(3 + *k) * npts + i] - jacobian[*k * npts + i] * jacobian[(3 + *j) * npts + i]).powi(2)
                })
                .sum(),
        );
    }
}
/// Compute determinants of 3 by 3 matrices
pub fn compute_dets33<T: RlstScalar<Real = T>>(jacobian: &[T], jdets: &mut [T]) {
    let npts = jdets.len();
    for (i, jdet) in jdets.iter_mut().enumerate() {
        *jdet = T::abs(
            [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
                .iter()
                .map(|(i, j, k)| {
                    jacobian[*i * npts + i]
                        * (jacobian[(3 + *j) * npts + i] * jacobian[(6 + *k) * npts + i] - jacobian[(3 + *k) * npts + i] * jacobian[(6 + *j) * npts + i])
                })
                .sum(),
        );
    }
}

/// Compute determinants of matrices
pub fn compute_dets<T: RlstScalar<Real = T>>(jacobians: &[T], tdim: usize, gdim: usize, jdets: &mut [T]) {
    let npts = jdets.len();
    assert_eq!(jacobians.len(), npts * tdim * gdim);
    match tdim {
        1 => match gdim {
            1 => compute_dets11(jacobians, jdets),
            2 => compute_dets12(jacobians, jdets),
            3 => compute_dets13(jacobians, jdets),
            _ => {
                unimplemented!("compute_dets() not implemented for topological dimension {tdim} and geometric dimension: {gdim}");
            }
        },
        2 => match gdim {
            2 => compute_dets22(jacobians, jdets),
            3 => compute_dets23(jacobians, jdets),
            _ => {
                unimplemented!("compute_dets() not implemented for topological dimension {tdim} and geometric dimension: {gdim}");
            }
        },
        3 => match gdim {
            3 => compute_dets33(jacobians, jdets),
            _ => {
                unimplemented!("compute_dets() not implemented for topological dimension {tdim} and geometric dimension: {gdim}");
            }
        },
        _ => {
            unimplemented!("compute_dets() not implemented for topological dimension {tdim}");
        }
    }
}









/// Compute the diameter of a triangle
pub fn compute_diameter_triangle<
    T: Float + Float + RlstScalar<Real = T>,
    ArrayImpl: UnsafeRandomAccessByValue<1, Item = T> + Shape<1>,
>(
    v0: Array<T, ArrayImpl, 1>,
    v1: Array<T, ArrayImpl, 1>,
    v2: Array<T, ArrayImpl, 1>,
) -> T {
    let a = (v0.view() - v1.view()).norm_2();
    let b = (v0 - v2.view()).norm_2();
    let c = (v1 - v2).norm_2();
    RlstScalar::sqrt((b + c - a) * (a + c - b) * (a + b - c) / (a + b + c))
}

/// Compute the diameter of a quadrilateral
pub fn compute_diameter_quadrilateral<
    T: Float + RlstScalar<Real = T>,
    ArrayImpl: UnsafeRandomAccessByValue<1, Item = T> + Shape<1>,
>(
    v0: Array<T, ArrayImpl, 1>,
    v1: Array<T, ArrayImpl, 1>,
    v2: Array<T, ArrayImpl, 1>,
    v3: Array<T, ArrayImpl, 1>,
) -> T {
    T::max((v0 - v3).norm_2(), (v1 - v2).norm_2())
}
