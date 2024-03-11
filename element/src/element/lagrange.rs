//! Lagrange elements

use crate::element::{reference_cell, CiarletElement, ElementFamily};
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

/// Create a Lagrange element
pub fn create<T: RlstScalar>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    let dim = polynomial_count(cell_type, degree);
    let tdim = reference_cell::dim(cell_type);
    let mut wcoeffs = rlst_dynamic_array3!(T, [dim, 1, dim]);
    for i in 0..dim {
        *wcoeffs.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
    }

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];
    let entity_counts = reference_cell::entity_counts(cell_type);
    let vertices = reference_cell::vertices::<T::Real>(cell_type);
    if degree == 0 {
        if continuity == Continuity::Continuous {
            panic!("Cannot create continuous degree 0 Lagrange element");
        }
        for (d, counts) in entity_counts.iter().enumerate() {
            for _e in 0..*counts {
                x[d].push(rlst_dynamic_array2!(T, [0, tdim]));
                m[d].push(rlst_dynamic_array3!(T, [0, 1, 0]));
            }
        }
        let mut midp = rlst_dynamic_array2!(T, [1, tdim]);
        let nvertices = entity_counts[0];
        for i in 0..tdim {
            for vertex in &vertices {
                *midp.get_mut([0, i]).unwrap() += T::from(vertex[i]).unwrap();
            }
            *midp.get_mut([0, i]).unwrap() /= T::from(nvertices).unwrap();
        }
        x[tdim].push(midp);
        let mut mentry = rlst_dynamic_array3!(T, [1, 1, 1]);
        *mentry.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
        m[tdim].push(mentry);
    } else {
        let edges = reference_cell::edges(cell_type);
        let faces = reference_cell::faces(cell_type);
        // TODO: GLL points
        for vertex in &vertices {
            let mut pts = rlst_dynamic_array2!(T, [1, tdim]);
            for (i, v) in vertex.iter().enumerate() {
                *pts.get_mut([0, i]).unwrap() = T::from(*v).unwrap();
            }
            x[0].push(pts);
            let mut mentry = rlst_dynamic_array3!(T, [1, 1, 1]);
            *mentry.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
            m[0].push(mentry);
        }
        for e in &edges {
            let mut pts = rlst_dynamic_array2!(T, [degree - 1, tdim]);
            let [vn0, vn1] = e[..] else {
                panic!();
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let mut ident = rlst_dynamic_array3!(T, [degree - 1, 1, degree - 1]);

            for i in 1..degree {
                *ident.get_mut([i - 1, 0, i - 1]).unwrap() = T::from(1.0).unwrap();
                for j in 0..tdim {
                    *pts.get_mut([i - 1, j]).unwrap() = T::from(v0[j]).unwrap()
                        + T::from(i).unwrap() / T::from(degree).unwrap()
                            * T::from(v1[j] - v0[j]).unwrap();
                }
            }
            x[1].push(pts);
            m[1].push(ident);
        }
        for (e, face_type) in reference_cell::entity_types(cell_type)[2]
            .iter()
            .enumerate()
        {
            let npts = match face_type {
                ReferenceCellType::Triangle => {
                    if degree > 2 {
                        (degree - 1) * (degree - 2) / 2
                    } else {
                        0
                    }
                }
                ReferenceCellType::Quadrilateral => (degree - 1).pow(2),
                _ => {
                    panic!("Unsupported face type");
                }
            };
            let mut pts = rlst_dynamic_array2!(T, [npts, tdim]);

            let [vn0, vn1, vn2] = faces[e][..3] else {
                panic!();
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let v2 = &vertices[vn2];

            match face_type {
                ReferenceCellType::Triangle => {
                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree - i0 {
                            for j in 0..tdim {
                                *pts.get_mut([n, j]).unwrap() = T::from(v0[j]).unwrap()
                                    + T::from(i0).unwrap() / T::from(degree).unwrap()
                                        * T::from(v1[j] - v0[j]).unwrap()
                                    + T::from(i1).unwrap() / T::from(degree).unwrap()
                                        * T::from(v2[j] - v0[j]).unwrap();
                            }
                            n += 1;
                        }
                    }
                }
                ReferenceCellType::Quadrilateral => {
                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree {
                            for j in 0..tdim {
                                *pts.get_mut([n, j]).unwrap() = T::from(v0[j]).unwrap()
                                    + T::from(i0).unwrap() / T::from(degree).unwrap()
                                        * T::from(v1[j] - v0[j]).unwrap()
                                    + T::from(i1).unwrap() / T::from(degree).unwrap()
                                        * T::from(v2[j] - v0[j]).unwrap();
                            }
                            n += 1;
                        }
                    }
                }
                _ => {
                    panic!("Unsupported face type.");
                }
            };

            let mut ident = rlst_dynamic_array3!(T, [npts, 1, npts]);
            for i in 0..npts {
                *ident.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
            }
            x[2].push(pts);
            m[2].push(ident);
        }
    }
    CiarletElement::<T>::create(
        cell_type,
        ElementFamily::Lagrange,
        degree,
        vec![],
        wcoeffs,
        x,
        m,
        MapType::Identity,
        continuity,
        degree,
    )
}
