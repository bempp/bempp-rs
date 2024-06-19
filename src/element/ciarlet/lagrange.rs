//! Lagrange elements

use crate::element::ciarlet::{reference_cell, CiarletElement};
use crate::element::polynomials::polynomial_count;
use crate::traits::element::{Continuity, ElementFamily, MapType};
use crate::traits::types::ReferenceCell;
use rlst::{rlst_dynamic_array2, rlst_dynamic_array3, RandomAccessMut};
use rlst::{LinAlg, RlstScalar};
use std::marker::PhantomData;

/// Create a Lagrange element
pub fn create<T: RlstScalar + LinAlg>(
    cell_type: ReferenceCell,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T> {
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
                x[d].push(rlst_dynamic_array2!(T::Real, [0, tdim]));
                m[d].push(rlst_dynamic_array3!(T, [0, 1, 0]));
            }
        }
        let mut midp = rlst_dynamic_array2!(T::Real, [1, tdim]);
        let nvertices = entity_counts[0];
        for i in 0..tdim {
            for vertex in &vertices {
                *midp.get_mut([0, i]).unwrap() += num::cast::<_, T::Real>(vertex[i]).unwrap();
            }
            *midp.get_mut([0, i]).unwrap() /= num::cast::<_, T::Real>(nvertices).unwrap();
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
            let mut pts = rlst_dynamic_array2!(T::Real, [1, tdim]);
            for (i, v) in vertex.iter().enumerate() {
                *pts.get_mut([0, i]).unwrap() = num::cast::<_, T::Real>(*v).unwrap();
            }
            x[0].push(pts);
            let mut mentry = rlst_dynamic_array3!(T, [1, 1, 1]);
            *mentry.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
            m[0].push(mentry);
        }
        for e in &edges {
            let mut pts = rlst_dynamic_array2!(T::Real, [degree - 1, tdim]);
            let [vn0, vn1] = e[..] else {
                panic!();
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let mut ident = rlst_dynamic_array3!(T, [degree - 1, 1, degree - 1]);

            for i in 1..degree {
                *ident.get_mut([i - 1, 0, i - 1]).unwrap() = T::from(1.0).unwrap();
                for j in 0..tdim {
                    *pts.get_mut([i - 1, j]).unwrap() = num::cast::<_, T::Real>(v0[j]).unwrap()
                        + num::cast::<_, T::Real>(i).unwrap()
                            / num::cast::<_, T::Real>(degree).unwrap()
                            * num::cast::<_, T::Real>(v1[j] - v0[j]).unwrap();
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
                ReferenceCell::Triangle => {
                    if degree > 2 {
                        (degree - 1) * (degree - 2) / 2
                    } else {
                        0
                    }
                }
                ReferenceCell::Quadrilateral => (degree - 1).pow(2),
                _ => {
                    panic!("Unsupported face type");
                }
            };
            let mut pts = rlst_dynamic_array2!(T::Real, [npts, tdim]);

            let [vn0, vn1, vn2] = faces[e][..3] else {
                panic!();
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let v2 = &vertices[vn2];

            match face_type {
                ReferenceCell::Triangle => {
                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree - i0 {
                            for j in 0..tdim {
                                *pts.get_mut([n, j]).unwrap() = num::cast::<_, T::Real>(v0[j])
                                    .unwrap()
                                    + num::cast::<_, T::Real>(i0).unwrap()
                                        / num::cast::<_, T::Real>(degree).unwrap()
                                        * num::cast::<_, T::Real>(v1[j] - v0[j]).unwrap()
                                    + num::cast::<_, T::Real>(i1).unwrap()
                                        / num::cast::<_, T::Real>(degree).unwrap()
                                        * num::cast::<_, T::Real>(v2[j] - v0[j]).unwrap();
                            }
                            n += 1;
                        }
                    }
                }
                ReferenceCell::Quadrilateral => {
                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree {
                            for j in 0..tdim {
                                *pts.get_mut([n, j]).unwrap() = num::cast::<_, T::Real>(v0[j])
                                    .unwrap()
                                    + num::cast::<_, T::Real>(i0).unwrap()
                                        / num::cast::<_, T::Real>(degree).unwrap()
                                        * num::cast::<_, T::Real>(v1[j] - v0[j]).unwrap()
                                    + num::cast::<_, T::Real>(i1).unwrap()
                                        / num::cast::<_, T::Real>(degree).unwrap()
                                        * num::cast::<_, T::Real>(v2[j] - v0[j]).unwrap();
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

/// Lagrange element family
pub struct LagrangeElementFamily<T: LinAlg + RlstScalar> {
    degree: usize,
    continuity: Continuity,
    _t: PhantomData<T>,
}

impl<T: LinAlg + RlstScalar> LagrangeElementFamily<T> {
    /// Create new family
    pub fn new(degree: usize, continuity: Continuity) -> Self {
        Self {
            degree,
            continuity,
            _t: PhantomData,
        }
    }
}

impl<T: LinAlg + RlstScalar> ElementFamily for LagrangeElementFamily<T> {
    type T = T;
    type FiniteElement = CiarletElement<T>;
    fn element(&self, cell_type: ReferenceCell) -> CiarletElement<T> {
        create::<T>(cell_type, self.degree, self.continuity)
    }
}
