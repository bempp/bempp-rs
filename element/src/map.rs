//! Push forward and pull back maps

use crate::cell::*;
use crate::element::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum MapType {
    Identity = 0,
    CovariantPiola = 1,
    ContravariantPiola = 2,
    L2Piola = 3,
}

pub fn identity_push_forward<'a, 'b, F: FiniteElement, F2: FiniteElement, C: ReferenceCell>(
    data: &mut TabulatedData<'a, F>,
    points: &[f64],
    geometry: &PhysicalCell<'b, F2, C>,
) {
    assert_eq!(data.deriv_count(), 1);
}

pub fn identity_pull_back<'a, 'b, F: FiniteElement, F2: FiniteElement, C: ReferenceCell>(
    data: &mut TabulatedData<'a, F>,
    points: &[f64],
    geometry: &PhysicalCell<'b, F2, C>,
) {
    assert_eq!(data.deriv_count(), 1);
}

pub fn contravariant_piola_push_forward<
    'a,
    'b,
    F: FiniteElement,
    F2: FiniteElement,
    C: ReferenceCell,
>(
    data: &mut TabulatedData<'a, F>,
    points: &[f64],
    geometry: &PhysicalCell<'b, F2, C>,
) {
    assert_eq!(data.deriv_count(), 1);

    if geometry.tdim() == 2 && geometry.gdim() == 2 {
        let gdim = geometry.gdim();
        let npts = points.len() / gdim;
        let geometry_npts = geometry.npts();
        let nbasis = data.basis_count();

        // TODO: get rid of memory assignment inside this function
        let mut derivs = TabulatedData::new(geometry.coordinate_element(), 1, npts);
        geometry
            .coordinate_element()
            .tabulate(&points, 1, &mut derivs);

        let mut j = vec![0.0, 0.0, 0.0, 0.0];
        let mut det_j = 0.0;

        let mut temp_data = vec![0.0, 0.0];

        for p in 0..npts {
            j[0] = 0.0;
            j[1] = 0.0;
            j[2] = 0.0;
            j[3] = 0.0;
            for gp in 0..geometry_npts {
                j[0] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim];
                j[1] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim];
                j[2] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
                j[3] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
            }
            det_j = j[0] * j[3] - j[1] * j[2];

            for i in 0..nbasis {
                temp_data[0] = *data.get(0, p, i, 0);
                temp_data[1] = *data.get(0, p, i, 1);

                *data.get_mut(0, p, i, 0) = (j[0] * temp_data[0] + j[1] * temp_data[1]) / det_j;
                *data.get_mut(0, p, i, 1) = (j[2] * temp_data[0] + j[3] * temp_data[1]) / det_j;
            }
        }
    } else {
        unimplemented!("push_forward not yet implemented for this element");
    }
}

pub fn contravariant_piola_pull_back<
    'a,
    'b,
    F: FiniteElement,
    F2: FiniteElement,
    C: ReferenceCell,
>(
    data: &mut TabulatedData<'a, F>,
    points: &[f64],
    geometry: &PhysicalCell<'b, F2, C>,
) {
    assert_eq!(data.deriv_count(), 1);

    if geometry.tdim() == 2 && geometry.gdim() == 2 {
        let gdim = geometry.gdim();
        let npts = points.len() / gdim;
        let geometry_npts = geometry.npts();
        let nbasis = data.basis_count();

        // TODO: get rid of memory assignment inside this function
        let mut derivs = TabulatedData::new(geometry.coordinate_element(), 1, npts);
        geometry
            .coordinate_element()
            .tabulate(&points, 1, &mut derivs);

        let mut jinv = vec![0.0, 0.0, 0.0, 0.0];
        let mut det_j = 0.0;

        let mut temp_data = vec![0.0, 0.0];

        for p in 0..npts {
            jinv[0] = 0.0;
            jinv[1] = 0.0;
            jinv[2] = 0.0;
            jinv[3] = 0.0;
            for gp in 0..geometry_npts {
                jinv[3] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim];
                jinv[1] -= derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim];
                jinv[2] -= derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
                jinv[0] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
            }

            for i in 0..nbasis {
                temp_data[0] = *data.get(0, p, i, 0);
                temp_data[1] = *data.get(0, p, i, 1);

                *data.get_mut(0, p, i, 0) = jinv[0] * temp_data[0] + jinv[1] * temp_data[1];
                *data.get_mut(0, p, i, 1) = jinv[2] * temp_data[0] + jinv[3] * temp_data[1];
            }
        }
    } else {
        unimplemented!("push_forward not yet implemented for this element");
    }
}

pub fn covariant_piola_push_forward<
    'a,
    'b,
    F: FiniteElement,
    F2: FiniteElement,
    C: ReferenceCell,
>(
    data: &mut TabulatedData<'a, F>,
    points: &[f64],
    geometry: &PhysicalCell<'b, F2, C>,
) {
    assert_eq!(data.deriv_count(), 1);

    if geometry.tdim() == 2 && geometry.gdim() == 2 {
        let gdim = geometry.gdim();
        let npts = points.len() / gdim;
        let geometry_npts = geometry.npts();
        let nbasis = data.basis_count();

        // TODO: get rid of memory assignment inside this function
        let mut derivs = TabulatedData::new(geometry.coordinate_element(), 1, npts);
        geometry
            .coordinate_element()
            .tabulate(&points, 1, &mut derivs);

        let mut jinv_t = vec![0.0, 0.0, 0.0, 0.0];
        let mut det_j = 0.0;

        let mut temp_data = vec![0.0, 0.0];

        for p in 0..npts {
            jinv_t[0] = 0.0;
            jinv_t[1] = 0.0;
            jinv_t[2] = 0.0;
            jinv_t[3] = 0.0;
            for gp in 0..geometry_npts {
                jinv_t[3] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim];
                jinv_t[2] -= derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim];
                jinv_t[1] -= derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
                jinv_t[0] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
            }
            det_j = jinv_t[0] * jinv_t[3] - jinv_t[1] * jinv_t[2];
            jinv_t[0] /= det_j;
            jinv_t[1] /= det_j;
            jinv_t[2] /= det_j;
            jinv_t[3] /= det_j;

            for i in 0..nbasis {
                temp_data[0] = *data.get(0, p, i, 0);
                temp_data[1] = *data.get(0, p, i, 1);

                *data.get_mut(0, p, i, 0) = jinv_t[0] * temp_data[0] + jinv_t[1] * temp_data[1];
                *data.get_mut(0, p, i, 1) = jinv_t[2] * temp_data[0] + jinv_t[3] * temp_data[1];
            }
        }
    } else {
        unimplemented!("push_forward not yet implemented for this element");
    }
}

pub fn covariant_piola_pull_back<'a, 'b, F: FiniteElement, F2: FiniteElement, C: ReferenceCell>(
    data: &mut TabulatedData<'a, F>,
    points: &[f64],
    geometry: &PhysicalCell<'b, F2, C>,
) {
    assert_eq!(data.deriv_count(), 1);

    if geometry.tdim() == 2 && geometry.gdim() == 2 {
        let gdim = geometry.gdim();
        let npts = points.len() / gdim;
        let geometry_npts = geometry.npts();
        let nbasis = data.basis_count();

        // TODO: get rid of memory assignment inside this function
        let mut derivs = TabulatedData::new(geometry.coordinate_element(), 1, npts);
        geometry
            .coordinate_element()
            .tabulate(&points, 1, &mut derivs);

        let mut j_t = vec![0.0, 0.0, 0.0, 0.0];
        let mut det_j = 0.0;

        let mut temp_data = vec![0.0, 0.0];

        for p in 0..npts {
            j_t[0] = 0.0;
            j_t[1] = 0.0;
            j_t[2] = 0.0;
            j_t[3] = 0.0;
            for gp in 0..geometry_npts {
                j_t[0] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim];
                j_t[2] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim];
                j_t[1] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
                j_t[3] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
            }

            for i in 0..nbasis {
                temp_data[0] = *data.get(0, p, i, 0);
                temp_data[1] = *data.get(0, p, i, 1);

                *data.get_mut(0, p, i, 0) = j_t[0] * temp_data[0] + j_t[1] * temp_data[1];
                *data.get_mut(0, p, i, 1) = j_t[2] * temp_data[0] + j_t[3] * temp_data[1];
            }
        }
    } else {
        unimplemented!("push_forward not yet implemented for this element");
    }
}

pub fn l2_piola_push_forward<'a, 'b, F: FiniteElement, F2: FiniteElement, C: ReferenceCell>(
    data: &mut TabulatedData<'a, F>,
    points: &[f64],
    geometry: &PhysicalCell<'b, F2, C>,
) {
    assert_eq!(data.deriv_count(), 1);

    if geometry.tdim() == 2 && geometry.gdim() == 2 {
        let gdim = geometry.gdim();
        let npts = points.len() / gdim;
        let geometry_npts = geometry.npts();
        let nbasis = data.basis_count();

        // TODO: get rid of memory assignment inside this function
        let mut derivs = TabulatedData::new(geometry.coordinate_element(), 1, npts);
        geometry
            .coordinate_element()
            .tabulate(&points, 1, &mut derivs);

        let mut j = vec![0.0, 0.0, 0.0, 0.0];
        let mut det_j = 0.0;

        let mut temp_data = vec![0.0, 0.0];

        for p in 0..npts {
            j[0] = 0.0;
            j[1] = 0.0;
            j[2] = 0.0;
            j[3] = 0.0;
            for gp in 0..geometry_npts {
                j[0] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim];
                j[1] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim];
                j[2] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
                j[3] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
            }
            det_j = j[0] * j[3] - j[1] * j[2];

            for i in 0..nbasis {
                *data.get_mut(0, p, i, 0) *= det_j;
            }
        }
    } else {
        unimplemented!("push_forward not yet implemented for this element");
    }
}

pub fn l2_piola_pull_back<'a, 'b, F: FiniteElement, F2: FiniteElement, C: ReferenceCell>(
    data: &mut TabulatedData<'a, F>,
    points: &[f64],
    geometry: &PhysicalCell<'b, F2, C>,
) {
    assert_eq!(data.deriv_count(), 1);

    if geometry.tdim() == 2 && geometry.gdim() == 2 {
        let gdim = geometry.gdim();
        let npts = points.len() / gdim;
        let geometry_npts = geometry.npts();
        let nbasis = data.basis_count();

        // TODO: get rid of memory assignment inside this function
        let mut derivs = TabulatedData::new(geometry.coordinate_element(), 1, npts);
        geometry
            .coordinate_element()
            .tabulate(&points, 1, &mut derivs);

        let mut j = vec![0.0, 0.0, 0.0, 0.0];
        let mut det_j = 0.0;

        let mut temp_data = vec![0.0, 0.0];

        for p in 0..npts {
            j[0] = 0.0;
            j[1] = 0.0;
            j[2] = 0.0;
            j[3] = 0.0;
            for gp in 0..geometry_npts {
                j[0] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim];
                j[1] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim];
                j[2] += derivs.get(1, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
                j[3] += derivs.get(2, p, gp, 0) * geometry.vertices()[gp * gdim + 1];
            }
            det_j = j[0] * j[3] - j[1] * j[2];

            for i in 0..nbasis {
                *data.get_mut(0, p, i, 0) /= det_j;
            }
        }
    } else {
        unimplemented!("push_forward not yet implemented for this element");
    }
}

#[cfg(test)]
mod test {
    use crate::cell::*;
    use crate::element::*;
    use crate::map::*;
    use approx::*;

    #[test]
    fn test_identity() {
        let e = LagrangeElementTriangleDegree1 {};
        let mut data = TabulatedData::new(&e, 0, 1);

        *data.get_mut(0, 0, 0, 0) = 0.5;
        *data.get_mut(0, 0, 1, 0) = 0.4;
        *data.get_mut(0, 0, 2, 0) = 0.3;

        let coord_e = LagrangeElementTriangleDegree1 {};
        let ref_cell = Triangle {};
        let vertices = vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let geometry = PhysicalCell::new(&ref_cell, &vertices, &coord_e, 2);

        let pts = vec![0.3, 0.3];

        identity_push_forward(&mut data, &pts, &geometry);

        assert_relative_eq!(*data.get(0, 0, 0, 0), 0.5);
        assert_relative_eq!(*data.get(0, 0, 1, 0), 0.4);
        assert_relative_eq!(*data.get(0, 0, 2, 0), 0.3);

        identity_pull_back(&mut data, &pts, &geometry);

        assert_relative_eq!(*data.get(0, 0, 0, 0), 0.5);
        assert_relative_eq!(*data.get(0, 0, 1, 0), 0.4);
        assert_relative_eq!(*data.get(0, 0, 2, 0), 0.3);
    }

    #[test]
    fn test_contravariant_piola() {
        let e = RaviartThomasElementTriangleDegree1 {};
        let mut data = TabulatedData::new(&e, 0, 1);

        *data.get_mut(0, 0, 0, 0) = 0.5;
        *data.get_mut(0, 0, 0, 1) = 0.4;
        *data.get_mut(0, 0, 1, 0) = 0.3;
        *data.get_mut(0, 0, 1, 1) = 0.2;
        *data.get_mut(0, 0, 2, 0) = 0.1;
        *data.get_mut(0, 0, 2, 1) = 0.0;

        let coord_e = LagrangeElementTriangleDegree1 {};
        let ref_cell = Triangle {};
        let vertices = vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let geometry = PhysicalCell::new(&ref_cell, &vertices, &coord_e, 2);

        let pts = vec![0.3, 0.3];

        contravariant_piola_push_forward(&mut data, &pts, &geometry);

        assert_relative_eq!(*data.get(0, 0, 0, 0), 0.65);
        assert_relative_eq!(*data.get(0, 0, 0, 1), -0.25);
        assert_relative_eq!(*data.get(0, 0, 1, 0), 0.35);
        assert_relative_eq!(*data.get(0, 0, 1, 1), -0.15);
        assert_relative_eq!(*data.get(0, 0, 2, 0), 0.05);
        assert_relative_eq!(*data.get(0, 0, 2, 1), -0.05);

        contravariant_piola_pull_back(&mut data, &pts, &geometry);

        assert_relative_eq!(*data.get(0, 0, 0, 0), 0.5);
        assert_relative_eq!(*data.get(0, 0, 0, 1), 0.4);
        assert_relative_eq!(*data.get(0, 0, 1, 0), 0.3);
        assert_relative_eq!(*data.get(0, 0, 1, 1), 0.2);
        assert_relative_eq!(*data.get(0, 0, 2, 0), 0.1);
        assert_relative_eq!(*data.get(0, 0, 2, 1), 0.0);
    }

    #[test]
    fn test_covariant_piola() {
        let e = RaviartThomasElementTriangleDegree1 {};
        let mut data = TabulatedData::new(&e, 0, 1);

        *data.get_mut(0, 0, 0, 0) = 0.5;
        *data.get_mut(0, 0, 0, 1) = 0.4;
        *data.get_mut(0, 0, 1, 0) = 0.3;
        *data.get_mut(0, 0, 1, 1) = 0.2;
        *data.get_mut(0, 0, 2, 0) = 0.1;
        *data.get_mut(0, 0, 2, 1) = 0.0;

        let coord_e = LagrangeElementTriangleDegree1 {};
        let ref_cell = Triangle {};
        let vertices = vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let geometry = PhysicalCell::new(&ref_cell, &vertices, &coord_e, 2);

        let pts = vec![0.3, 0.3];

        covariant_piola_push_forward(&mut data, &pts, &geometry);

        assert_relative_eq!(*data.get(0, 0, 0, 0), 0.2);
        assert_relative_eq!(*data.get(0, 0, 0, 1), -0.3);
        assert_relative_eq!(*data.get(0, 0, 1, 0), 0.1);
        assert_relative_eq!(*data.get(0, 0, 1, 1), -0.2);
        assert_relative_eq!(*data.get(0, 0, 2, 0), 0.0);
        assert_relative_eq!(*data.get(0, 0, 2, 1), -0.1);

        covariant_piola_pull_back(&mut data, &pts, &geometry);

        assert_relative_eq!(*data.get(0, 0, 0, 0), 0.5);
        assert_relative_eq!(*data.get(0, 0, 0, 1), 0.4);
        assert_relative_eq!(*data.get(0, 0, 1, 0), 0.3);
        assert_relative_eq!(*data.get(0, 0, 1, 1), 0.2);
        assert_relative_eq!(*data.get(0, 0, 2, 0), 0.1);
        assert_relative_eq!(*data.get(0, 0, 2, 1), 0.0);
    }

    #[test]
    fn test_l2_piola() {
        let e = LagrangeElementTriangleDegree1 {};
        let mut data = TabulatedData::new(&e, 0, 1);

        *data.get_mut(0, 0, 0, 0) = 0.5;
        *data.get_mut(0, 0, 1, 0) = 0.4;
        *data.get_mut(0, 0, 2, 0) = 0.3;

        let coord_e = LagrangeElementTriangleDegree1 {};
        let ref_cell = Triangle {};
        let vertices = vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let geometry = PhysicalCell::new(&ref_cell, &vertices, &coord_e, 2);

        let pts = vec![0.3, 0.3];

        l2_piola_push_forward(&mut data, &pts, &geometry);

        assert_relative_eq!(*data.get(0, 0, 0, 0), 1.0);
        assert_relative_eq!(*data.get(0, 0, 1, 0), 0.8);
        assert_relative_eq!(*data.get(0, 0, 2, 0), 0.6);

        l2_piola_pull_back(&mut data, &pts, &geometry);

        assert_relative_eq!(*data.get(0, 0, 0, 0), 0.5);
        assert_relative_eq!(*data.get(0, 0, 1, 0), 0.4);
        assert_relative_eq!(*data.get(0, 0, 2, 0), 0.3);
    }
}
