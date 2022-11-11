//! Wrappers for methods on Domains.
use mpi::{ffi::MPI_Comm, topology::UserCommunicator, traits::*};

use crate::types::{domain::Domain, point::PointType};

#[no_mangle]
pub unsafe extern "C" fn domain_from_local_points(
    p_points: *const [PointType; 3],
    npoints: usize,
) -> *mut Domain {
    let points = std::slice::from_raw_parts(p_points, npoints);
    let domain = Domain::from_local_points(points);

    Box::into_raw(Box::new(domain))
}

#[no_mangle]
pub unsafe extern "C" fn domain_from_global_points(
    p_points: *const [PointType; 3],
    npoints: usize,
    comm: *mut usize,
) -> *mut Domain {
    let points = std::slice::from_raw_parts(p_points, npoints);
    let comm = std::mem::ManuallyDrop::new(UserCommunicator::from_raw(*comm as MPI_Comm).unwrap());
    let domain = Domain::from_global_points(points, &comm);

    Box::into_raw(Box::new(domain))
}
