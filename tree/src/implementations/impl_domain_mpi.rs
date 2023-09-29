use memoffset::offset_of;
use mpi::{
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    topology::UserCommunicator,
    traits::*,
    Address,
};

use crate::types::{domain::Domain, point::PointType};

unsafe impl Equivalence for Domain {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(Domain, origin) as Address,
                offset_of!(Domain, diameter) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(3, &PointType::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(3, &PointType::equivalent_datatype()).as_ref(),
            ],
        )
    }
}

impl Domain {
    /// Compute the points domain over all nodes by computing `local' domains on each MPI process, communicating the bounds
    /// globally and using the local domains to create a globally defined domain. Relies on an `all to all` communication.
    ///
    /// # Arguments
    /// * `local_points` - A slice of point coordinates, expected in column major order  [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
    /// * `comm` - An MPI (User) communicator over which the domain is defined.
    pub fn from_global_points(local_points: &[PointType], comm: &UserCommunicator) -> Domain {
        let size = comm.size();

        let local_domain = Domain::from_local_points(local_points);
        let local_bounds: Vec<Domain> = vec![local_domain; size as usize];
        let mut buffer = vec![Domain::default(); size as usize];

        comm.all_to_all_into(&local_bounds, &mut buffer[..]);

        // Find minimum origin
        let min_x = buffer
            .iter()
            .min_by(|a, b| a.origin[0].partial_cmp(&b.origin[0]).unwrap())
            .unwrap()
            .origin[0];
        let min_y = buffer
            .iter()
            .min_by(|a, b| a.origin[1].partial_cmp(&b.origin[1]).unwrap())
            .unwrap()
            .origin[1];
        let min_z = buffer
            .iter()
            .min_by(|a, b| a.origin[2].partial_cmp(&b.origin[2]).unwrap())
            .unwrap()
            .origin[2];

        let min_origin = [min_x, min_y, min_z];

        // Find maximum diameter (+max origin)
        let max_x = buffer
            .iter()
            .max_by(|a, b| a.diameter[0].partial_cmp(&b.diameter[0]).unwrap())
            .unwrap()
            .diameter[0];
        let max_y = buffer
            .iter()
            .max_by(|a, b| a.diameter[1].partial_cmp(&b.diameter[1]).unwrap())
            .unwrap()
            .diameter[1];
        let max_z = buffer
            .iter()
            .max_by(|a, b| a.diameter[2].partial_cmp(&b.diameter[2]).unwrap())
            .unwrap()
            .diameter[2];

        let max_diameter = [max_x, max_y, max_z];

        Domain {
            origin: min_origin,
            diameter: max_diameter,
        }
    }
}
