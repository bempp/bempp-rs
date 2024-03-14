//! Implementation of an equivalent MPI type for a Domain object, and constructor for distributed Domains.
use memoffset::offset_of;
use mpi::{
    datatype::{UncommittedUserDatatype, UserDatatype},
    topology::UserCommunicator,
    traits::{Buffer, BufferMut, Communicator, CommunicatorCollectives, Equivalence},
    Address,
};
use std::fmt::Debug;

use num::traits::Float;

use crate::types::domain::Domain;

unsafe impl<T: Float + Equivalence + Default> Equivalence for Domain<T> {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(Domain<T>, origin) as Address,
                offset_of!(Domain<T>, diameter) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(3, &T::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(3, &T::equivalent_datatype()).as_ref(),
            ],
        )
    }
}

impl<T: Float + Default + Debug> Domain<T>
where
    [Domain<T>]: BufferMut,
    Vec<Domain<T>>: Buffer,
{
    /// Compute the points domain over all nodes by computing `local' domains on each MPI process, communicating the bounds
    /// globally and using the local domains to create a globally defined domain. Relies on an `all to all` communication.
    ///
    /// # Arguments
    /// * `local_points` - A slice of point coordinates, expected in column major order  [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
    /// * `comm` - An MPI (User) communicator over which the domain is defined.
    pub fn from_global_points(local_points: &[T], comm: &UserCommunicator) -> Domain<T> {
        let size = comm.size();

        let dim = 3;
        let nlocal_points = vec![local_points.len() / dim; size as usize];

        let local_domain = Domain::<T>::from_local_points(local_points);
        let local_bounds: Vec<Domain<T>> = vec![local_domain; size as usize];
        let mut buffer_bounds = vec![Domain::<T>::default(); size as usize];

        comm.all_to_all_into(&local_bounds, &mut buffer_bounds[..]);

        let mut buffer_npoints = vec![0usize; size as usize];
        comm.all_to_all_into(&nlocal_points, &mut buffer_npoints);

        // Find minimum origin
        let min_x = buffer_bounds
            .iter()
            .min_by(|a, b| a.origin[0].partial_cmp(&b.origin[0]).unwrap())
            .unwrap()
            .origin[0];
        let min_y = buffer_bounds
            .iter()
            .min_by(|a, b| a.origin[1].partial_cmp(&b.origin[1]).unwrap())
            .unwrap()
            .origin[1];
        let min_z = buffer_bounds
            .iter()
            .min_by(|a, b| a.origin[2].partial_cmp(&b.origin[2]).unwrap())
            .unwrap()
            .origin[2];

        let min_origin = [min_x, min_y, min_z];

        // Find maximum diameter (+max origin)
        let max_x = buffer_bounds
            .iter()
            .max_by(|a, b| a.diameter[0].partial_cmp(&b.diameter[0]).unwrap())
            .unwrap()
            .diameter[0];
        let max_y = buffer_bounds
            .iter()
            .max_by(|a, b| a.diameter[1].partial_cmp(&b.diameter[1]).unwrap())
            .unwrap()
            .diameter[1];
        let max_z = buffer_bounds
            .iter()
            .max_by(|a, b| a.diameter[2].partial_cmp(&b.diameter[2]).unwrap())
            .unwrap()
            .diameter[2];

        let max_diameter = [max_x, max_y, max_z];

        let nglobal_points: usize = buffer_npoints.iter().sum();

        Domain {
            origin: min_origin,
            diameter: max_diameter,
            npoints: nglobal_points,
        }
    }
}
