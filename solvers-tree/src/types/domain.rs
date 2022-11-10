//! Data structures and methods for defining the computational domain.
use memoffset::offset_of;

use mpi::{
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    topology::UserCommunicator,
    traits::*,
    Address,
};

use crate::types::point::PointType;

/// A domain is defined by an origin coordinate, and its diameter along all three Cartesian axes.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct Domain {
    pub origin: [PointType; 3],
    pub diameter: [PointType; 3],
}

impl Domain {
    /// Compute the domain defined by a set of points on a local node.
    pub fn from_local_points(points: &[[PointType; 3]]) -> Domain {
        let max_x = points
            .iter()
            .max_by(|a, b| a[0].partial_cmp(&b[0]).unwrap())
            .unwrap()[0];
        let max_y = points
            .iter()
            .max_by(|a, b| a[1].partial_cmp(&b[1]).unwrap())
            .unwrap()[1];
        let max_z = points
            .iter()
            .max_by(|a, b| a[2].partial_cmp(&b[2]).unwrap())
            .unwrap()[2];

        let min_x = points
            .iter()
            .min_by(|a, b| a[0].partial_cmp(&b[0]).unwrap())
            .unwrap()[0];
        let min_y = points
            .iter()
            .min_by(|a, b| a[1].partial_cmp(&b[1]).unwrap())
            .unwrap()[1];
        let min_z = points
            .iter()
            .min_by(|a, b| a[2].partial_cmp(&b[2]).unwrap())
            .unwrap()[2];

        Domain {
            origin: [min_x, min_y, min_z],
            diameter: [max_x - min_x, max_y - min_y, max_z - min_z],
        }
    }

    /// Compute the points domain over all nodes.
    pub fn from_global_points(local_points: &[[PointType; 3]], comm: &UserCommunicator) -> Domain {
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

        let max_diameter = [
            max_x,
            max_y,
            max_z,
        ];

        Domain {
            origin: min_origin,
            diameter: max_diameter,
        }
    }
}

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

#[cfg(test)]
mod tests {

    use super::*;

    use rand::prelude::*;
    use rand::SeedableRng;

    use crate::{
        constants::{NCRIT, ROOT},
        distributed::DistributedTree,
        types::{domain::Domain, morton::MortonKey},
    };

    const NPOINTS: u64 = 100000;

    #[test]
    fn test_compute_bounds() {
        // Generate a set of randomly distributed points
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0 as f64..1.0 as f64);
        let mut points = Vec::new();

        for _ in 0..NPOINTS {
            points.push([
                between.sample(&mut range),
                between.sample(&mut range),
                between.sample(&mut range),
            ])
        }

        let domain = Domain::from_local_points(&points);

        // Test that all local points are contained within the local domain
        for point in points {
            assert!(
                domain.origin[0] <= point[0] && point[0] <= domain.origin[0] + domain.diameter[0]
            );
            assert!(
                domain.origin[1] <= point[1] && point[1] <= domain.origin[1] + domain.diameter[1]
            );
            assert!(
                domain.origin[2] <= point[2] && point[2] <= domain.origin[2] + domain.diameter[2]
            );
        }
    }
}
