//! Constructor for a single node Domain.
use num::Float;

use crate::types::{domain::Domain, point::PointType};

impl<T: Float + Default> Domain<T> {
    /// Compute the domain defined by a set of points on a local node. When defined by a set of points
    /// The domain adds a small threshold such that no points lie on the actual edge of the domain to
    /// ensure correct Morton encoding.
    ///
    /// # Arguments
    /// * `points` - A slice of point coordinates, expected in column major order  [x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N].
    pub fn from_local_points(points: &[PointType<T>]) -> Domain<T> {
        // Increase size of bounding box to capture all points
        let err = T::from(1e-5).unwrap();
        // TODO: Should be parametrised by dimension
        let dim = 3;
        let npoints = points.len() / dim;
        let x = points[0..npoints].to_vec();
        let y = points[npoints..2 * npoints].to_vec();
        let z = points[2 * npoints..].to_vec();

        let max_x = x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_y = y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_z = z.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let min_x = x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_y = y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_z = z.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        // Find maximum dimension, this will define the size of the boxes in the domain
        let diameter_x = (*max_x - *min_x).abs();
        let diameter_y = (*max_y - *min_y).abs();
        let diameter_z = (*max_z - *min_z).abs();

        // Want a cubic box to place everything in
        let diameter = diameter_x.max(diameter_y).max(diameter_z);
        let two = T::from(2.0).unwrap();
        let diameter = [
            diameter + two * err,
            diameter + two * err,
            diameter + two * err,
        ];

        // The origin is defined by the minimum point
        let origin = [*min_x - err, *min_y - err, *min_z - err];

        Domain { origin, diameter }
    }

    pub fn union(&self, other: &Self) -> Self {

        // Find minimum origin
        let min_x = self.origin[0].min(other.origin[0]);
        let min_y = self.origin[1].min(other.origin[1]);
        let min_z = self.origin[2].min(other.origin[2]);

        let min_origin = [min_x, min_y, min_z];

        // Find maximum diameter (+max origin)
        let max_x = self.diameter[0].max(other.diameter[0]);
        let max_y = self.diameter[0].max(other.diameter[0]);
        let max_z = self.diameter[0].max(other.diameter[0]);

        let max_diameter = [max_x, max_y, max_z];

        Domain {
            origin: min_origin,
            diameter: max_diameter,
        }
    }


    /// Construct a domain a user specified origin and diameter.
    ///
    /// # Arguments
    /// * `origin` - The point from which to construct a cuboid domain.
    /// * `diameter` - The diameter along each axis of the domain.
    pub fn new(origin: &[T; 3], diameter: &[T; 3]) -> Self {
        Domain {
            origin: *origin,
            diameter: *diameter,
        }
    }
}

#[cfg(test)]
mod test {
    use bempp_traits::types::Scalar;
    use rlst_dense::traits::{RawAccess, Shape};

    use crate::implementations::helpers::{points_fixture, points_fixture_col, PointsMat};

    use super::*;

    fn test_compute_bounds<T>(points: PointsMat<T>)
    where
        T: Float + Default + Scalar,
    {
        let domain = Domain::from_local_points(points.data());

        // Test that the domain remains cubic
        assert!(domain.diameter.iter().all(|&x| x == domain.diameter[0]));

        // Test that all local points are contained within the local domain
        let npoints = points.shape()[0];
        for i in 0..npoints {
            let point = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];

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

    #[test]
    fn test_bounds() {
        let npoints = 10000;

        // Test points in positive octant only
        let points = points_fixture::<f64>(npoints, None, None);
        test_compute_bounds(points);

        // Test points in positive and negative octants
        let points = points_fixture::<f64>(npoints, Some(-1.), Some(1.));
        test_compute_bounds(points);

        // Test rectangular distributions of points
        let points = points_fixture_col::<f64>(npoints);
        test_compute_bounds(points);
    }
}
