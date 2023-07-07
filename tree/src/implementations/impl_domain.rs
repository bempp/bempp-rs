use crate::types::{domain::Domain, point::PointType};

impl Domain {
    /// Compute the domain defined by a set of points on a local node. When defined by a set of points
    /// The domain adds a small threshold such that no points lie on the actual edge of the domain to
    /// ensure correct Morton Encoding.
    pub fn from_local_points(points: &[PointType]) -> Domain {
        // Increase size of bounding box to capture all points
        let err: f64 = 1e-5;
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
        let diameter_x = (max_x - min_x).abs();
        let diameter_y = (max_y - min_y).abs();
        let diameter_z = (max_z - min_z).abs();

        let diameter = diameter_x.max(diameter_y).max(diameter_z);
        let diameter = [
            diameter + 2. * err,
            diameter + 2. * err,
            diameter + 2. * err,
        ];

        // The origin is defined by the minimum point
        let origin = [min_x - err, min_y - err, min_z - err];

        Domain { origin, diameter }
    }
}

#[cfg(test)]
mod test {
    use rlst::dense::{RawAccess, Shape};

    use crate::implementations::helpers::{points_fixture, points_fixture_col, PointsMat};

    use super::*;

    fn test_compute_bounds(points: PointsMat) {
        let domain = Domain::from_local_points(&points.data());

        // Test that the domain remains cubic
        assert!(domain.diameter.iter().all(|&x| x == domain.diameter[0]));

        // Test that all local points are contained within the local domain
        let npoints = points.shape().0;
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
        let points = points_fixture(npoints, None, None);
        test_compute_bounds(points);

        // Test points in positive and negative octants
        let points = points_fixture(npoints, Some(-1.), Some(1.));
        test_compute_bounds(points);

        // Test rectangular distributions of points
        let points = points_fixture_col(npoints);
        test_compute_bounds(points);
    }
}
