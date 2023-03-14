use crate::types::{domain::Domain, point::PointType};

impl Domain {
    /// Compute the domain defined by a set of points on a local node. When defined by a set of points
    /// The domain adds a small threshold such that no points lie on the actual edge of the domain to
    /// ensure correct Morton Encoding.
    pub fn from_local_points(points: &[[PointType; 3]]) -> Domain {
        // Increase size of bounding box to capture all points
        let err: f64 = 0.001;
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
            origin: [min_x - err, min_y - err, min_z - err],
            diameter: [
                (max_x - min_x) + 2. * err,
                (max_y - min_y) + 2. * err,
                (max_z - min_z) + 2. * err,
            ],
        }
    }
}

#[cfg(test)]
mod test {

    use rand::prelude::*;
    use rand::SeedableRng;

    use crate::types::domain::Domain;

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
