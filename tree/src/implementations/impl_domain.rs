use crate::types::{domain::Domain, point::PointType};

impl Domain {
    /// Compute the domain defined by a set of points on a local node. When defined by a set of points
    /// The domain adds a small threshold such that no points lie on the actual edge of the domain to
    /// ensure correct Morton Encoding.
    pub fn from_local_points(points: &[PointType]) -> Domain {
        // Increase size of bounding box to capture all points
        let err: f64 = 0.001;
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

        // let max_x = points
        //     .iter()
        //     .max_by(|a, b| a[0].partial_cmp(&b[0]).unwrap())
        //     .unwrap()[0];
        // let max_y = points
        //     .iter()
        //     .max_by(|a, b| a[1].partial_cmp(&b[1]).unwrap())
        //     .unwrap()[1];
        // let max_z = points
        //     .iter()
        //     .max_by(|a, b| a[2].partial_cmp(&b[2]).unwrap())
        //     .unwrap()[2];

        // let min_x = points
        //     .iter()
        //     .min_by(|a, b| a[0].partial_cmp(&b[0]).unwrap())
        //     .unwrap()[0];
        // let min_y = points
        //     .iter()
        //     .min_by(|a, b| a[1].partial_cmp(&b[1]).unwrap())
        //     .unwrap()[1];
        // let min_z = points
        //     .iter()
        //     .min_by(|a, b| a[2].partial_cmp(&b[2]).unwrap())
        //     .unwrap()[2];

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
    use rlst::common::traits::ColumnMajorIterator;
    use rlst::dense::{base_matrix::BaseMatrix, rlst_mat, Dynamic, Matrix, VectorContainer, RawAccess};

    fn points_fixture(
        npoints: usize,
    ) -> Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>
    {
        // Generate a set of randomly distributed points
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0_f64..1.0_f64);
        let mut points = rlst_mat![f64, (npoints, 3)];

        for i in 0..npoints {
            points[[i, 0]] = between.sample(&mut range);
            points[[i, 1]] = between.sample(&mut range);
            points[[i, 2]] = between.sample(&mut range);
        }

        points
    }

    #[test]
    fn test_compute_bounds() {
        let npoints = 10000;
        let points = points_fixture(npoints);
        let domain = Domain::from_local_points(&points.data());

        // Test that all local points are contained within the local domain
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
}
