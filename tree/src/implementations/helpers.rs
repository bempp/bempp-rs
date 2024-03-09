//! Helper functions used in testing tree implementations, specifically test point generators,
//! as well as helpers for handling surfaces that discretise a box corresponding to a Morton key.

use bempp_traits::types::RlstScalar;
use num::Float;
use rand::prelude::*;
use rand::SeedableRng;

use rlst_dense::{
    array::Array, base_array::BaseArray, data_container::VectorContainer, rlst_dynamic_array2,
};

/// Alias for an rlst container for point data.
pub type PointsMat<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

/// Points fixture for testing, uniformly samples in each axis from min to max.
///
/// # Arguments
/// * `npoints` - The number of points to sample.
/// * `min` - The minumum coordinate value along each axis.
/// * `max` - The maximum coordinate value along each axis.
pub fn points_fixture<T: Float + RlstScalar + rand::distributions::uniform::SampleUniform>(
    npoints: usize,
    min: Option<T>,
    max: Option<T>,
) -> PointsMat<T> {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);

    let between;
    if let (Some(min), Some(max)) = (min, max) {
        between = rand::distributions::Uniform::from(min..max);
    } else {
        between = rand::distributions::Uniform::from(T::zero()..T::one());
    }

    let mut points = rlst_dynamic_array2!(T, [npoints, 3]);

    for i in 0..npoints {
        points[[i, 0]] = between.sample(&mut range);
        points[[i, 1]] = between.sample(&mut range);
        points[[i, 2]] = between.sample(&mut range);
    }

    points
}

/// Points fixture for testing, uniformly samples on surface of a sphere of diameter 1.
///
/// # Arguments
/// * `npoints` - The number of points to sample.
/// * `min` - The minumum coordinate value along each axis.
/// * `max` - The maximum coordinate value along each axis.
pub fn points_fixture_sphere<T: RlstScalar + rand::distributions::uniform::SampleUniform>(
    npoints: usize,
) -> PointsMat<T> {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);
    let pi = T::from(3.134159).unwrap();
    let two = T::from(2.0).unwrap();
    let half = T::from(0.5).unwrap();

    let between = rand::distributions::Uniform::from(T::zero()..T::one());

    let mut points = rlst_dynamic_array2!(T, [npoints, 3]);
    let mut phi = rlst_dynamic_array2!(T, [npoints, 1]);
    let mut theta = rlst_dynamic_array2!(T, [npoints, 1]);

    for i in 0..npoints {
        phi[[i, 0]] = between.sample(&mut range) * two * pi;
        theta[[i, 0]] = ((between.sample(&mut range) - half) * two).acos();
    }

    for i in 0..npoints {
        points[[i, 0]] = half * theta[[i, 0]].sin() * phi[[i, 0]].cos() + half;
        points[[i, 1]] = half * theta[[i, 0]].sin() * phi[[i, 0]].sin() + half;
        points[[i, 2]] = half * theta[[i, 0]].cos() + half;
    }

    points
}

///. Points fixture for testing, uniformly samples in the bounds [[0, 1), [0, 1), [0, 500)] for the x, y, and z
/// axes respectively.
///
/// # Arguments
/// * `npoints` - The number of points to sample.
pub fn points_fixture_col<T: Float + RlstScalar + rand::distributions::uniform::SampleUniform>(
    npoints: usize,
) -> PointsMat<T> {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);

    let between1 = rand::distributions::Uniform::from(T::zero()..T::from(0.1).unwrap());
    let between2 = rand::distributions::Uniform::from(T::zero()..T::from(500).unwrap());

    let mut points = rlst_dynamic_array2!(T, [npoints, 3]);

    for i in 0..npoints {
        // One axis has a different sampling
        points[[i, 0]] = between1.sample(&mut range);
        points[[i, 1]] = between1.sample(&mut range);
        points[[i, 2]] = between2.sample(&mut range);
    }

    points
}

/// Find the corners of a box discretising the surface of a box described by a Morton Key. The coordinates
/// are expected in column major order [x_1, x_2...x_N, y_1, y_2....y_N, z_1, z_2...z_N]
///
/// # Arguements:
/// * `coordinates` - points on the surface of a box.
pub fn find_corners<T: Float>(coordinates: &[T]) -> Vec<T> {
    let n = coordinates.len() / 3;

    let xs = coordinates.iter().take(n);
    let ys = coordinates[n..].iter().take(n);
    let zs = coordinates[2 * n..].iter().take(n);

    let x_min = *xs
        .clone()
        .min_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let x_max = *xs
        .clone()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let y_min = *ys
        .clone()
        .min_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let y_max = *ys
        .clone()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let z_min = *zs
        .clone()
        .min_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let z_max = *zs
        .clone()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Returned in column major order
    let corners = vec![
        x_min, x_max, x_min, x_max, x_min, x_max, x_min, x_max, y_min, y_min, y_max, y_max, y_min,
        y_min, y_max, y_max, z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max,
    ];

    corners
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::types::morton::MortonKey;

    #[test]
    fn test_find_corners() {
        let order = 5;
        let grid_1 = MortonKey::surface_grid::<f64>(order);

        let order = 2;
        let grid_2 = MortonKey::surface_grid::<f64>(order);

        let corners_1 = find_corners(&grid_1);
        let corners_2 = find_corners(&grid_2);

        // Test the corners are invariant by order of grid
        for (&c1, c2) in corners_1.iter().zip(corners_2) {
            assert!(c1 == c2);
        }

        // Test that the corners are the ones expected
        for (&c1, g2) in corners_1.iter().zip(grid_2) {
            assert!(c1 == g2);
        }
    }
}
