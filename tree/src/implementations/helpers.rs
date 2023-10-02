use std::collections::HashMap;

use rand::prelude::*;
use rand::SeedableRng;

use rlst::dense::{base_matrix::BaseMatrix, rlst_mat, Dynamic, Matrix, VectorContainer};

use crate::types::{domain::Domain, morton::MortonKey};

pub type PointsMat =
    Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

/// Points fixture for testing, uniformly samples in each axis from min to max.
///
/// # Arguments
/// * `npoints` - The number of points to sample.
/// # `min` - The minumum coordinate value along each axis.
/// # `max` - The maximum coordinate value along each axis.
pub fn points_fixture(npoints: usize, min: Option<f64>, max: Option<f64>) -> PointsMat {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);

    let between;
    if let (Some(min), Some(max)) = (min, max) {
        between = rand::distributions::Uniform::from(min..max);
    } else {
        between = rand::distributions::Uniform::from(0.0_f64..1.0_f64);
    }

    let mut points = rlst_mat![f64, (npoints, 3)];

    for i in 0..npoints {
        points[[i, 0]] = between.sample(&mut range);
        points[[i, 1]] = between.sample(&mut range);
        points[[i, 2]] = between.sample(&mut range);
    }

    points
}

//. Points fixture for testing, uniformly samples in the bounds [[0, 1), [0, 1), [0, 500)] for the x, y, and z
/// axes respectively.
///
/// # Arguments
/// * `npoints` - The number of points to sample.
pub fn points_fixture_col(npoints: usize) -> PointsMat {
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);

    let between1 = rand::distributions::Uniform::from(0f64..0.1f64);
    let between2 = rand::distributions::Uniform::from(0f64..500f64);

    let mut points = rlst_mat![f64, (npoints, 3)];

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
/// * `coordinates`
pub fn find_corners(coordinates: &[f64]) -> Vec<f64> {
    let n = coordinates.len() / 3;

    let xs = coordinates.iter().take(n);
    let ys = coordinates[n..].iter().take(n);
    let zs = coordinates[2*n..].iter().take(n);

    let x_min = xs.clone().min_by(|&a, &b| a.partial_cmp(b).unwrap()).unwrap().clone();
    let x_max = xs.clone().max_by(|&a, &b| a.partial_cmp(b).unwrap()).unwrap().clone();
    let y_min = ys.clone().min_by(|&a, &b| a.partial_cmp(b).unwrap()).unwrap().clone();
    let y_max = ys.clone().max_by(|&a, &b| a.partial_cmp(b).unwrap()).unwrap().clone();
    let z_min = zs.clone().min_by(|&a, &b| a.partial_cmp(b).unwrap()).unwrap().clone();
    let z_max = zs.clone().max_by(|&a, &b| a.partial_cmp(b).unwrap()).unwrap().clone();

    // Returned in column major order
    let corners = vec![
        x_min, x_min, x_min, x_min, x_max, x_max, x_max, x_max, 
        y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max,
        z_min, z_max, z_min, z_max, z_min, z_max, z_min, z_max
    ];

    corners
}

/// We need to quickly map between corners and their corresponding positions in the surface grid via
/// their respective indices.
/// 
/// # Arguments
/// * `order` - the order of expansions used in constructing the surface grid
pub fn map_corners_to_surface(order: usize)
->  (
    HashMap<usize, usize>,
    HashMap<usize, usize>
)
{
    // Pick an arbitrary point to draw a surface using existing API TODO: Replace
    let point = [0.5, 0.5, 0.5];
    let domain = Domain {
        origin: [0., 0., 0.],
        diameter: [1., 1., 1.],
    };

    let key = MortonKey::from_point(&point, &domain, 0);

    let (_, surface_multindex) = MortonKey::surface_grid(order);

    let nsurf = surface_multindex.len() / 3;
    let ncorners = 8;
    let corners_multindex = vec![
        0, 0, 0, 0, order-1, order-1, order-1, order-1, 
        0, 0, order-1, order-1, 0, 0, order-1, order-1,
        0, order-1, 0, order-1, 0, order-1, 0, order-1
    ];

    let mut _map = HashMap::new();
    let mut _inv_map = HashMap::new();

    for i in 0..nsurf {
        let s = [
            surface_multindex[i],
            surface_multindex[nsurf+i],
            surface_multindex[2*nsurf+i]
        ];

        for j in 0..ncorners {
            let c = [
                corners_multindex[j],
                corners_multindex[8+j],
                corners_multindex[16+j]
            ];

            if s[0] == c[0] && s[1] == c[1] && s[2] == c[2] {
                _map.insert(i, j);
                _inv_map.insert(j, i);
            }
        }
    }

    (_map, _inv_map)
}

#[cfg(test)]
mod test {

    use crate::implementations::helpers::map_corners_to_surface;
    use crate::types::morton::MortonKey;
    use crate::types::domain::Domain;
    use super::find_corners;

    #[test]
    fn test_find_corners() {

        let order = 5;
        let (grid_1, _) = MortonKey::surface_grid(order);
        
        let order = 2;
        let (grid_2, _) = MortonKey::surface_grid(order);
        
        let corners_1 = find_corners(&grid_1);
        let corners_2= find_corners(&grid_2);

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