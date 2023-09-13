use rand::prelude::*;
use rand::SeedableRng;
// use rlst::dense::{base_matrix::BaseMatrix, rlst_mat, Dynamic, Matrix, VectorContainer};

// pub type PointsMat =
//     Matrix<f64, BaseMatrix<f64, VectorContainer<f64>, Dynamic, Dynamic>, Dynamic, Dynamic>;

// Points fixture for testing, uniformly samples in each axis from min to max.
// pub fn points_fixture(npoints: usize, min: Option<f64>, max: Option<f64>) -> PointsMat {
//     // Generate a set of randomly distributed points
//     let mut range = StdRng::seed_from_u64(0);

//     let between;
//     if let (Some(min), Some(max)) = (min, max) {
//         between = rand::distributions::Uniform::from(min..max);
//     } else {
//         between = rand::distributions::Uniform::from(0.0_f64..1.0_f64);
//     }

//     let mut points = rlst_mat![f64, (npoints, 3)];

//     for i in 0..npoints {
//         points[[i, 0]] = between.sample(&mut range);
//         points[[i, 1]] = between.sample(&mut range);
//         points[[i, 2]] = between.sample(&mut range);
//     }

//     points
// }

pub fn points_fixture(
    npoints: usize,
    min: Option<f64>,
    max: Option<f64>,
) -> Vec<f64>
{
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);

    let between;
    if let (Some(min), Some(max)) = (min, max) {
        between = rand::distributions::Uniform::from(min..max);
    } else {
        between = rand::distributions::Uniform::from(0.0_f64..1.0_f64);
    }

    // let mut points = rlst_mat![f64, (npoints, 3)];
    let mut points = Vec::new();

    for i in 0..npoints {
        let mut tmp = Vec::new();
        for _ in 0..3 {
            tmp.push(between.sample(&mut range))
        }
        points.append(&mut tmp)
    }

    points
}

// Points fixture for testing, uniformly samples in the bounds [[0, 1), [0, 1), [0, 500)] for the x, y, and z
// axes respectively.
// pub fn points_fixture_col(npoints: usize) -> PointsMat {
//     // Generate a set of randomly distributed points
//     let mut range = StdRng::seed_from_u64(0);

//     let between1 = rand::distributions::Uniform::from(0f64..0.1f64);
//     let between2 = rand::distributions::Uniform::from(0f64..500f64);

//     let mut points = rlst_mat![f64, (npoints, 3)];

//     for i in 0..npoints {
//         // One axis has a different sampling
//         points[[i, 0]] = between1.sample(&mut range);
//         points[[i, 1]] = between1.sample(&mut range);
//         points[[i, 2]] = between2.sample(&mut range);
//     }

//     points
// }

pub fn points_fixture_col(
    npoints: usize,
    min: Option<f64>,
    max: Option<f64>,
) -> Vec<f64>
{
    // Generate a set of randomly distributed points
    let mut range = StdRng::seed_from_u64(0);

    let between1 = rand::distributions::Uniform::from(0f64..0.1f64);
    let between2 = rand::distributions::Uniform::from(0f64..500f64);

    // let mut points = rlst_mat![f64, (npoints, 3)];
    let mut points = Vec::new();

    for i in 0..npoints {
        let mut tmp = Vec::new();
        tmp.push(between1.sample(&mut range));
        tmp.push(between1.sample(&mut range));
        tmp.push(between2.sample(&mut range));
        points.append(&mut tmp)
    }

    points
}
