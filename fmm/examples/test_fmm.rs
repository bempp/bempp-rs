use rand::prelude::*;
use rand::SeedableRng;

use mpi::{environment::Universe, traits::*};

use solvers_fmm::laplace::LaplaceKernel;
use solvers_traits::fmm::Fmm;
use solvers_traits::fmm::FmmTree;

use solvers_fmm::laplace::KiFmm;
use solvers_tree::types::point::PointType;
use solvers_tree::types::single_node::SingleNodeTree;

pub fn points_fixture(npoints: i32) -> Vec<[f64; 3]> {
    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points: Vec<[PointType; 3]> = Vec::new();

    for _ in 0..npoints {
        points.push([
            between.sample(&mut range),
            between.sample(&mut range),
            between.sample(&mut range),
        ])
    }

    points
}

fn main() {
    // Setup tree parameters
    let adaptive = true;
    let n_crit = Some(50);
    let depth: Option<_> = None;
    let n_points = 10000;

    let points = points_fixture(n_points);

    let mut tree = SingleNodeTree::new(&points, adaptive, n_crit, depth);

    tree.create_let();

    let kernel = Box::new(LaplaceKernel {
        dim: 3,
        is_singular: true,
        value_dimension: 3,
    });
    let tree = Box::new(tree);

    let mut fmm = KiFmm {
        kernel,
        tree,
        order_check: 5,
        order_equivalent: 5,
        alpha_inner: 1.05,
        alpha_outer: 1.95,
    };

    fmm.run(3);
}
