//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
use rand::prelude::*;
#[cfg(feature = "mpi")]
use rand::SeedableRng;

#[cfg(feature = "mpi")]
use mpi::{environment::Universe, traits::*};

#[cfg(feature = "mpi")]
use solvers_traits::tree::FmmTree;

#[cfg(feature = "mpi")]
use solvers_tree::types::{multi_node::MultiNodeTree, point::PointType};

#[cfg(feature = "mpi")]
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

#[cfg(feature = "mpi")]
fn main() {
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Setup tree parameters
    // let adaptive = false;
    let adaptive = true;
    let n_crit = Some(50);
    // let n_crit: Option<_> = None;
    let depth: Option<_> = None;
    // let depth = Some(3);
    let n_points = 10000;
    let k: Option<_> = None;

    let points = points_fixture(n_points);

    let mut tree = MultiNodeTree::new(&comm, k, &points, adaptive, n_crit, depth);

    tree.create_let();

    println!(
        "rank {:?} has {:?} leaves",
        tree.world.rank(),
        tree.leaves.len()
    );
}

#[cfg(not(feature = "mpi"))]
fn main() {}
