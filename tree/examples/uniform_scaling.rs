// ? mpirun -n {{NPROCESSES}} --features "mpi"

use std::time::Instant;

use mpi::collective::SystemOperation;
use rand::prelude::*;
use rand::SeedableRng;

use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

use bempp_traits::tree::Tree;

use bempp_tree::types::{
    domain::Domain, morton::MortonKey, multi_node::MultiNodeTree, point::PointType,
};


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

fn main() {
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();
    let rank = world.rank();
    let size = world.size();

    // Setup tree parameters
    let adaptive = false;

    let depth: u64 = 5;
    let n_crit: Option<_> = None;
    let k = 2;

    let depth = Some(depth);

    let n_points = 1000000;

    // Generate some random test data local to each process
    let points = points_fixture(n_points, None, None);
    let global_idxs: Vec<usize> = (0..n_points).collect();

    // Calculate the global domain
    let domain = Domain::from_global_points(&points[..], &comm);

    // Create a uniform tree
    let s = Instant::now();
    let tree = MultiNodeTree::new(&comm, &points[..], adaptive, n_crit, depth, k, &global_idxs[..]);
    let time = s.elapsed();
    let nleaves = tree.leaves.len();
    let mut sum = 0;

    if rank == 0 {
        world
            .process_at_rank(0)
            .reduce_into_root(&nleaves, &mut sum, SystemOperation::sum());

        println!("{:?}, {:?}, {:?}", size, sum, time)

    } else {
        world
            .process_at_rank(0)
            .reduce_into(&nleaves, SystemOperation::sum())
    }


}
