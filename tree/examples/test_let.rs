use itertools::Itertools;
use rand::prelude::*;
use rand::SeedableRng;

use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

use solvers_traits::tree::{LocallyEssentialTree, Tree};

use solvers_tree::types::{
    multi_node::MultiNodeTree, point::PointType, single_node::SingleNodeTree, morton::MortonKey
};

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
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();
    let rank = world.rank();
    let size = world.size();

    // Setup tree parameters
    // let adaptive = false;
    let adaptive = true;
    let n_crit = Some(50);
    // let n_crit: Option<_> = None;
    let depth: Option<_> = None;
    // let depth = Some(3);
    let n_points = 100000;
    let k: Option<_> = None;

    let points = points_fixture(n_points);

    let mut tree = MultiNodeTree::new(&comm, k, &points, adaptive, n_crit, depth);

    // Create locally essential tree
    tree.create_let();

    // Test that the local leaves have retained Morton ordering.
    let mut local: Vec<MortonKey> = tree.leaves.iter().sorted().cloned().collect();

    let min = local.iter().min().unwrap();
    let max = local.iter().max().unwrap();

    // Communicate with nearest neighbours.
    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let previous_process = world.process_at_rank(previous_rank);
    let next_process = world.process_at_rank(next_rank);

    // Send max to partner
    if rank < (size - 1) {
        next_process.send(max);
    }

    let mut partner_max = MortonKey::default();

    if rank > 0 {
        previous_process.receive_into(&mut partner_max);
    }

    // Test that the partner's max node is less than the process's maximum node,
    // i.e. in approximate Morton order with some degree of overlap.
    if rank > 0 {
        assert!(partner_max <= *max)
    }

}
