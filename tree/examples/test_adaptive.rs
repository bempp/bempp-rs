//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
use rand::prelude::*;
#[cfg(feature = "mpi")]
use rand::SeedableRng;

#[cfg(feature = "mpi")]
use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

#[cfg(feature = "mpi")]
use solvers_traits::tree::Tree;

#[cfg(feature = "mpi")]
use solvers_tree::types::{
    domain::Domain, morton::MortonKey, multi_node::MultiNodeTree, point::PointType,
};

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

/// Test that the leaves on separate nodes do not overlap.
#[cfg(feature = "mpi")]
fn test_no_overlaps(world: &UserCommunicator, tree: &MultiNodeTree) {
    // Communicate bounds from each process
    let max = tree.leaves.iter().max().unwrap();
    let min = tree.leaves.iter().min().unwrap();

    // Gather all bounds at root
    let size = world.size();
    let rank = world.rank();

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

    // Test that the partner's minimum node is greater than the process's maximum node
    if rank > 0 {
        assert!(partner_max < *min)
    }
}

#[cfg(feature = "mpi")]
fn test_adaptive(tree: &MultiNodeTree) {
    let levels: Vec<u64> = tree.get_keys().iter().map(|key| key.level()).collect();
    let first = levels[0];
    assert_eq!(false, levels.iter().all(|level| *level == first));
}

/// Test that the globally defined domain contains all the points at a given node.
#[cfg(feature = "mpi")]
fn test_global_bounds(world: &UserCommunicator) {
    let points = points_fixture(10000);

    let comm = world.duplicate();

    let domain = Domain::from_global_points(&points, &comm);

    // Test that all local points are contained within the global domain
    for point in points {
        assert!(domain.origin[0] <= point[0] && point[0] <= domain.origin[0] + domain.diameter[0]);
        assert!(domain.origin[1] <= point[1] && point[1] <= domain.origin[1] + domain.diameter[1]);
        assert!(domain.origin[2] <= point[2] && point[2] <= domain.origin[2] + domain.diameter[2]);
    }
}

#[cfg(feature = "mpi")]
fn main() {
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Setup tree parameters
    let adaptive = true;
    let n_crit = Some(50);
    let depth: Option<_> = None;
    let n_points = 10000;
    let k: Option<_> = None;

    let points = points_fixture(n_points);

    let tree = MultiNodeTree::new(&comm, k, &points, adaptive, n_crit, depth);

    test_global_bounds(&comm);
    if world.rank() == 0 {
        println!("\t ... test_global_bounds passed on adaptive tree");
    }
    test_adaptive(&tree);
    if world.rank() == 0 {
        println!("\t ... test_adaptive passed on adaptive tree");
    }
    test_no_overlaps(&comm, &tree);
    if world.rank() == 0 {
        println!("\t ... test_no_overlaps passed on adaptive tree");
    }
}

#[cfg(not(feature = "mpi"))]
fn main() {}
