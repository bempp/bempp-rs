use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use rand::prelude::*;
use rand::SeedableRng;

use mpi::{
    environment::Universe,
    topology::{UserCommunicator},
    traits::*
};

use solvers_traits::{
    tree::Tree
};

use solvers_tree::{
    constants::{ROOT, DEEPEST_LEVEL},
    types::{
        multi_node::{MultiNodeTree},
        point::{PointType, Point, Points},
        morton::{MortonKey, MortonKeys},
        domain::{Domain}
    }
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

/// Test that the leaves on separate nodes do not overlap.
fn test_no_overlaps(world: &UserCommunicator, tree: &MultiNodeTree) {
    // Communicate bounds from each process
    let max = tree.get_keys().iter().max().unwrap();
    let min = *tree.get_keys().iter().min().unwrap();

    // Gather all bounds at root
    let size = world.size();
    let rank = world.rank();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let previous_process = world.process_at_rank(previous_rank);
    let next_process = world.process_at_rank(next_rank);

    // Send min to partner
    if rank > 0 {
        previous_process.send(&min);
    }

    let mut partner_min = MortonKey::default();

    if rank < (size - 1) {
        next_process.receive_into(&mut partner_min);
    }

    // Test that the partner's minimum node is greater than the process's maximum node
    if rank < size - 1 {
        assert!(max < &partner_min)
    }
}

/// Test that the tree spans the entire domain specified by the point distribution.
fn test_span(tree: &MultiNodeTree) {

    let leaf_set: HashSet<MortonKey> = tree.get_keys().iter().cloned().collect();
    let points: Points = tree.get_points().to_vec();
    let max_level = leaf_set.iter().map(|l| l.level()).max().unwrap();

    for point in tree.get_points() {
        let ancestors: HashSet<MortonKey> = point.key.ancestors().iter().cloned().collect();
        
        let int = ancestors.intersection(&leaf_set).collect_vec();
        assert!(int.len() > 0);
    }
}

fn test_uniform(tree: &MultiNodeTree) {
    let levels: Vec<u64> = tree.get_keys().iter().map(|key| key.level()).collect();
    let first = levels[0];
    assert_eq!(true, levels.iter().all(|level| *level == first));
}

/// Test that the globally defined domain contains all the points at a given node.
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


fn main() {
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Setup tree parameters
    let adaptive = false;
    let n_crit: Option<_> = None;
    let depth = Some(4);
    let n_points = 10000;

    let points = points_fixture(n_points);

    let tree = MultiNodeTree::new(&points, adaptive, n_crit.clone(), depth, &comm);
    test_span(&tree);
    test_global_bounds(&comm);
    test_uniform(&tree);
    test_no_overlaps(&comm, &tree);
}
