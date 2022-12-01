use std::collections::{HashMap, HashSet};

use rand::prelude::*;
use rand::SeedableRng;

use mpi::{
    environment::Universe,
    topology::{SystemCommunicator, UserCommunicator},
    traits::*,
};

use solvers_tree::{
    constants::{NCRIT, ROOT},
    types::{
        domain::Domain,
        morton::{MortonKey, MortonKeys},
        multi_node::MultiNodeTree,
        point::{PointType, Points},
    },
};

const NPOINTS: u64 = 5000;

/// Test fixture for NPOINTS randomly distributed points.
fn points_fixture() -> Vec<[f64; 3]> {
    let mut range = StdRng::seed_from_u64(0);
    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut points = Vec::new();

    for _ in 0..NPOINTS {
        points.push([
            between.sample(&mut range),
            between.sample(&mut range),
            between.sample(&mut range),
        ])
    }
    points
}

/// Test fixture for an unbalanced tree.
fn unbalanced_tree_fixture(world: &SystemCommunicator) -> MultiNodeTree {
    let points = points_fixture();
    let comm = world.duplicate();

    MultiNodeTree::new(&points, false, &comm)
}

/// Test fixture for an balanced tree.
fn balanced_tree_fixture(world: &SystemCommunicator) -> MultiNodeTree {
    let points = points_fixture();
    let comm = world.duplicate();

    MultiNodeTree::new(&points, true, &comm)
}

/// Test that the tree satisfies the ncrit condition.
fn test_ncrit(tree: &HashMap<MortonKey, Points>) {
    for points in tree.values() {
        assert!(points.len() <= NCRIT);
    }
}

/// Test that the tree spans the entire domain specified by the point distribution.
fn test_span(tree: &MortonKeys) {
    let min = tree.iter().min().unwrap();
    let max = tree.iter().max().unwrap();
    let block_set: HashSet<MortonKey> = tree.iter().cloned().collect();
    let max_level = tree.iter().map(|block| block.level()).max().unwrap();

    // Generate a uniform tree at the max level, and filter for range in this processor

    let mut level = 0;
    let mut uniform = vec![ROOT];
    while level < max_level {
        let mut descendents: Vec<MortonKey> = Vec::new();

        for node in uniform.iter() {
            let mut children = node.children();
            descendents.append(&mut children);
        }

        uniform = descendents;

        level += 1;
    }

    uniform = uniform
        .into_iter()
        .filter(|node| min <= node && node <= max)
        .collect();

    // Test that each member of the uniform tree, or their ancestors are contained within the
    // tree.
    for node in uniform.iter() {
        let ancestors = node.ancestors();

        let int: Vec<MortonKey> = ancestors
            .intersection(&block_set)
            .into_iter()
            .cloned()
            .collect();
        assert!(int.iter().len() > 0);
    }
}

/// Test that the leaves on separate nodes do not overlap.
fn test_no_overlaps(world: &SystemCommunicator, tree: &MultiNodeTree) {
    // Communicate bounds from each process
    let max = tree.keys.iter().max().unwrap();
    let min = *tree.keys.iter().min().unwrap();

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
        // println!("rank {:?} partner {:?} less than {:?}", rank, next_rank, max < &partner_min);
        // println!("rank {:?} max {:?} partner min {:?}", rank, max, partner_min);
        assert!(max < &partner_min)
    }
}

/// Test that the globally defined domain contains all the points at a given node.
fn test_global_bounds(world: &SystemCommunicator) {
    let points = points_fixture();

    let comm = world.duplicate();

    let domain = Domain::from_global_points(&points, &comm);

    // Test that all local points are contained within the global domain
    for point in points {
        assert!(domain.origin[0] <= point[0] && point[0] <= domain.origin[0] + domain.diameter[0]);
        assert!(domain.origin[1] <= point[1] && point[1] <= domain.origin[1] + domain.diameter[1]);
        assert!(domain.origin[2] <= point[2] && point[2] <= domain.origin[2] + domain.diameter[2]);
    }
}

/// Parallel test suite.
fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    // Distributed Trees
    let unbalanced = unbalanced_tree_fixture(&world);
    let balanced = balanced_tree_fixture(&world);

    // Tests for the unbalanced tree
    test_ncrit(&unbalanced.keys_to_points);
    if rank == 0 {
        println!("test_ncrit ... passed for unbalanced trees");
    }

    test_span(&unbalanced.keys);
    if rank == 0 {
        println!("test_span ... passed for unbalanced trees");
    }

    test_no_overlaps(&world, &unbalanced);
    if rank == 0 {
        println!("test_no_overlaps ... passed for unbalanced trees");
    }

    // Tests for balanced trees
    test_span(&balanced.keys);
    if rank == 0 {
        println!("test_span ... passed for balanced trees");
    }

    test_no_overlaps(&world, &balanced);
    if rank == 0 {
        println!("test_no_overlaps ... passed for balanced trees");
    }

    // Other parallel functionality
    test_global_bounds(&world);
    if rank == 0 {
        println!("test_global_bounds ... passed");
    }
}
