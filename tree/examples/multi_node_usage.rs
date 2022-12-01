// Example of how to use a multi node tree with its Tree trait interface
use std::collections::{HashMap, HashSet};

use rand::prelude::*;
use rand::SeedableRng;

use mpi::{
    environment::Universe,
    topology::{SystemCommunicator, UserCommunicator},
    traits::*,
};

use solvers_traits::tree::Tree;

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

fn main() {
    // Initialise MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    // Distributed Tree
    let points = points_fixture();
    let comm = world.duplicate();
    let balanced = MultiNodeTree::new(&points, true, &comm);

    // Getters for points, keys, domain and balancing info
    let points: &Points = balanced.get_points();
    let keys: &MortonKeys = balanced.get_keys();
    let domain: &Domain = balanced.get_domain();
    let balance: bool = balanced.get_balanced();

    // Maps for point -> key, and key -> points
    let key = keys[0];
    let points = balanced.map_key_to_points(&key).unwrap();
    let point = points[0];
    let check = balanced.map_point_to_key(&point).unwrap();
    assert!(*check == key);
    println!(
        "{:?} points associated with node {:?}",
        points.len(),
        key.anchor
    );
    println!(
        "The point {:?} is associated with node {:?} \n",
        point.coordinate, check.anchor
    );
}
