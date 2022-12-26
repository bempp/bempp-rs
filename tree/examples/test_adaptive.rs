use std::collections::HashSet;

use rand::prelude::*;
use rand::SeedableRng;

use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

use solvers_traits::tree::Tree;

use solvers_tree::types::{
    domain::Domain, morton::{MortonKey, MortonKeys}, multi_node::MultiNodeTree, point::PointType,
    single_node::SingleNodeTree,
};
use solvers_tree::implementations::impl_morton::encode_anchor;
use solvers_tree::constants::{LEVEL_SIZE, DEEPEST_LEVEL};

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
    let max = tree.keys.iter().max().unwrap();
    let min = tree.keys.iter().min().unwrap();

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

/// Test that the tree spans the entire domain specified by the point distribution.
/// Test that the tree spans the entire domain specified by the point distribution.
fn test_span(points: &[[f64; 3]], tree: &MultiNodeTree) {
    let min: &MortonKey = tree.get_keys().iter().min().unwrap();
    let max: &MortonKey = tree.get_keys().iter().max().unwrap();
    let block_set: HashSet<MortonKey> = tree.get_keys().iter().cloned().collect();
    let max_level = tree
        .get_keys()
        .iter()
        .map(|block| block.level())
        .max()
        .unwrap();

    let diameter = 1 << (DEEPEST_LEVEL - max_level as u64);

    let uniform = MortonKeys {
            keys: (0..LEVEL_SIZE)
                .step_by(diameter)
                .flat_map(|i| (0..LEVEL_SIZE).step_by(diameter).map(move |j| (i, j)))
                .flat_map(|(i, j)| (0..LEVEL_SIZE).step_by(diameter).map(move |k| [i, j, k]))
                .map(|anchor| {
                    let morton = encode_anchor(&anchor, max_level as u64);
                    MortonKey { anchor, morton }
                })
                .filter(|k| (k >= min) && ( k <= max))
                .collect(),
            index: 0,
        };       

        
    for (i, node) in uniform.iter().enumerate() {
        // println!("considering i {:?}", i);
            let ancestors= node.ancestors();
            let int: Vec<MortonKey> = ancestors
                .intersection(&block_set)
                .into_iter()
                .cloned()
                .collect();
            let mut ancestors: Vec<MortonKey> = ancestors.into_iter().collect();
            ancestors.sort();

            if (int.iter().len() == 0) {
                println!("\n NODE {:?} LEVEL {:?} \n siblings {:?} \n ANCESTORS {:?} \n int {:?}\n\n", node, node.level(), node.siblings(), ancestors, int);
            }
            println!("int {:?} node {:?}", int, node);
            assert!(int.iter().len() > 0);
        }
        ////////////////

}

fn test_adaptive(tree: &MultiNodeTree) {
    let levels: Vec<u64> = tree.get_keys().iter().map(|key| key.level()).collect();
    let first = levels[0];
    assert_eq!(false, levels.iter().all(|level| *level == first));
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
    let adaptive = true;
    let n_crit = Some(50);
    let depth: Option<_> = None;
    let n_points = 1000;
    let k: Option<_> = None;

    let points = points_fixture(n_points);

    let tree = MultiNodeTree::new(&comm, k, &points, adaptive, n_crit, depth);
    // test_span(&points, &tree);
    // if world.rank() == 0 {
    //     println!("\t ... test_span passed on adaptive tree");
    // }
    // test_global_bounds(&comm);
    // if world.rank() == 0 {
    //     println!("\t ... test_global_bounds passed on adaptive tree");
    // }
    // test_adaptive(&tree);
    // if world.rank() == 0 {
    //     println!("\t ... test_adaptive passed on adaptive tree");
    // }
    // test_no_overlaps(&comm, &tree);
    // if world.rank() == 0 {
    //     println!("\t ... test_no_overlaps passed on adaptive tree");
    // }
}
