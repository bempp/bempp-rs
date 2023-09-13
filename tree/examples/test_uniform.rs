// ? mpirun -n {{NPROCESSES}} --features "mpi"

use rand::prelude::*;
use rand::SeedableRng;

use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

use bempp_traits::tree::Tree;

// use rlst::{
//     dense::{
//         base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix, rlst_col_vec,
//         rlst_mat, rlst_pointer_mat, traits::*, Dot, global,
//     },
// };

use bempp_tree::types::{
    domain::Domain, morton::MortonKey, multi_node::MultiNodeTree, point::PointType,
};

fn points_fixture(
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

// /// Test that the leaves on separate nodes do not overlap.
// fn test_no_overlaps(world: &UserCommunicator, tree: &MultiNodeTree) {
//     // Communicate bounds from each process
//     let max = tree.get_keys().iter().max().unwrap();
//     let min = *tree.get_keys().iter().min().unwrap();

//     // Gather all bounds at root
//     let size = world.size();
//     let rank = world.rank();

//     let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
//     let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

//     let previous_process = world.process_at_rank(previous_rank);
//     let next_process = world.process_at_rank(next_rank);

//     // Send min to partner
//     if rank > 0 {
//         previous_process.send(&min);
//     }

//     let mut partner_min = MortonKey::default();

//     if rank < (size - 1) {
//         next_process.receive_into(&mut partner_min);
//     }

//     // Test that the partner's minimum node is greater than the process's maximum node
//     if rank < size - 1 {
//         assert!(max < &partner_min)
//     }
// }

// fn test_uniform(tree: &MultiNodeTree) {
//     let levels: Vec<u64> = tree.get_keys().iter().map(|key| key.level()).collect();
//     let first = levels[0];
//     assert_eq!(true, levels.iter().all(|level| *level == first));
// }

// /// Test that the globally defined domain contains all the points at a given node.
// fn test_global_bounds(world: &UserCommunicator) {
//     let points = points_fixture(10000);

//     let comm = world.duplicate();

//     let domain = Domain::from_global_points(&points, &comm);

//     // Test that all local points are contained within the global domain
//     for point in points {
//         assert!(domain.origin[0] <= point[0] && point[0] <= domain.origin[0] + domain.diameter[0]);
//         assert!(domain.origin[1] <= point[1] && point[1] <= domain.origin[1] + domain.diameter[1]);
//         assert!(domain.origin[2] <= point[2] && point[2] <= domain.origin[2] + domain.diameter[2]);
//     }
// }

fn main() {
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    // Setup tree parameters
    let adaptive = false;
    let n_crit: Option<_> = None;
    let k = 2;
    let depth = Some(3);
    let n_points = 10000;

    // Generate some random test data local to each process
    let points = points_fixture(n_points, None, None);
    let global_idxs: Vec<_> = (0..n_points).collect();

    // Calculate the global domain
    let domain = Domain::from_global_points(&points[..], &comm);

    // Create a uniform tree
    let tree = MultiNodeTree::new(&comm, &points[..], adaptive, n_crit, depth, k, &global_idxs);

    println!("Rank {:?}, leaves {:?}", world.rank(), tree.leaves.len())
    // test_global_bounds(&comm);
    // if world.rank() == 0 {
    //     println!("\t ... test_global_bounds passed on uniform tree");
    // }
    // test_uniform(&tree);
    // if world.rank() == 0 {
    //     println!("\t ... test_uniform passed on uniform tree");
    // }
    // test_no_overlaps(&comm, &tree);
    // if world.rank() == 0 {
    //     println!("\t ... test_no_overlaps passed on uniform tree");
    // }
}
// fn main() {}
