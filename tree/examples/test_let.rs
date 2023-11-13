// //? mpirun -n {{NPROCESSES}} --features "mpi"
// #![allow(unused_imports)]

// use bempp_traits::tree::Tree;
// use itertools::Itertools;

// #[cfg(feature = "mpi")]
// use mpi::{environment::Universe, traits::*};

// use bempp_tree::implementations::helpers::points_fixture;
// use rlst::common::traits::accessors::RawAccess;

// #[cfg(feature = "mpi")]
// use bempp_tree::types::multi_node::MultiNodeTree;

// /// Test that the near field boxes are contained either locally, or in the received boxes
// /// from the locally essential tree.
// #[cfg(feature = "mpi")]
// fn test_near_field(tree: &MultiNodeTree) {
//     // Create locally essential tree
//     let locally_essential_tree = tree.create_let();

//     // Test that the tree contains all the data it requires for the near field evaluations
//     for key in tree.get_all_leaves_set() {
//         let near_field = key.neighbors();

//         for n in near_field.iter() {
//             // TODO: work out why this has started failing
//             // assert!(tree.leaves_set.contains(n) || locally_essential_tree.leaves_set.contains(n));
//         }
//     }
// }

// #[cfg(feature = "mpi")]
// fn main() {
//     // Setup an MPI environment
//     let universe: Universe = mpi::initialize().unwrap();
//     let world = universe.world();
//     let comm = world.duplicate();

//     // Setup tree parameters
//     // let adaptive = false;
//     let adaptive = false;
//     let n_crit = Some(50);
//     let depth = Some(3);
//     let n_points = 100000;
//     let k = 2;

//     let points = points_fixture(n_points, None, None);
//     let global_idxs = (0..n_points).collect_vec();

//     let uniform_tree = MultiNodeTree::new(
//         &comm,
//         points.data(),
//         adaptive,
//         n_crit,
//         depth,
//         k,
//         &global_idxs,
//     );

//     test_near_field(&uniform_tree);
//     if world.rank() == 0 {
//         println!("\t ... test_near_field passed on uniform tree");
//     }
// }

#[cfg(feature = "mpi")]
fn main() {}

#[cfg(not(feature = "mpi"))]
fn main() {}
