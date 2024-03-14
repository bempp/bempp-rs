// // ? mpirun -n {{NPROCESSES}} --features "mpi"
// #![allow(unused_imports)]

// #[cfg(feature = "mpi")]
// use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

// use bempp_traits::tree::Tree;
// use bempp_traits::types::RlstScalar;

// use rand::distributions::uniform::SampleUniform;

// #[cfg(feature = "mpi")]
// use bempp_tree::types::{domain::Domain, morton::MortonKey, multi_node::MultiNodeTree};

// use bempp_tree::implementations::helpers::points_fixture;
// use rlst::dense::RawAccess;

// use num::traits::Float;

// /// Test that the leaves on separate nodes do not overlap.
// #[cfg(feature = "mpi")]
// fn test_no_overlaps<T: Float + Default + RlstScalar<Real = T>>(
//     world: &UserCommunicator,
//     tree: &MultiNodeTree<T>,
// ) {
//     // Communicate bounds from each process
//     let max = tree.get_all_leaves_set().iter().max().unwrap();
//     let min = tree.get_all_leaves_set().iter().min().unwrap();

//     // Gather all bounds at root
//     let size = world.size();
//     let rank = world.rank();

//     let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
//     let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

//     let previous_process = world.process_at_rank(previous_rank);
//     let next_process = world.process_at_rank(next_rank);

//     // Send min to partner
//     if rank > 0 {
//         previous_process.send(min);
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

// /// Test that the globally defined domain contains all the points at a given node.
// #[cfg(feature = "mpi")]
// fn test_global_bounds<T: RlstScalar + Float + Default + Equivalence + SampleUniform>(
//     world: &UserCommunicator,
// ) {
//     let npoints = 10000;
//     let points = points_fixture::<T>(npoints, None, None);

//     let comm = world.duplicate();

//     let domain = Domain::from_global_points(points.data(), &comm);

//     // Test that all local points are contained within the global domain
//     for i in 0..npoints {
//         let x = points.data()[i];
//         let y = points.data()[i + npoints];
//         let z = points.data()[i + 2 * npoints];

//         assert!(domain.origin[0] <= x && x <= domain.origin[0] + domain.diameter[0]);
//         assert!(domain.origin[1] <= y && y <= domain.origin[1] + domain.diameter[1]);
//         assert!(domain.origin[2] <= z && z <= domain.origin[2] + domain.diameter[2]);
//     }
// }

// #[cfg(feature = "mpi")]
// fn main() {
//     // Setup an MPI environment
//     let universe: Universe = mpi::initialize().unwrap();
//     let world = universe.world();
//     let comm = world.duplicate();

//     // Setup tree parameters
//     let adaptive = false;
//     let k = 2;
//     let depth = Some(3);
//     let n_points = 10000;

//     // Generate some random test data local to each process
//     let points = points_fixture::<f32>(n_points, None, None);
//     let global_idxs: Vec<_> = (0..n_points).collect();

//     // Create a uniform tree
//     let tree = MultiNodeTree::new(&comm, points.data(), adaptive, None, depth, k, &global_idxs);

//     test_global_bounds::<f32>(&comm);
//     if world.rank() == 0 {
//         println!("\t ... test_global_bounds passed on uniform tree");
//     }

//     test_no_overlaps(&comm, &tree);
//     if world.rank() == 0 {
//         println!("\t ... test_no_overlaps passed on uniform tree");
//     }
// }

#[cfg(not(feature = "mpi"))]
fn main() {}

#[cfg(feature = "mpi")]
fn main() {}
