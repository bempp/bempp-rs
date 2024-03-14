// //? mpirun -n {{NPROCESSES}} --features "mpi"
// #![allow(unused_imports)]

// #[cfg(feature = "mpi")]
// use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

// use bempp_tree::implementations::helpers::points_fixture;

// #[cfg(feature = "mpi")]
// use bempp_tree::types::{domain::Domain, morton::MortonKey, multi_node::MultiNodeTree};

// use bempp_traits::types::RlstScalar;
// use rlst::dense::RawAccess;

// use rand::distributions::uniform::SampleUniform;

// use num::traits::Float;

// /// Test that the leaves on separate nodes do not overlap.
// #[cfg(feature = "mpi")]
// fn test_no_overlaps<T: Float + Default + RlstScalar<Real = T>>(
//     world: &UserCommunicator,
//     tree: &MultiNodeTree<T>,
// ) {
//     // Communicate bounds from each process
//     let max = tree.leaves.iter().max().unwrap();
//     let min = tree.leaves.iter().min().unwrap();

//     // Gather all bounds at root
//     let size = world.size();
//     let rank = world.rank();

//     let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
//     let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

//     let previous_process = world.process_at_rank(previous_rank);
//     let next_process = world.process_at_rank(next_rank);

//     // Send max to partner
//     if rank < (size - 1) {
//         next_process.send(max);
//     }

//     let mut partner_max = MortonKey::default();

//     if rank > 0 {
//         previous_process.receive_into(&mut partner_max);
//     }

//     // Test that the partner's minimum node is greater than the process's maximum node
//     if rank > 0 {
//         assert!(partner_max < *min)
//     }
// }

// /// Test that the globally defined domain contains all the points at a given node.
// #[cfg(feature = "mpi")]
// fn test_global_bounds<T: Float + Default + RlstScalar + Equivalence + SampleUniform>(
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
//     let adaptive = true;
//     let n_crit = Some(50);
//     let depth: Option<_> = None;
//     let n_points = 10000;
//     let k = 2;

//     let points = points_fixture::<f64>(n_points, None, None);
//     let global_idxs: Vec<_> = (0..n_points).collect();

//     let tree = MultiNodeTree::new(
//         &comm,
//         points.data(),
//         adaptive,
//         n_crit,
//         depth,
//         k,
//         &global_idxs,
//     );

//     test_global_bounds::<f64>(&comm);
//     if world.rank() == 0 {
//         println!("\t ... test_global_bounds passed on adaptive tree");
//     }

//     test_no_overlaps(&comm, &tree);
//     if world.rank() == 0 {
//         println!("\t ... test_no_overlaps passed on adaptive tree");
//     }
// }
#[cfg(feature = "mpi")]
fn main() {}

#[cfg(not(feature = "mpi"))]
fn main() {}
