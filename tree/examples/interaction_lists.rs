use std::collections::HashSet;

use rand::prelude::*;
use rand::SeedableRng;

use mpi::{environment::Universe, topology::UserCommunicator, traits::*};

use solvers_traits::tree::{LocallyEssentialTree, Tree};

use solvers_tree::types::{
    domain::Domain, morton::MortonKey, multi_node::MultiNodeTree, point::PointType,
    single_node::SingleNodeTree,
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

    // Setup tree parameters
    let adaptive = true;
    let n_crit = Some(15);
    let depth: Option<_> = None;
    let n_points = 10000;
    let k: Option<_> = None;

    let points = points_fixture(n_points);

    let mut tree = MultiNodeTree::new(&comm, k, &points, adaptive, n_crit, depth);

    tree.create_let();
    // tree.load_balance_let();
}

// use mpi;
// use mpi::point_to_point::Status;
// use mpi::traits::*;

// const COUNT: usize = 5;

// fn main() {
//     let universe = mpi::initialize().unwrap();
//     let world = universe.world();
//     let size = world.size();
//     let rank = world.rank();

//     let global_send_partners = [[1, 2], [2, 3], [3, 0], [0, 1]];
//     let global_recv_partners = [[3, 2], [0, 3], [1, 0], [2, 1]];

//     let recv_partners = global_recv_partners[rank as usize];
//     let send_partners = global_send_partners[rank as usize];
    
//     let send_count = send_partners.len();
//     let recv_count = recv_partners.len();
//     let nreqs = send_count + recv_count;
//     let mut results: Vec<Vec<i32>> = Vec::new();

//     for recv_partner in recv_partners.iter() {
//         results.push(vec![0i32; (recv_partner+1) as usize])
//     }
    
//     let mut packets: Vec<Vec<i32>> = Vec::new();

//     for send_partner in send_partners {
//         packets.push(vec![rank; (rank+1) as usize])
//     }


//     mpi::request::multiple_scope(nreqs as usize, |scope, coll| {


//         for (i, packet) in packets.iter().enumerate() {
//             let sreq = world
//                 .process_at_rank(send_partners[i])
//                 .immediate_send(scope, &packet[..]);
//                 coll.add(sreq);
//         }
        
//         for (i, result) in results.iter_mut().enumerate() {
//             let rreq = world
//                 .process_at_rank(recv_partners[i])
//                 .immediate_receive_into(scope, &mut result[..]);

//             coll.add(rreq);
//         }


//         let mut out = vec![];
//         coll.wait_all(&mut out);
//         assert_eq!(out.len(), nreqs as usize);
//         // let mut send_count = 0;
//         // let mut recv_count = 0;
//         // for (_, _, result) in out {
//         //     if *result == rank {
//         //         send_count += 1;
//         //     } else {
//         //         recv_count += 1;
//         //     }
//         // }
//         // assert_eq!(send_count, COUNT);
//         // assert_eq!(recv_count, COUNT);
//     });

//     println!("rank {:?} received {:?}", rank, results);

//     // Check wait_*() with a buffer of increasing values
//     // let x: Vec<i32> = (0..COUNT as i32).collect();
//     // let mut result: Vec<i32> = vec![0; COUNT];
//     // mpi::request::multiple_scope(2 * COUNT, |scope, coll| {
//     //     for elm in &x {
//     //         let sreq = world.process_at_rank(next_proc).immediate_send(scope, elm);
//     //         coll.add(sreq);
//     //     }
//     //     for val in result.iter_mut() {
//     //         let rreq = world
//     //             .process_at_rank(prev_proc)
//     //             .immediate_receive_into(scope, val);
//     //         coll.add(rreq);
//     //     }
//     //     let mut out: Vec<(usize, Status, &i32)> = vec![];
//     //     coll.wait_all(&mut out);
//     //     assert_eq!(out.len(), 2 * COUNT);
//     // });
//     // // Ensure the result and x are an incrementing array of integers
//     // result.sort();
//     // assert_eq!(result, x);
// }