//? mpirun -n {{NPROCESSES}} --features "mpi"
#![allow(unused_imports)]

use itertools::Itertools;
use rand::distributions::Uniform;
use rand::Rng;

#[cfg(feature = "mpi")]
use mpi::{collective::SystemOperation, environment::Universe, traits::*, Rank};

#[cfg(feature = "mpi")]
use bempp_tree::implementations::mpi_helpers::all_to_allv_sparse;

#[cfg(feature = "mpi")]
fn main() {
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();

    let rank = world.rank();
    let size = world.size();

    // Send a random number of packets from this process
    let mut rng = rand::thread_rng();
    let range = Uniform::from(1..size);
    let nsend = rng.sample(range);

    // Send packets to 'nsend' other processors in communicator, excluding
    // this process.
    let mut packet_destinations = Vec::new();
    let mut destination: Rank;
    let range = Uniform::from(0..size);

    while packet_destinations.len() < nsend as usize {
        destination = rng.sample(range);
        if destination != rank && !packet_destinations.contains(&destination) {
            packet_destinations.push(destination)
        }
    }

    // Form packets
    let mut packets: Vec<Vec<i32>> = vec![Vec::new(); nsend as usize];

    for packet in packets.iter_mut() {
        *packet = vec![rank; (rank + 1) as usize];
    }

    assert_eq!(packet_destinations.len(), packets.len());

    // Communicate the number of packets that this process will receive.
    let mut to_receive = vec![0 as Rank; size as usize];
    let mut dest_vec = vec![0 as Rank; size as usize];
    for &rank in packet_destinations.iter() {
        dest_vec[rank as usize] = 1;
    }

    world.all_reduce_into(&dest_vec, &mut to_receive, SystemOperation::sum());

    let recv_count = to_receive[rank as usize];

    let received = all_to_allv_sparse(&comm, &packets, &packet_destinations, &recv_count);

    // Test that the correct number of packets were received.
    let unique: Vec<i32> = received.iter().unique().cloned().collect();
    assert_eq!(unique.len() as i32, recv_count);
}
#[cfg(not(feature = "mpi"))]
fn main() {}
