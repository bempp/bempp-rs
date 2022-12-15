use mpi::{
    topology::{Rank, UserCommunicator},
    traits::*,
};

use rand::Rng;

use hyksort::hyksort;

fn main () {

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let mut rng = rand::thread_rng();

    let k = 2;

    let nsamples = 1000;
    let mut arr: Vec<i32> = (0..nsamples).map(|_| rng.gen_range(0..=20)).collect();
    let mut replica = arr.iter().cloned().collect();
    arr.append(&mut replica);
    hyksort(&mut arr, k, comm.duplicate());

    let min = arr.iter().min().unwrap().clone();
    let max = arr.iter().max().unwrap().clone();
    // Gather all bounds at root
    let size = comm.size();
    let rank = comm.rank();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let previous_process = comm.process_at_rank(previous_rank);
    let next_process = comm.process_at_rank(next_rank);

    // Send min to partner
    if rank > 0 {
        previous_process.send(&min);
    }

    let mut partner_min: i32 = 0;

    if rank < (size - 1) {
        next_process.receive_into(&mut partner_min);
    }

    // Test that the partner's minimum node is greater than the process's maximum node
    if rank < size - 1 {
        assert!(max < partner_min)
    }
    
    // Test that each node's portion is locally sorted

    for i in 0..(arr.iter().len() -1) {

        let a = arr[i];
        let b = arr[i+1];

        assert!(a <= b);
    }

}