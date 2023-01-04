//! Hyksort [1] implementation in Rust, a near direct port of [2].
//!
//! **References**
//!
//! [1] Sundar, H., Malhotra, D., & Biros, G. (2013, June). Hyksort: a new variant of hypercube quicksort on distributed memory architectures. In Proceedings of the 27th international ACM conference on international conference on supercomputing (pp. 293-302).
//!
//! [2] https://github.com/hsundar/usort
extern crate superslice;

use std::convert::TryInto;

use rand::Rng;

use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::request::WaitGuard;
use mpi::topology::Rank;
use mpi::topology::UserCommunicator;
use mpi::traits::*;
use mpi::Count;

use superslice::*;

/// Modulo function compatible with signed integers.
pub fn modulo(a: i32, b: i32) -> i32 {
    ((a % b) + b) % b
}

/// Parallel selection algorithm to determine 'k' splitters from the global array currently being
/// considered in the communicator.
pub fn parallel_select<T>(arr: &Vec<T>, &k: &Rank, comm: UserCommunicator) -> Vec<T>
where
    T: Default + Clone + Copy + Equivalence + Ord + std::fmt::Debug,
{
    let p: Rank = comm.size();

    // Store problem size in u64 to handle very large arrays
    let mut problem_size: u64 = 0;
    let mut min_arr_size: u64 = 0;
    let arr_len: u64 = arr.len().try_into().unwrap();

    // Communicate the total problem size to each process in communicator
    comm.all_reduce_into(&arr.len(), &mut problem_size, SystemOperation::sum());
    comm.all_reduce_into(&arr.len(), &mut min_arr_size, SystemOperation::min());

    // Determine number of samples for splitters, beta=20 taken from paper
    let beta = 20;
    let numerator: f64 = (beta * k * (arr_len as Count)) as f64;
    let denominator: f64 = problem_size as f64;
    let mut split_count: Count = (numerator / denominator).ceil() as Count;
    if split_count > arr_len as Count {
        split_count = arr_len as Count;
    }

    let mut rng = rand::thread_rng();

    // Randomly sample splitters from local section of array
    let mut splitters: Vec<T> = vec![T::default(); split_count as usize];

    for i in 0..split_count {
        let mut idx: u64 = rng.gen::<u64>();
        idx %= arr_len;
        splitters[i as usize] = arr[idx as usize];
    }

    // Gather sampled splitters from all processes at each process
    let mut global_split_counts: Vec<Count> = vec![0; p as usize];

    comm.all_gather_into(&split_count, &mut global_split_counts[..]);

    let global_split_displacements: Vec<Count> = global_split_counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    let global_split_count =
        global_split_displacements[(p - 1) as usize] + global_split_counts[(p - 1) as usize];

    let mut global_splitters: Vec<T> = vec![T::default(); global_split_count as usize];

    {
        let mut partition = PartitionMut::new(
            &mut global_splitters[..],
            global_split_counts,
            &global_split_displacements[..],
        );
        comm.all_gather_varcount_into(&splitters[..], &mut partition)
    }

    // Sort the sampled splitters
    global_splitters.sort();

    // Find associated rank due to splitters locally, arr is assumed to be sorted locally
    let mut disp: Vec<u64> = vec![0; global_split_count as usize];

    for i in 0..global_split_count {
        disp[i as usize] = arr.lower_bound(&global_splitters[i as usize]) as u64;
    }

    // The global rank is found via a simple sum
    let root_rank = 0;
    let root_process = comm.process_at_rank(root_rank);
    let mut global_disp: Vec<u64> = vec![0; global_split_count as usize];

    for i in 0..(global_split_count as usize) {
        if comm.rank() == root_rank {
            comm.process_at_rank(root_rank).reduce_into_root(
                &disp[i],
                &mut global_disp[i],
                SystemOperation::sum(),
            );
        } else {
            comm.process_at_rank(root_rank)
                .reduce_into(&disp[i], SystemOperation::sum());
        }
    }

    root_process.broadcast_into(&mut global_disp);

    // We're performing a k-way split, find the keys associated with a split by comparing the
    // optimal splitters with the sampled ones
    let mut split_keys: Vec<T> = vec![T::default(); k as usize];

    for i in 0..k {
        let mut _disp = 0;
        let optimal_splitter: i64 = (((i + 1) as u64) * problem_size / (k as u64 + 1))
            .try_into()
            .unwrap();

        for j in 0..(global_split_count as usize) {
            if (global_disp[j] as i64 - optimal_splitter).abs()
                < (global_disp[_disp] as i64 - optimal_splitter).abs()
            {
                _disp = j;
            }
        }

        split_keys[i as usize] = global_splitters[_disp as usize]
    }

    split_keys.sort();
    split_keys
}

/// HykSort of Sundar et. al. without the parallel merge logic.
pub fn hyksort<T>(arr: &mut Vec<T>, mut k: Rank, mut comm: UserCommunicator)
where
    T: Default + Clone + Copy + Equivalence + Ord + std::fmt::Debug,
{
    let mut p: Rank = comm.size();
    let mut rank: Rank = comm.rank();

    // Store problem size in u64 to handle very large arrays
    let mut problem_size: u64 = 0;
    let arr_len: u64 = arr.len().try_into().unwrap();

    comm.all_reduce_into(&arr_len, &mut problem_size, SystemOperation::sum());

    // Allocate all buffers
    let mut arr_: Vec<T> = vec![T::default(); (arr_len * 2) as usize];

    // Perform local sort
    arr.sort();

    // If k is greater than size of communicator set to traditional dense all to all
    if k > p {
        k = p;
    }

    while p > 1 && problem_size > 0 {
        // Find size of color block
        let color_size = p / k;
        assert_eq!(color_size * k, p);

        let color = rank / color_size;
        let new_rank = modulo(rank, color_size);

        // Find (k-1) splitters to define a k-way split
        let tmp = comm.duplicate();
        let split_keys: Vec<T> = parallel_select(arr, &(k - 1), tmp);

        // Communicate
        {
            // Determine send size
            let mut send_size: Vec<u64> = vec![0; k as usize];
            let mut send_disp: Vec<u64> = vec![0; (k + 1) as usize];

            // Packet displacement and size to each partner process determined by the splitters found
            send_disp[k as usize] = arr.len() as u64;
            for i in 1..k {
                send_disp[i as usize] = arr.lower_bound(&split_keys[(i - 1) as usize]) as u64;
            }

            for i in 0..k {
                send_size[i as usize] = send_disp[(i + 1) as usize] - send_disp[i as usize];
            }

            // Determine receive sizes
            let mut recv_iter: u64 = 0;
            let mut recv_cnt: Vec<u64> = vec![0; k as usize];
            let mut recv_size: Vec<u64> = vec![0; k as usize];
            let mut recv_disp: Vec<u64> = vec![0; (k + 1) as usize];

            // Communicate packet sizes
            for i_ in 0..=(k / 2) {
                let i1 = modulo(color + i_, k);

                // Add k to ensure that this always works
                let i2 = modulo(color + k - i_, k);

                for j in 0..(if i_ == 0 || i_ == k / 2 { 1 } else { 2 }) {
                    let i = if i_ == 0 {
                        i1
                    } else {
                        if (j + color / i_) % 2 == 0 {
                            i1
                        } else {
                            i2
                        }
                    };

                    let partner_rank = color_size * i + new_rank;
                    let partner_process = comm.process_at_rank(partner_rank);

                    mpi::point_to_point::send_receive_into(
                        &send_size[i as usize],
                        &partner_process,
                        &mut recv_size[recv_iter as usize],
                        &partner_process,
                    );

                    recv_disp[(recv_iter + 1) as usize] =
                        recv_disp[recv_iter as usize] + recv_size[recv_iter as usize];
                    recv_cnt[recv_iter as usize] = recv_size[recv_iter as usize];
                    recv_iter += 1;
                }
            }

            // Communicate packets
            // Resize buffers
            arr_.resize(recv_disp[k as usize] as usize, T::default());

            // Reset recv_iter
            recv_iter = 0;

            for i_ in 0..=(k / 2) {
                let i1 = modulo(color + i_, k);

                // Add k to ensure that this always works
                let i2 = modulo(color + k - i_, k);

                for j in 0..(if i_ == 0 || i_ == k / 2 { 1 } else { 2 }) {
                    let i = if i_ == 0 {
                        i1
                    } else {
                        if (j + color / i_) % 2 == 0 {
                            i1
                        } else {
                            i2
                        }
                    };
                    let partner_rank = color_size * i + new_rank;
                    let partner_process = comm.process_at_rank(partner_rank);

                    // Receive packet bounds indices
                    let r_lidx: usize = recv_disp[recv_iter as usize] as usize;
                    let r_ridx: usize = r_lidx + recv_size[recv_iter as usize] as usize;
                    assert!(r_lidx <= r_ridx);

                    // Send packet bounds indices
                    let s_lidx: usize = send_disp[i as usize] as usize;
                    let s_ridx: usize = s_lidx + send_size[i as usize] as usize;
                    assert!(s_lidx <= s_ridx);

                    mpi::request::scope(|scope| {
                        let _rreq = WaitGuard::from(
                            partner_process
                                .immediate_receive_into(scope, &mut arr_[r_lidx..r_ridx]),
                        );
                        let _sreq = WaitGuard::from(
                            partner_process.immediate_synchronous_send(scope, &arr[s_lidx..s_ridx]),
                        );
                    });

                    recv_iter += 1;
                }
            }

            // Swap send and receive buffers
            std::mem::swap(arr, &mut arr_);

            // Local sort of received data
            arr.sort();

            // Split the communicator
            comm = comm
                .split_by_color(mpi::topology::Color::with_value(color))
                .unwrap();
            p = comm.size();
            rank = comm.rank();
        }
    }
}
