//! Helper Routines for MPI functionality
use mpi::{
    topology::UserCommunicator,
    traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Source},
    Count, Rank,
};

/// Sparse MPI_AllToAllV, i.e. each process only communicates
/// to a subset of the communicator.
///
/// For Example, you may have four processes in a communicator
/// Communicator = \[P0, P1, P2, P3\]. Each process communicates
/// with a subset of the communicator excluding itself,
/// ie. P0 -> \[P0, P2\], P1 -> \[P0\], P2-> \[\], P3 -> \[P0, P1, P2\].
/// This function expects these packets to be separated in a
/// `Vec<Vec<T>>`, their destination ranks, as well as the number of
/// packets this process expects to receive overall `recv_count`.
pub fn all_to_allv_sparse<T>(
    comm: &UserCommunicator,
    packets: &[Vec<T>],
    packet_destinations: &[Rank],
    &recv_count: &Count,
) -> Vec<T>
where
    T: Default + Clone + Equivalence,
{
    let rank = comm.rank();

    let send_count = packets.len() as Count;
    let nreqs = send_count + recv_count;

    let packet_sizes: Vec<Count> = packets.iter().map(|p| p.len() as Count).collect();

    let mut received_packet_sizes = vec![0 as Count; recv_count as usize];
    let mut received_packet_sources = vec![0 as Rank; recv_count as usize];

    mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
        for (i, &rank) in packet_destinations.iter().enumerate() {
            let tag = rank;
            let sreq =
                comm.process_at_rank(rank)
                    .immediate_send_with_tag(scope, &packet_sizes[i], tag);
            coll.add(sreq);
        }

        for (i, size) in received_packet_sizes.iter_mut().enumerate() {
            let (msg, status) = loop {
                // Spin for message availability. There is no guarantee that
                // immediate sends, even to the same process, will be immediately
                // visible to an immediate probe.
                let preq = comm.any_process().immediate_matched_probe_with_tag(rank);
                if let Some(p) = preq {
                    break p;
                }
            };

            let rreq = msg.immediate_matched_receive_into(scope, size);
            received_packet_sources[i] = status.source_rank();

            coll.add(rreq);
        }

        let mut complete = vec![];
        coll.wait_all(&mut complete);
    });

    // Setup send and receives for data
    let mut buffers: Vec<Vec<T>> = Vec::new();

    for &len in received_packet_sizes.iter() {
        buffers.push(vec![T::default(); len as usize])
    }

    comm.barrier();

    mpi::request::multiple_scope(nreqs as usize, |scope, coll| {
        for (i, packet) in packets.iter().enumerate() {
            let sreq = comm
                .process_at_rank(packet_destinations[i])
                .immediate_send_with_tag(scope, &packet[..], packet_destinations[i]);
            coll.add(sreq);
        }

        for (i, buffer) in buffers.iter_mut().enumerate() {
            let rreq = comm
                .process_at_rank(received_packet_sources[i])
                .immediate_receive_into(scope, &mut buffer[..]);

            coll.add(rreq);
        }

        let mut complete = vec![];
        coll.wait_all(&mut complete);
    });

    buffers.into_iter().flatten().collect()
}
