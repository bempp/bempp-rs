/// Helper Routines for MPI functionality
use mpi::{
    collective::SystemOperation,
    datatype::Equivalence,
    request::{RequestCollection, Scope, WaitGuard},
    topology::{Communicator, UserCommunicator},
    traits::*,
    Count, Rank,
};

/// Sparse MPI_AllToAllV, i.e. each process only communicates
/// to a subset of the communicator.
pub fn all_to_allv_sparse<T>(
    world: &UserCommunicator,
    mut packets: &Vec<Vec<T>>,
    mut packet_destinations: &Vec<Rank>,
    &recv_count: &Count,
) -> Vec<T>
where
    T: Default + Clone + Equivalence,
{
    let rank = world.rank();
    let size = world.size();

    let send_count = packets.len() as Count;
    let nreqs = send_count + recv_count;
    let mut packet_sizes: Vec<usize> = Vec::new();

    for p in packets.iter() {
        packet_sizes.push(p.len())
    }

    // Communicate the packet sizes to relevant destinationss
    for (i, packet) in packets.iter().enumerate() {
        // Tag with destination rank
        let tag = packet_destinations[i];
        // Send packet size
        let msg = packet.len() as Count;
        let partner_process = world.process_at_rank(packet_destinations[i]);

        mpi::request::scope(|scope| {
            let _sreq = WaitGuard::from(partner_process.immediate_send_with_tag(scope, &msg, tag));
        })
    }

    let mut received_packet_sizes = vec![0 as Count; recv_count as usize];
    let mut received_packet_sources = vec![0 as Rank; recv_count as usize];

    let mut msg = 0 as Count;
    let mut source_rank = 0 as Rank;

    for i in (0..recv_count as usize) {
        // Look for messages destined for this process
        mpi::request::scope(|scope| {
            let status = world.any_process().probe_with_tag(rank);
            source_rank = status.source_rank();
            // let _rreq = WaitGuard::from(world.process_at_rank(source_rank).immediate_receive_into_with_tag(scope, &mut msg, rank));
        });
        let _rreq = world
            .process_at_rank(source_rank)
            .receive_into_with_tag(&mut msg, rank);

        received_packet_sources[i] = source_rank;
        received_packet_sizes[i] = msg;
    }

    // Setup send and receives for data
    let mut buffers: Vec<Vec<T>> = Vec::new();

    for &len in received_packet_sizes.iter() {
        buffers.push(vec![T::default(); len as usize])
    }

    // mpi::request::multiple_scope(nreqs as usize, |scope, coll| {

    //     for (i, packet) in packets.iter().enumerate() {
    //         let sreq = world
    //             .process_at_rank(packet_destinations[i])
    //             .immediate_send(scope, &packet[..]);
    //             coll.add(sreq);
    //     }

    //     for (i, buffer) in buffers.iter_mut().enumerate() {
    //         let rreq = world
    //             .process_at_rank(received_packet_sources[i])
    //             .immediate_receive_into(scope, &mut buffer[..]);
    //         coll.add(rreq);
    //     }

    //     let mut out = vec![];
    //     coll.wait_all(&mut out);
    //     assert_eq!(out.len(), nreqs as usize);
    // });

    buffers.into_iter().flatten().collect()
}
