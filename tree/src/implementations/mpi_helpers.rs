/// Helper Routines for MPI functionality
use mpi::{
    traits::*,
    Count,
    datatype::Equivalence,
    topology::{UserCommunicator, Communicator},
    request::WaitGuard,
    collective::SystemOperation,
};


/// Sparse MPI_AllToAllV, i.e. each process only communicates
/// to a subset of the communicator.
pub fn all_to_allv_sparse<T>(
    world: &UserCommunicator, 
    mut packets: &Vec<Vec<T>>, 
    mut packet_destinations: &Vec<Count>,
    &recv_count: &Count
) -> Vec<T>
where
    T: Default + Clone + Equivalence,
{
    let rank = world.rank();
    let size = world.size();

    // Communicate the packet sizes to relevant destinations
    for (i, packet) in packets.iter().enumerate() {
        let msg = vec![rank, packet.len() as Count];

        let partner_process = world.process_at_rank(packet_destinations[i]);

        mpi::request::scope(|scope| {
            let _sreq = WaitGuard::from(partner_process.immediate_ready_send(scope, &msg[..]));
        })
    }

    let mut received_packet_sizes = vec![0 as Count; recv_count as usize];
    let mut receive_packet_source = vec![0 as Count; recv_count as usize];

    for i in (0..recv_count as usize) {
        let mut msg = vec![0 as Count; 2];

        mpi::request::scope(|scope| {
            let rreq = WaitGuard::from(
                world
                    .any_process()
                    .immediate_receive_into(scope, &mut msg),
            );
        });

        receive_packet_source[i] = msg[0];
        received_packet_sizes[i] = msg[1];
    }

    // Setup send and receives for data
    let mut buffers: Vec<Vec<T>> = Vec::new();

    for &len in received_packet_sizes.iter() {
        buffers.push(vec![T::default(); len as usize])
    }

    for (i, packet) in packets.iter().enumerate() {
        let partner_process = world.process_at_rank(packet_destinations[i]);

        mpi::request::scope(|scope| {
            let _sreq = WaitGuard::from(partner_process.immediate_ready_send(
                scope,
                &packet[..],
             ));
        })
    }

    for (i, recv_rank) in receive_packet_source.iter().enumerate() {
        let partner_process = world.process_at_rank(*recv_rank);

        mpi::request::scope(|scope| {
            let rreq =
                WaitGuard::from(partner_process.immediate_receive_into(scope, &mut buffers[i][..]));
        });
    }

    let result: Vec<T> = buffers.into_iter().flatten().collect();
    result
}
