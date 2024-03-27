//! MPI parallelised grid

use super::{Builder, GridType};
use mpi::traits::Communicator;
use std::collections::HashMap;

pub trait ParallelBuilder<const GDIM: usize>: Builder<GDIM> {
    //! Trait to build a MPI parellelised mesh from a grid builder.

    /// The type of the parallel grid that the builder creates
    type ParallelGridType<'a, C: Communicator + 'a>: GridType;

    /// Create the grid on the main process
    fn create_parallel_grid<'a, C: Communicator>(
        self,
        comm: &'a C,
        cell_owners: &HashMap<usize, usize>,
    ) -> Self::ParallelGridType<'a, C>;

    /// Create the grid on a subprocess
    fn receive_parallel_grid<C: Communicator>(
        self,
        comm: &C,
        root_process: usize,
    ) -> Self::ParallelGridType<'_, C>;
}
