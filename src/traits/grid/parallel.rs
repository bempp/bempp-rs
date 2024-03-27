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

pub trait ParallelGridType: GridType {
    //! An MPI parallelised grid
    /// The MPI communicator type
    type Comm: Communicator;
    /// The type of the subgrid on each process
    type LocalGridType: GridType<T = <Self as GridType>::T>;

    /// The MPI communicator
    fn comm(&self) -> &Self::Comm;

    /// The subgrid on the process
    fn local_grid(&self) -> &Self::LocalGridType;
}
