//! Functions and functions spaces
#[cfg(feature = "mpi")]
use mpi::traits::Communicator;
use ndelement::{traits::FiniteElement, types::ReferenceCellType};
#[cfg(feature = "mpi")]
use ndgrid::traits::ParallelGrid;
use ndgrid::{traits::Grid, types::Ownership};
use rlst::RlstScalar;
use std::collections::HashMap;

/// A function space
pub trait FunctionSpace {
    /// Scalar type
    type T: RlstScalar;
    /// The grid type
    type Grid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType>;
    /// The finite element type
    type FiniteElement: FiniteElement<T = Self::T> + Sync;

    /// Get the grid that the element is defined on
    fn grid(&self) -> &Self::Grid;

    /// Get the finite element used to define this function space
    fn element(&self, cell_type: ReferenceCellType) -> &Self::FiniteElement;

    /// Check if the function space is stored in serial
    fn is_serial(&self) -> bool {
        // self.grid().is_serial()
        true
    }

    /// Get the DOF numbers on the local process associated with the given entity
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize];

    /// Get the number of DOFs associated with the local process
    fn local_size(&self) -> usize;

    /// Get the number of DOFs on all processes
    fn global_size(&self) -> usize;

    /// Get the local DOF numbers associated with a cell
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]>;

    /// Get the local DOF numbers associated with a cell
    ///
    /// # Safety
    /// The function uses unchecked array access
    unsafe fn cell_dofs_unchecked(&self, cell: usize) -> &[usize];

    /// Compute a colouring of the cells so that no two cells that share an entity with DOFs associated with it are assigned the same colour
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>>;

    /// Get the global DOF index associated with a local DOF index
    fn global_dof_index(&self, local_dof_index: usize) -> usize;

    /// Get ownership of a local DOF
    fn ownership(&self, local_dof_index: usize) -> Ownership;
}

#[cfg(feature = "mpi")]
/// A function space in parallel
pub trait ParallelFunctionSpace<C: Communicator>: FunctionSpace {
    /// Parallel grid type
    type ParallelGrid: ParallelGrid<C>
        + Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType>;
    /// The type of the serial space on each process
    type LocalSpace<'a>: FunctionSpace<T = Self::T> + Sync
    where
        Self: 'a;

    /// MPI communicator
    fn comm(&self) -> &C;

    /// Get the grid
    fn grid(&self) -> &Self::ParallelGrid;

    /// Get the local space on the process
    fn local_space(&self) -> &Self::LocalSpace<'_>;
}
