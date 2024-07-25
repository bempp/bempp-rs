//! Functions and functions spaces
use ndgrid::traits::Grid;
#[cfg(feature = "mpi")]
use crate::traits::grid::ParallelGridType;
use ndgrid::types::Ownership;
use ndelement::traits::FiniteElement;
use ndelement::types::ReferenceCellType;
use std::collections::HashMap;

/// A function space
pub trait FunctionSpace {
    /// The grid type
    type Grid: Grid;
    /// The finite element type
    type FiniteElement: FiniteElement;

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

    /// Compute a colouring of the cells so that no two cells that share an entity with DOFs associated with it are assigned the same colour
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>>;

    /// Get the global DOF indes associated with a local DOF indec
    fn global_dof_index(&self, local_dof_index: usize) -> usize;

    /// Get ownership of a local DOF
    fn ownership(&self, local_dof_index: usize) -> Ownership;
}

#[cfg(feature = "mpi")]
/// A function space in parallel
pub trait FunctionSpaceInParallel {
    /// The parallel grid type
    type ParallelGrid: Grid + ParallelGrid;
    /// The type of the serial space on each process
    type SerialSpace: FunctionSpace<Grid = <Self::ParallelGrid as ParallelGrid>::LocalGrid>;

    /// Get the local space on the process
    fn local_space(&self) -> &Self::SerialSpace;
}
