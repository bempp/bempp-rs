//! Boundary element method traits
use crate::element::FiniteElement;
use crate::grid::GridType;
use crate::types::ReferenceCellType;
use std::collections::HashMap;

/// A function space
pub trait FunctionSpace {
    /// The grid type
    type Grid: GridType;
    /// The finite element type
    type FiniteElement: FiniteElement;

    /// Get the grid that the element is defined on
    fn grid(&self) -> &Self::Grid;

    /// Get the finite element used to define this function space
    fn element(&self, cell_type: ReferenceCellType) -> &Self::FiniteElement;

    /// Check if the function space is stored in serial
    fn is_serial(&self) -> bool {
        self.grid().is_serial()
    }

    /// Get the DOF numbers on the local process associated with the given entity
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize];

    /// Get the global DOF numbers associated with the given entity
    fn get_global_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize];

    /// Get the number of DOFs associated with the local process
    fn local_size(&self) -> usize;

    /// Get the number of DOFs on all processes
    fn global_size(&self) -> usize;

    /// Get the local DOF numbers associated with a cell
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]>;

    /// Compute a colouring of the cells so that no two cells that share an entity with DOFs associated with it are assigned the same colour
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>>;
}
