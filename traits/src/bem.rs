use crate::element::FiniteElement;
use crate::grid::Grid;

pub trait DofMap {
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
}

pub trait FunctionSpace {
    type DofMap: DofMap;
    type Grid: Grid;
    type FiniteElement: FiniteElement;

    fn dof_map(&self) -> &Self::DofMap;
    fn grid(&self) -> &Self::Grid;
    fn element(&self) -> &Self::FiniteElement;
}
