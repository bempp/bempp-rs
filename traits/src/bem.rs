use crate::element::FiniteElement;
use crate::grid::Grid;
use num::Num;

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

    // Check if the function space is stored in serial
    fn is_serial(&self) -> bool;
}

pub trait FunctionSpace<'a> {
    type DofMap: DofMap;
    type Grid: Grid<'a>;
    type FiniteElement: FiniteElement;

    /// Get the function space's DOF map
    fn dofmap(&self) -> &Self::DofMap;

    /// Get the grid that the element is defined on
    fn grid(&self) -> &Self::Grid;

    /// Get the finite element used to define this function space
    fn element(&self) -> &Self::FiniteElement;

    // Check if the function space is stored in serial
    fn is_serial(&self) -> bool {
        self.dofmap().is_serial() && self.grid().is_serial()
    }
}

pub trait Kernel<T: Num> {
    fn test_element_dim(&self) -> usize;
    fn trial_element_dim(&self) -> usize;
    fn test_geometry_element_dim(&self) -> usize;
    fn trial_geometry_element_dim(&self) -> usize;
    fn same_cell_kernel(&self, result: &mut [T], test_vertices: &[T], trial_vertices: &[T]);
    fn shared_edge_kernel(&self, result: &mut [T], test_vertices: &[T], trial_vertices: &[T], qid: u8);
    fn shared_vertex_kernel(&self, result: &mut [T], test_vertices: &[T], trial_vertices: &[T], qid: u8);
    fn nonneighbour_kernel(&self, result: &mut [T], test_vertices: &[T], trial_vertices: &[T]);
}
