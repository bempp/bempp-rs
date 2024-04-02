//! Parallel function space

use crate::bem::function_space::SerialFunctionSpace;
use crate::element::ciarlet::CiarletElement;
use crate::traits::{
    bem::FunctionSpace,
    element::ElementFamily,
    grid::{GridType, ParallelGridType},
    types::ReferenceCellType,
};
use rlst::RlstScalar;
use std::collections::HashMap;

/// A parallel function space
pub struct ParallelFunctionSpace<
    'a,
    T: RlstScalar,
    GridImpl: ParallelGridType + GridType<T = T::Real>,
> {
    grid: &'a GridImpl,
    local_space: SerialFunctionSpace<'a, T, <GridImpl as ParallelGridType>::LocalGridType>,
    global_size: usize,
}

impl<'a, T: RlstScalar, GridImpl: ParallelGridType + GridType<T = T::Real>>
    ParallelFunctionSpace<'a, T, GridImpl>
{
    /// Create new function space
    pub fn new(
        grid: &'a GridImpl,
        e_family: &impl ElementFamily<T = T, FiniteElement = CiarletElement<T>>,
    ) -> Self {

        let comm = grid.comm();


        let global_size = 100;
        Self {
            grid,
            local_space: SerialFunctionSpace::new(grid.local_grid(), e_family),
            global_size,
        }
    }
    /// Get the local space on a process
    pub fn local_space(
        &self,
    ) -> &SerialFunctionSpace<'a, T, <GridImpl as ParallelGridType>::LocalGridType> {
        &self.local_space
    }
}

impl<'a, T: RlstScalar, GridImpl: ParallelGridType + GridType<T = T::Real>> FunctionSpace
    for ParallelFunctionSpace<'a, T, GridImpl>
{
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<T>;

    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self, cell_type: ReferenceCellType) -> &CiarletElement<T> {
        self.local_space.element(cell_type)
    }
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        self.local_space
            .get_local_dof_numbers(entity_dim, entity_number)
    }
    fn local_size(&self) -> usize {
        self.local_space.local_size()
    }
    fn get_global_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        self.local_space
            .get_global_dof_numbers(entity_dim, entity_number)
    }
    fn global_size(&self) -> usize {
        self.global_size
    }
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]> {
        self.local_space.cell_dofs(cell)
    }
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>> {
        self.local_space.cell_colouring()
    }
}
