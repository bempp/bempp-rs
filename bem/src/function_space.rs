use crate::dofmap::SerialDofMap;
use bempp_grid::grid::SerialGrid;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::FiniteElement;

pub struct SerialFunctionSpace<'a, E: FiniteElement> {
    grid: &'a SerialGrid,
    element: E,
    dofmap: SerialDofMap,
}

impl<'a, E: FiniteElement> SerialFunctionSpace<'a, E> {
    pub fn new(grid: &'a SerialGrid, element: E) -> Self {
        let dofmap = SerialDofMap::new(grid, &element);
        Self {
            grid,
            element,
            dofmap,
        }
    }
}

impl<E: FiniteElement> FunctionSpace for SerialFunctionSpace<'_, E> {
    type DofMap = SerialDofMap;
    type Grid = SerialGrid;
    type FiniteElement = E;

    fn dof_map(&self) -> &Self::DofMap {
        &self.dofmap
    }
    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self) -> &E {
        &self.element
    }
}
