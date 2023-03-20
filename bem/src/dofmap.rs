use bempp_tools::arrays::AdjacencyList;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Grid, Topology};

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
    fn cell_dofs(&self, cell: usize) -> &[usize];
}

pub struct SerialDofMap {
    entity_dofs: [AdjacencyList<usize>; 4],
    cell_dofs: Vec<Vec<usize>>, // TODO: use 2darray
    size: usize,
}

impl SerialDofMap {
    pub fn new(grid: &impl Grid, element: &impl FiniteElement) -> Self {
        let mut size = 0;
        let mut entity_dofs_data: Vec<Vec<Vec<usize>>> = vec![];
        let tdim = grid.topology().dim();
        for d in 0..tdim + 1 {
            entity_dofs_data.push(vec![vec![]; grid.topology().entity_count(d)]);
        }
        let mut cell_dofs = vec![];
        for cell in 0..grid.topology().entity_count(tdim) {
            let mut dofs = vec![0; element.dim()];
            for d in 0..tdim + 1 {
                for (i, e) in unsafe {
                    grid.topology()
                        .connectivity(tdim, d)
                        .row_unchecked(cell)
                        .iter()
                        .enumerate()
                } {
                    let e_dofs = element.entity_dofs(d, i);
                    if e_dofs.len() > 0 {
                        if entity_dofs_data[d][*e].len() == 0 {
                            for _ in &e_dofs {
                                entity_dofs_data[d][*e].push(size);
                                size += 1
                            }
                        }
                        assert_eq!(entity_dofs_data[d][*e].len(), e_dofs.len()); // TODO: debug assert?
                        for (j, k) in e_dofs.iter().enumerate() {
                            dofs[*k] = entity_dofs_data[d][*e][j];
                        }
                    }
                }
            }
            cell_dofs.push(dofs);
        }

        let mut entity_dofs = [
            AdjacencyList::<usize>::new(),
            AdjacencyList::<usize>::new(),
            AdjacencyList::<usize>::new(),
            AdjacencyList::<usize>::new(),
        ];
        for d in 0..tdim + 1 {
            for row in &entity_dofs_data[d] {
                entity_dofs[d].add_row(row);
            }
        }

        Self {
            entity_dofs,
            cell_dofs,
            size,
        }
    }
}

impl DofMap for SerialDofMap {
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        &self.entity_dofs[entity_dim].row(entity_number).unwrap()
    }
    fn local_size(&self) -> usize {
        self.size
    }
    fn get_global_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        self.get_local_dof_numbers(entity_dim, entity_number)
    }
    fn global_size(&self) -> usize {
        self.local_size()
    }
    fn cell_dofs(&self, cell: usize) -> &[usize] {
        &self.cell_dofs[cell]
    }
}

#[cfg(test)]
mod test {
    use crate::dofmap::*;
    use bempp_element::element::{
        LagrangeElementTriangleDegree0, LagrangeElementTriangleDegree1,
        LagrangeElementTriangleDegree2,
    };
    use bempp_grid::shapes::regular_sphere;

    #[test]
    fn test_dofmap_lagrange0() {
        let grid = regular_sphere(2);
        let element = LagrangeElementTriangleDegree0 {};
        let dofmap = SerialDofMap::new(&grid, &element);
        assert_eq!(dofmap.local_size(), dofmap.global_size());
        assert_eq!(
            dofmap.local_size(),
            grid.topology().entity_count(grid.topology().dim())
        );
    }

    #[test]
    fn test_dofmap_lagrange1() {
        let grid = regular_sphere(2);
        let element = LagrangeElementTriangleDegree1 {};
        let dofmap = SerialDofMap::new(&grid, &element);
        assert_eq!(dofmap.local_size(), dofmap.global_size());
        assert_eq!(dofmap.local_size(), grid.topology().entity_count(0));
    }

    #[test]
    fn test_dofmap_lagrange2() {
        let grid = regular_sphere(2);
        let element = LagrangeElementTriangleDegree2 {};
        let dofmap = SerialDofMap::new(&grid, &element);
        assert_eq!(dofmap.local_size(), dofmap.global_size());
        assert_eq!(
            dofmap.local_size(),
            grid.topology().entity_count(0) + grid.topology().entity_count(1)
        );
    }
}
