use bempp_tools::arrays::{AdjacencyList, Array2D};
use bempp_traits::arrays::{AdjacencyListAccess, Array2DAccess};
use bempp_traits::bem::DofMap;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Grid, Topology};

pub struct SerialDofMap {
    entity_dofs: [AdjacencyList<usize>; 4],
    cell_dofs: Array2D<usize>,
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
        let mut cell_dofs =
            Array2D::<usize>::new((grid.topology().entity_count(tdim), element.dim()));
        for cell in 0..grid.topology().entity_count(tdim) {
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
                        for (j, k) in e_dofs.iter().enumerate() {
                            *cell_dofs.get_mut(cell, *k).unwrap() = entity_dofs_data[d][*e][j];
                        }
                    }
                }
            }
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
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]> {
        self.cell_dofs.row(cell)
    }
    fn is_serial(&self) -> bool {
        true
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
