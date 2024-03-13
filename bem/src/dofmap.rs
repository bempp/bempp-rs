//! DOF map
use bempp_traits::bem::DofMap;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{CellType, GridType, TopologyType};

/// A serial DOF map
pub struct SerialDofMap {
    entity_dofs: [Vec<Vec<usize>>; 4],
    cell_dofs: Vec<Vec<usize>>,
    size: usize,
}

impl SerialDofMap {
    /// Create new DOF map
    pub fn new(grid: &impl GridType, element: &impl FiniteElement) -> Self {
        let mut size = 0;
        let mut entity_dofs: [Vec<Vec<usize>>; 4] = [vec![], vec![], vec![], vec![]];
        let tdim = grid.domain_dimension();

        let mut entity_counts = vec![];
        entity_counts.push(grid.number_of_vertices());
        if tdim > 1 {
            entity_counts.push(grid.number_of_edges());
        }
        if tdim > 2 {
            unimplemented!("DOF maps not implemented for cells with tdim > 2.");
        }
        entity_counts.push(grid.number_of_cells());

        for d in 0..tdim + 1 {
            entity_dofs[d] = vec![vec![]; entity_counts[d]];
        }
        let mut cell_dofs = vec![vec![0; element.dim()]; entity_counts[tdim]];

        for cell in grid.iter_all_cells() {
            let topology = cell.topology();
            for (i, e) in topology.vertex_indices().enumerate() {
                let e_dofs = element.entity_dofs(0, i).unwrap();
                if !e_dofs.is_empty() {
                    if entity_dofs[0][e].is_empty() {
                        for _ in e_dofs {
                            entity_dofs[0][e].push(size);
                            size += 1
                        }
                    }
                    for (j, k) in e_dofs.iter().enumerate() {
                        cell_dofs[cell.index()][*k] = entity_dofs[0][e][j];
                    }
                }
            }
            if tdim >= 1 {
                for (i, e) in topology.edge_indices().enumerate() {
                    let e_dofs = element.entity_dofs(1, i).unwrap();
                    if !e_dofs.is_empty() {
                        if entity_dofs[1][e].is_empty() {
                            for _ in e_dofs {
                                entity_dofs[1][e].push(size);
                                size += 1
                            }
                        }
                        for (j, k) in e_dofs.iter().enumerate() {
                            cell_dofs[cell.index()][*k] = entity_dofs[1][e][j];
                        }
                    }
                }
            }
            if tdim >= 2 {
                for (i, e) in topology.face_indices().enumerate() {
                    let e_dofs = element.entity_dofs(2, i).unwrap();
                    if !e_dofs.is_empty() {
                        if entity_dofs[2][e].is_empty() {
                            for _ in e_dofs {
                                entity_dofs[2][e].push(size);
                                size += 1
                            }
                        }
                        for (j, k) in e_dofs.iter().enumerate() {
                            cell_dofs[cell.index()][*k] = entity_dofs[2][e][j];
                        }
                    }
                }
            }
            if tdim >= 3 {
                unimplemented!("DOF maps not implemented for cells with tdim > 2.");
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
        &self.entity_dofs[entity_dim][entity_number]
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
        if cell < self.cell_dofs.len() {
            Some(&self.cell_dofs[cell])
        } else {
            None
        }
    }
    fn is_serial(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod test {
    use crate::dofmap::*;
    use bempp_element::element::{create_element, ElementFamily};
    use bempp_grid::shapes::regular_sphere;
    use bempp_traits::element::Continuity;
    use bempp_traits::types::ReferenceCellType;

    #[test]
    fn test_dofmap_lagrange0() {
        let grid = regular_sphere::<f64>(2);
        let element = create_element::<f64>(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let dofmap = SerialDofMap::new(&grid, &element);
        assert_eq!(dofmap.local_size(), dofmap.global_size());
        assert_eq!(dofmap.local_size(), grid.number_of_cells());
    }

    #[test]
    fn test_dofmap_lagrange1() {
        let grid = regular_sphere::<f64>(2);
        let element = create_element::<f64>(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let dofmap = SerialDofMap::new(&grid, &element);
        assert_eq!(dofmap.local_size(), dofmap.global_size());
        assert_eq!(dofmap.local_size(), grid.number_of_vertices());
    }

    #[test]
    fn test_dofmap_lagrange2() {
        let grid = regular_sphere::<f64>(2);
        let element = create_element::<f64>(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            2,
            Continuity::Continuous,
        );
        let dofmap = SerialDofMap::new(&grid, &element);
        assert_eq!(dofmap.local_size(), dofmap.global_size());
        assert_eq!(
            dofmap.local_size(),
            grid.number_of_vertices() + grid.number_of_edges()
        );
    }
}
