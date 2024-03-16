//! Funciton space

use bempp_element::element::CiarletElement;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{CellType, GridType, TopologyType};
use rlst::RlstScalar;

/// A serial function space
pub struct SerialFunctionSpace<'a, T: RlstScalar, GridImpl: GridType<T = T::Real>> {
    grid: &'a GridImpl,
    element: &'a CiarletElement<T>,
    entity_dofs: [Vec<Vec<usize>>; 4],
    cell_dofs: Vec<Vec<usize>>,
    size: usize,
}

impl<'a, T: RlstScalar, GridImpl: GridType<T = T::Real>> SerialFunctionSpace<'a, T, GridImpl> {
    /// Create new function space
    pub fn new(grid: &'a GridImpl, element: &'a CiarletElement<T>) -> Self {
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
            grid,
            element,
            entity_dofs,
            cell_dofs,
            size,
        }
    }

    /// Compute cell colouring
    pub fn compute_cell_colouring(&self) -> Vec<Vec<usize>> {
        let mut colouring: Vec<Vec<usize>> = vec![];
        let mut edim = 0;
        while self.element.entity_dofs(edim, 0).unwrap().is_empty() {
            edim += 1;
        }

        let mut entity_colours = vec![
            vec![];
            if edim == 0 {
                self.grid.number_of_vertices()
            } else if edim == 1 {
                self.grid.number_of_edges()
            } else if edim == 2 && self.grid.domain_dimension() == 2 {
                self.grid.number_of_cells()
            } else {
                unimplemented!();
            }
        ];

        for cell in self.grid.iter_all_cells() {
            let indices = if edim == 0 {
                cell.topology().vertex_indices().collect::<Vec<_>>()
            } else if edim == 1 {
                cell.topology().edge_indices().collect::<Vec<_>>()
            } else if edim == 2 {
                cell.topology().face_indices().collect::<Vec<_>>()
            } else {
                unimplemented!();
            };

            let c = {
                let mut c = 0;
                while c < colouring.len() {
                    let mut found = false;
                    for v in &indices {
                        if entity_colours[*v].contains(&c) {
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        break;
                    }
                    c += 1;
                }
                c
            };
            if c == colouring.len() {
                colouring.push(vec![cell.index()]);
            } else {
                colouring[c].push(cell.index());
            }
            for v in &indices {
                entity_colours[*v].push(c);
            }
        }
        colouring
    }
}

impl<'a, T: RlstScalar, GridImpl: GridType<T = T::Real>> FunctionSpace
    for SerialFunctionSpace<'a, T, GridImpl>
{
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<T>;

    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self) -> &CiarletElement<T> {
        self.element
    }
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
}

#[cfg(test)]
mod test {
    use crate::function_space::*;
    use bempp_element::element::lagrange;
    use bempp_grid::shapes::regular_sphere;
    use bempp_traits::element::Continuity;
    use bempp_traits::grid::{CellType, TopologyType};
    use bempp_traits::types::ReferenceCellType;

    #[test]
    fn test_dofmap_lagrange0() {
        let grid = regular_sphere::<f64>(2);
        let element =
            lagrange::create::<f64>(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        assert_eq!(space.local_size(), space.global_size());
        assert_eq!(space.local_size(), grid.number_of_cells());
    }

    #[test]
    fn test_dofmap_lagrange1() {
        let grid = regular_sphere::<f64>(2);
        let element =
            lagrange::create::<f64>(ReferenceCellType::Triangle, 1, Continuity::Continuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        assert_eq!(space.local_size(), space.global_size());
        assert_eq!(space.local_size(), grid.number_of_vertices());
    }

    #[test]
    fn test_dofmap_lagrange2() {
        let grid = regular_sphere::<f64>(2);
        let element =
            lagrange::create::<f64>(ReferenceCellType::Triangle, 2, Continuity::Continuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        assert_eq!(space.local_size(), space.global_size());
        assert_eq!(
            space.local_size(),
            grid.number_of_vertices() + grid.number_of_edges()
        );
    }

    #[test]
    fn test_colouring_p1() {
        let grid = regular_sphere::<f64>(2);
        let element =
            lagrange::create::<f64>(ReferenceCellType::Triangle, 1, Continuity::Continuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        let colouring = space.compute_cell_colouring();
        let cells = grid.iter_all_cells().collect::<Vec<_>>();
        let mut n = 0;
        for i in &colouring {
            n += i.len()
        }
        assert_eq!(n, grid.number_of_cells());
        for (i, ci) in colouring.iter().enumerate() {
            for (j, cj) in colouring.iter().enumerate() {
                if i != j {
                    for cell0 in ci {
                        for cell1 in cj {
                            assert!(cell0 != cell1);
                        }
                    }
                }
            }
        }
        for ci in colouring {
            for cell0 in &ci {
                for cell1 in &ci {
                    if cell0 != cell1 {
                        for v0 in cells[*cell0].topology().vertex_indices() {
                            for v1 in cells[*cell1].topology().vertex_indices() {
                                assert!(v0 != v1);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_colouring_dp0() {
        let grid = regular_sphere::<f64>(2);
        let element =
            lagrange::create::<f64>(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        let colouring = space.compute_cell_colouring();
        let mut n = 0;
        for i in &colouring {
            n += i.len()
        }
        assert_eq!(n, grid.number_of_cells());
        for (i, ci) in colouring.iter().enumerate() {
            for (j, cj) in colouring.iter().enumerate() {
                if i != j {
                    for cell0 in ci {
                        for cell1 in cj {
                            assert!(cell0 != cell1);
                        }
                    }
                }
            }
        }
        assert_eq!(colouring.len(), 1);
    }

    #[test]
    fn test_colouring_rt1() {
        let grid = regular_sphere::<f64>(2);
        let element =
            lagrange::create::<f64>(ReferenceCellType::Triangle, 1, Continuity::Continuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        let colouring = space.compute_cell_colouring();
        let cells = grid.iter_all_cells().collect::<Vec<_>>();
        let mut n = 0;
        for i in &colouring {
            n += i.len()
        }
        assert_eq!(n, grid.number_of_cells());
        for (i, ci) in colouring.iter().enumerate() {
            for (j, cj) in colouring.iter().enumerate() {
                if i != j {
                    for cell0 in ci {
                        for cell1 in cj {
                            assert!(cell0 != cell1);
                        }
                    }
                }
            }
        }
        for ci in colouring {
            for cell0 in &ci {
                for cell1 in &ci {
                    if cell0 != cell1 {
                        for e0 in cells[*cell0].topology().edge_indices() {
                            for e1 in cells[*cell1].topology().edge_indices() {
                                assert!(e0 != e1);
                            }
                        }
                    }
                }
            }
        }
    }
}
