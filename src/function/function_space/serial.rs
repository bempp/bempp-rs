//! Serial function space

use crate::element::ciarlet::CiarletElement;
use crate::function::function_space::assign_dofs;
use crate::traits::{
    element::{ElementFamily, FiniteElement},
    function::FunctionSpace,
    grid::{CellType, Grid, TopologyType},
    types::{Ownership, ReferenceCellType},
};
use rlst::RlstScalar;
use std::collections::HashMap;

/// A serial function space
pub struct SerialFunctionSpace<'a, T: RlstScalar, GridImpl: Grid<T = T::Real>> {
    pub(crate) grid: &'a GridImpl,
    pub(crate) elements: HashMap<ReferenceCellType, CiarletElement<T>>,
    pub(crate) entity_dofs: [Vec<Vec<usize>>; 4],
    pub(crate) cell_dofs: Vec<Vec<usize>>,
    pub(crate) size: usize,
}

impl<'a, T: RlstScalar, GridImpl: Grid<T = T::Real>> SerialFunctionSpace<'a, T, GridImpl> {
    /// Create new function space
    pub fn new(
        grid: &'a GridImpl,
        e_family: &impl ElementFamily<T = T, FiniteElement = CiarletElement<T>>,
    ) -> Self {
        let (cell_dofs, entity_dofs, size, _) = assign_dofs(0, grid, e_family);

        let mut elements = HashMap::new();
        for cell in grid.cell_types() {
            elements.insert(*cell, e_family.element(*cell));
        }

        Self {
            grid,
            elements,
            entity_dofs,
            cell_dofs,
            size,
        }
    }
}

impl<'a, T: RlstScalar, GridImpl: Grid<T = T::Real>> FunctionSpace
    for SerialFunctionSpace<'a, T, GridImpl>
{
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<T>;

    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self, cell_type: ReferenceCellType) -> &CiarletElement<T> {
        &self.elements[&cell_type]
    }
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        &self.entity_dofs[entity_dim][entity_number]
    }
    fn local_size(&self) -> usize {
        self.size
    }
    fn global_size(&self) -> usize {
        self.size
    }
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]> {
        if cell < self.cell_dofs.len() {
            Some(&self.cell_dofs[cell])
        } else {
            None
        }
    }
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>> {
        let mut colouring = HashMap::new();
        //: HashMap<ReferenceCellType, Vec<Vec<usize>>>
        for cell in self.grid.cell_types() {
            colouring.insert(*cell, vec![]);
        }
        let mut edim = 0;
        while self.elements[&self.grid.cell_types()[0]]
            .entity_dofs(edim, 0)
            .unwrap()
            .is_empty()
        {
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
            let cell_type = cell.topology().cell_type();
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
                while c < colouring[&cell_type].len() {
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
            if c == colouring[&cell_type].len() {
                for ct in self.grid.cell_types() {
                    colouring.get_mut(ct).unwrap().push(if *ct == cell_type {
                        vec![cell.index()]
                    } else {
                        vec![]
                    });
                }
            } else {
                colouring.get_mut(&cell_type).unwrap()[c].push(cell.index());
            }
            for v in &indices {
                entity_colours[*v].push(c);
            }
        }
        colouring
    }
    fn global_dof_index(&self, local_dof_index: usize) -> usize {
        local_dof_index
    }
    fn ownership(&self, _local_dof_index: usize) -> Ownership {
        Ownership::Owned
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::element::ciarlet::LagrangeElementFamily;
    use crate::grid::shapes::regular_sphere;
    use crate::traits::element::Continuity;

    #[test]
    fn test_dofmap_lagrange0() {
        let grid = regular_sphere::<f64>(2);
        let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        assert_eq!(space.local_size(), space.global_size());
        assert_eq!(space.local_size(), grid.number_of_cells());
    }

    #[test]
    fn test_dofmap_lagrange1() {
        let grid = regular_sphere::<f64>(2);
        let element = LagrangeElementFamily::<f64>::new(1, Continuity::Continuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        assert_eq!(space.local_size(), space.global_size());
        assert_eq!(space.local_size(), grid.number_of_vertices());
    }

    #[test]
    fn test_dofmap_lagrange2() {
        let grid = regular_sphere::<f64>(2);
        let element = LagrangeElementFamily::<f64>::new(2, Continuity::Continuous);
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
        let element = LagrangeElementFamily::<f64>::new(1, Continuity::Continuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        let colouring = &space.cell_colouring()[&ReferenceCellType::Triangle];
        let cells = grid.iter_all_cells().collect::<Vec<_>>();
        let mut n = 0;
        for i in colouring {
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
            for cell0 in ci {
                for cell1 in ci {
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
        let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        let colouring = &space.cell_colouring()[&ReferenceCellType::Triangle];
        let mut n = 0;
        for i in colouring {
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
        let element = LagrangeElementFamily::<f64>::new(1, Continuity::Continuous);
        let space = SerialFunctionSpace::new(&grid, &element);
        let colouring = &space.cell_colouring()[&ReferenceCellType::Triangle];
        let cells = grid.iter_all_cells().collect::<Vec<_>>();
        let mut n = 0;
        for i in colouring {
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
            for cell0 in ci {
                for cell1 in ci {
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
