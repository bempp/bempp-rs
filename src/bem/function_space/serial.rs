//! Serial function space

use crate::element::ciarlet::CiarletElement;
use crate::traits::{
    bem::FunctionSpace,
    element::{ElementFamily, FiniteElement},
    grid::{CellType, GridType, TopologyType},
    types::{Ownership, ReferenceCellType},
};
use rlst::RlstScalar;
use std::collections::HashMap;

/// A serial function space
pub struct SerialFunctionSpace<'a, T: RlstScalar, GridImpl: GridType<T = T::Real>> {
    grid: &'a GridImpl,
    elements: HashMap<ReferenceCellType, CiarletElement<T>>,
    entity_dofs: [Vec<Vec<usize>>; 4],
    cell_dofs: Vec<Vec<usize>>,
    size: usize,
    global_dof_numbers: Vec<usize>,
}

pub(crate) fn assign_dofs<T: RlstScalar, GridImpl: GridType<T = T::Real>>(
    grid: &GridImpl,
    e_family: &impl ElementFamily<T = T, FiniteElement = CiarletElement<T>>,
) -> (Vec<Vec<usize>>, [Vec<Vec<usize>>; 4], usize, Vec<(usize, usize, usize)>){
    let mut size = 0;
    let mut entity_dofs: [Vec<Vec<usize>>; 4] = [vec![], vec![], vec![], vec![]];
    let mut owner_data = vec![];
    let tdim = grid.domain_dimension();

    let mut elements = HashMap::new();
    let mut element_dims = HashMap::new();
    for cell in grid.cell_types() {
        elements.insert(*cell, e_family.element(*cell));
        element_dims.insert(*cell, elements[cell].dim());
    }

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
    let mut cell_dofs = vec![vec![]; entity_counts[tdim]];

    for cell in grid.iter_all_cells() {
        cell_dofs[cell.index()] = vec![0; element_dims[&cell.topology().cell_type()]];
        let element = &elements[&cell.topology().cell_type()];
        let topology = cell.topology();
        for (i, e) in topology.vertex_indices().enumerate() {
            let e_dofs = element.entity_dofs(0, i).unwrap();
            if !e_dofs.is_empty() {
                if entity_dofs[0][e].is_empty() {
                    for d in e_dofs {
                        entity_dofs[0][e].push(size);
                        owner_data.push((grid.mpi_rank(), cell.index(), *d));
                        size += 1;
                    }
                }
                for (j, k) in e_dofs.iter().enumerate() {
                    cell_dofs[cell.index()][*k] = entity_dofs[0][e][j];
                    if let Ownership::Ghost(process, index) = cell.ownership() {
                        if process < owner_data[entity_dofs[0][e][j]].0 {
                            owner_data[entity_dofs[0][e][j]] = (process, index, *k);
                        }
                    }
                }
            }
        }
        if tdim >= 1 {
            for (i, e) in topology.edge_indices().enumerate() {
                let e_dofs = element.entity_dofs(1, i).unwrap();
                if !e_dofs.is_empty() {
                    if entity_dofs[1][e].is_empty() {
                        for d in e_dofs {
                            entity_dofs[1][e].push(size);
                            owner_data.push((grid.mpi_rank(), cell.index(), *d));
                            size += 1;
                        }
                    }
                    for (j, k) in e_dofs.iter().enumerate() {
                        cell_dofs[cell.index()][*k] = entity_dofs[1][e][j];
                        if let Ownership::Ghost(process, index) = cell.ownership() {
                            if process < owner_data[entity_dofs[0][e][j]].0 {
                                owner_data[entity_dofs[0][e][j]] = (process, index, *k);
                            }
                        }
                    }
                }
            }
        }
        if tdim >= 2 {
            for (i, e) in topology.face_indices().enumerate() {
                let e_dofs = element.entity_dofs(2, i).unwrap();
                if !e_dofs.is_empty() {
                    if entity_dofs[2][e].is_empty() {
                        for d in e_dofs {
                            entity_dofs[2][e].push(size);
                            owner_data.push((grid.mpi_rank(), cell.index(), *d));
                            size += 1;
                        }
                    }
                    for (j, k) in e_dofs.iter().enumerate() {
                        cell_dofs[cell.index()][*k] = entity_dofs[2][e][j];
                        if let Ownership::Ghost(process, index) = cell.ownership() {
                            if process < owner_data[entity_dofs[0][e][j]].0 {
                                owner_data[entity_dofs[0][e][j]] = (process, index, *k);
                            }
                        }
                    }
                }
            }
        }
        if tdim >= 3 {
            unimplemented!("DOF maps not implemented for cells with tdim > 2.");
        }
    }
    (cell_dofs, entity_dofs, size, owner_data)
}

impl<'a, T: RlstScalar, GridImpl: GridType<T = T::Real>> SerialFunctionSpace<'a, T, GridImpl> {
    /// Create new function space
    pub fn new(
        grid: &'a GridImpl,
        e_family: &impl ElementFamily<T = T, FiniteElement = CiarletElement<T>>,
    ) -> Self {
        let (cell_dofs, entity_dofs, size, _) = assign_dofs(grid, e_family);

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
            global_dof_numbers: (0..size).collect::<Vec<_>>(),
        }
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
    fn element(&self, cell_type: ReferenceCellType) -> &CiarletElement<T> {
        &self.elements[&cell_type]
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
