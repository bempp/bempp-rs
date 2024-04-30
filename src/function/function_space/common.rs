//! Serial function space

use crate::element::ciarlet::CiarletElement;
use crate::traits::{
    element::{ElementFamily, FiniteElement},
    grid::{CellType, GridType, TopologyType},
    types::Ownership,
};
use rlst::RlstScalar;
use std::collections::HashMap;

type DofList = Vec<Vec<usize>>;
type OwnerData = Vec<(usize, usize, usize)>;

pub(crate) fn assign_dofs<T: RlstScalar, GridImpl: GridType<T = T::Real>>(
    rank: usize,
    grid: &GridImpl,
    e_family: &impl ElementFamily<T = T, FiniteElement = CiarletElement<T>>,
) -> (DofList, [DofList; 4], usize, OwnerData) {
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

    let mut max_rank = rank;
    for cell in grid.iter_all_cells() {
        if let Ownership::Ghost(process, _index) = cell.ownership() {
            if process > max_rank {
                max_rank = process;
            }
        }
    }
    for cell in grid.iter_all_cells() {
        cell_dofs[cell.index()] = vec![0; element_dims[&cell.topology().cell_type()]];
        let element = &elements[&cell.topology().cell_type()];
        let topology = cell.topology();
        for (i, e) in topology.vertex_indices().enumerate() {
            let e_dofs = element.entity_dofs(0, i).unwrap();
            if !e_dofs.is_empty() {
                if entity_dofs[0][e].is_empty() {
                    for _d in e_dofs {
                        entity_dofs[0][e].push(size);
                        owner_data.push((max_rank + 1, 0, 0));
                        size += 1;
                    }
                }
                for (j, k) in e_dofs.iter().enumerate() {
                    cell_dofs[cell.index()][*k] = entity_dofs[0][e][j];
                    if let Ownership::Ghost(process, index) = cell.ownership() {
                        if process < owner_data[entity_dofs[0][e][j]].0 {
                            owner_data[entity_dofs[0][e][j]] = (process, index, *k);
                        }
                    } else if rank < owner_data[entity_dofs[0][e][j]].0 {
                        owner_data[entity_dofs[0][e][j]] = (rank, cell.index(), *k);
                    }
                }
            }
        }
        if tdim >= 1 {
            for (i, e) in topology.edge_indices().enumerate() {
                let e_dofs = element.entity_dofs(1, i).unwrap();
                if !e_dofs.is_empty() {
                    if entity_dofs[1][e].is_empty() {
                        for _d in e_dofs {
                            entity_dofs[1][e].push(size);
                            owner_data.push((max_rank + 1, 0, 0));
                            size += 1;
                        }
                    }
                    for (j, k) in e_dofs.iter().enumerate() {
                        cell_dofs[cell.index()][*k] = entity_dofs[1][e][j];
                        if let Ownership::Ghost(process, index) = cell.ownership() {
                            if process < owner_data[entity_dofs[1][e][j]].0 {
                                owner_data[entity_dofs[1][e][j]] = (process, index, *k);
                            }
                        } else if rank < owner_data[entity_dofs[1][e][j]].0 {
                            owner_data[entity_dofs[1][e][j]] = (rank, cell.index(), *k);
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
                        for _d in e_dofs {
                            entity_dofs[2][e].push(size);
                            owner_data.push((max_rank + 1, 0, 0));
                            size += 1;
                        }
                    }
                    for (j, k) in e_dofs.iter().enumerate() {
                        cell_dofs[cell.index()][*k] = entity_dofs[2][e][j];
                        if let Ownership::Ghost(process, index) = cell.ownership() {
                            owner_data[entity_dofs[2][e][j]] = (process, index, *k);
                        } else if rank < owner_data[entity_dofs[2][e][j]].0 {
                            owner_data[entity_dofs[2][e][j]] = (rank, cell.index(), *k);
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
