//! Serial function space

use crate::element::ciarlet::CiarletElement;
use crate::traits::{
    element::{ElementFamily, FiniteElement},
    grid::{CellType, EdgeType, GridType, PointType, TopologyType},
    types::Ownership,
};
use rlst::RlstScalar;
use std::collections::HashMap;

type DofList = Vec<Vec<usize>>;
type OwnerData = Vec<(usize, usize, usize, usize)>;

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

        // Assign DOFs to vertices
        for (i, e) in topology.vertex_indices().enumerate() {
            let e_dofs = element.entity_dofs(0, i).unwrap();
            if !e_dofs.is_empty() {
                if entity_dofs[0][e].is_empty() {
                    for (dof_i, _d) in e_dofs.iter().enumerate() {
                        entity_dofs[0][e].push(size);
                        if let Ownership::Ghost(process, index) =
                            grid.vertex_from_index(e).ownership()
                        {
                            owner_data.push((process, 0, index, dof_i));
                        } else {
                            owner_data.push((rank, 0, e, dof_i));
                        }
                        size += 1;
                    }
                }
                for (local_dof, dof) in e_dofs.iter().zip(&entity_dofs[0][e]) {
                    cell_dofs[cell.index()][*local_dof] = *dof;
                }
            }
        }

        // Assign DOFs to edges
        if tdim >= 1 {
            for (i, e) in topology.edge_indices().enumerate() {
                let e_dofs = element.entity_dofs(1, i).unwrap();
                if !e_dofs.is_empty() {
                    if entity_dofs[1][e].is_empty() {
                        for (dof_i, _d) in e_dofs.iter().enumerate() {
                            entity_dofs[1][e].push(size);
                            if let Ownership::Ghost(process, index) =
                                grid.edge_from_index(e).ownership()
                            {
                                owner_data.push((process, 1, index, dof_i));
                            } else {
                                owner_data.push((rank, 1, e, dof_i));
                            }
                            size += 1;
                        }
                    }
                    for (local_dof, dof) in e_dofs.iter().zip(&entity_dofs[1][e]) {
                        cell_dofs[cell.index()][*local_dof] = *dof;
                    }
                }
            }
        }

        // Assign DOFs to faces
        if tdim >= 2 {
            for (i, e) in topology.face_indices().enumerate() {
                let e_dofs = element.entity_dofs(2, i).unwrap();
                if !e_dofs.is_empty() {
                    if entity_dofs[2][e].is_empty() {
                        for (dof_i, _d) in e_dofs.iter().enumerate() {
                            entity_dofs[2][e].push(size);
                            if let Ownership::Ghost(process, index) = cell.ownership() {
                                owner_data.push((process, 2, index, dof_i));
                            } else {
                                owner_data.push((rank, 2, e, dof_i));
                            }
                            size += 1;
                        }
                    }
                    for (local_dof, dof) in e_dofs.iter().zip(&entity_dofs[2][e]) {
                        cell_dofs[cell.index()][*local_dof] = *dof;
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::element::ciarlet::{LagrangeElementFamily, RaviartThomasElementFamily};
    use crate::grid::shapes::screen_triangles;
    // use crate::grid::shapes::{screen_mixed, screen_quadrilaterals, screen_triangles};
    use crate::traits::element::Continuity;

    fn run_test(grid: &impl GridType<T = f64>, degree: usize, continuity: Continuity) {
        let family = LagrangeElementFamily::<f64>::new(degree, continuity);
        let (cell_dofs, entity_dofs, size, owner_data) = assign_dofs(0, grid, &family);

        for o in &owner_data {
            assert_eq!(o.0, 0);
        }
        for d in &cell_dofs {
            for (i, n) in d.iter().enumerate() {
                assert!(*n < size);
                for m in d.iter().skip(i + 1) {
                    assert!(*n != *m);
                }
            }
        }
        for i in &entity_dofs {
            for j in i {
                for k in j {
                    assert!(*k < size);
                }
            }
        }
    }

    fn run_test_rt(grid: &impl GridType<T = f64>, degree: usize, continuity: Continuity) {
        let family = RaviartThomasElementFamily::<f64>::new(degree, continuity);
        let (cell_dofs, entity_dofs, size, owner_data) = assign_dofs(0, grid, &family);

        for o in &owner_data {
            assert_eq!(o.0, 0);
        }
        for d in &cell_dofs {
            for (i, n) in d.iter().enumerate() {
                assert!(*n < size);
                for m in d.iter().skip(i + 1) {
                    assert!(*n != *m);
                }
            }
        }
        for i in &entity_dofs {
            for j in i {
                for k in j {
                    assert!(*k < size);
                }
            }
        }
    }

    #[test]
    fn test_dp0_triangles() {
        let grid = screen_triangles::<f64>(8);
        run_test(&grid, 0, Continuity::Discontinuous);
    }
    #[test]
    fn test_dp2_triangles() {
        let grid = screen_triangles::<f64>(8);
        run_test(&grid, 2, Continuity::Discontinuous);
    }
    #[test]
    fn test_p2_triangles() {
        let grid = screen_triangles::<f64>(8);
        run_test(&grid, 2, Continuity::Continuous);
    }
    #[test]
    fn test_p3_triangles() {
        let grid = screen_triangles::<f64>(8);
        run_test(&grid, 3, Continuity::Continuous);
    }
    #[test]
    fn test_rt1_triangles() {
        let grid = screen_triangles::<f64>(8);
        run_test_rt(&grid, 1, Continuity::Discontinuous);
    }

    // #[test]
    // fn test_dp0_quadrilaterals() {
    //     let grid = screen_quadrilaterals::<f64>(8);
    //     run_test(&grid, 0, Continuity::Discontinuous);
    // }
    // #[test]
    // fn test_dp2_quadrilaterals() {
    //     let grid = screen_quadrilaterals::<f64>(8);
    //     run_test(&grid, 2, Continuity::Discontinuous);
    // }
    // #[test]
    // fn test_p2_quadrilaterals() {
    //     let grid = screen_quadrilaterals::<f64>(8);
    //     run_test(&grid, 2, Continuity::Continuous);
    // }
    // #[test]
    // fn test_p3_quadrilaterals() {
    //     let grid = screen_quadrilaterals::<f64>(8);
    //     run_test(&grid, 3, Continuity::Continuous);
    // }

    // #[test]
    // fn test_dp0_mixed() {
    //     let grid = screen_mixed::<f64>(8);
    //     run_test(&grid, 0, Continuity::Discontinuous);
    // }
    // #[test]
    // fn test_dp2_mixed() {
    //     let grid = screen_mixed::<f64>(8);
    //     run_test(&grid, 2, Continuity::Discontinuous);
    // }
    // #[test]
    // fn test_p2_mixed() {
    //     let grid = screen_mixed::<f64>(8);
    //     run_test(&grid, 2, Continuity::Continuous);
    // }
    // #[test]
    // fn test_p3_mixed() {
    //     let grid = screen_mixed::<f64>(8);
    //     run_test(&grid, 3, Continuity::Continuous);
    // }
}
