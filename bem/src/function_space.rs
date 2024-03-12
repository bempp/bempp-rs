//! Funciton space

use crate::dofmap::SerialDofMap;
use bempp_element::element::CiarletElement;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{CellType, GridType, TopologyType};
use std::collections::HashMap;

pub struct SerialFunctionSpace<'a, GridImpl: GridType> {
    grid: &'a GridImpl,
    element: &'a CiarletElement<f64>,
    dofmap: SerialDofMap,
}

impl<'a, GridImpl: GridType> SerialFunctionSpace<'a, GridImpl> {
    pub fn new(grid: &'a GridImpl, element: &'a CiarletElement<f64>) -> Self {
        let dofmap = SerialDofMap::new(grid, element);
        Self {
            grid,
            element,
            dofmap,
        }
    }

    pub fn compute_cell_colouring(&self) -> Vec<Vec<usize>> {
        let mut colouring: Vec<Vec<usize>> = vec![];
        let mut edim = 0;
        while self.element.entity_dofs(edim, 0).unwrap().is_empty() {
            edim += 1;
        }

        let mut entity_colours: HashMap<GridImpl::IndexType, Vec<usize>> = HashMap::new();

        for cell in self.grid.iter_all_cells() {
            let indices = if edim == 0 {
                cell.topology().vertex_indices().collect::<Vec<_>>()
            } else if edim == 1 {
                cell.topology().edge_indices().collect::<Vec<_>>()
            } else if edim == 2 {
                cell.topology().face_indices().collect::<Vec<_>>()
            } else {
                panic!("");
            };

            let c = {
                let mut c = 0;
                while c < colouring.len() {
                    let mut found = false;
                    for v in &indices {
                        if let Some(vc) = entity_colours.get(v) {
                            if vc.contains(&c) {
                                found = true;
                                break;
                            }
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
                if let Some(vc) = entity_colours.get_mut(v) {
                    vc.push(c);
                } else {
                    entity_colours.insert(*v, vec![c]);
                }
            }
        }
        colouring
    }
}

impl<'a, GridImpl: GridType> FunctionSpace for SerialFunctionSpace<'a, GridImpl> {
    type DofMap = SerialDofMap;
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<f64>;

    fn dofmap(&self) -> &Self::DofMap {
        &self.dofmap
    }
    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self) -> &CiarletElement<f64> {
        self.element
    }
}

#[cfg(test)]
mod test {
    use crate::function_space::*;
    use bempp_element::element::{create_element, ElementFamily};
    use bempp_grid::shapes::regular_sphere;
    use bempp_traits::element::Continuity;
    use bempp_traits::grid::{CellType, TopologyType};
    use bempp_traits::types::ReferenceCellType;

    #[test]
    fn test_colouring_p1() {
        let grid = regular_sphere::<f64>(2);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
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
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
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
        let element = create_element(
            ElementFamily::RaviartThomas,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
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
