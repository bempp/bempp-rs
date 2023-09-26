use crate::dofmap::SerialDofMap;
use bempp_grid::grid::SerialGrid;
use bempp_traits::arrays::AdjacencyListAccess;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Grid, Topology};

pub struct SerialFunctionSpace<'a, E: FiniteElement> {
    grid: &'a SerialGrid,
    element: &'a E,
    dofmap: SerialDofMap,
}

impl<'a, E: FiniteElement> SerialFunctionSpace<'a, E> {
    pub fn new(grid: &'a SerialGrid, element: &'a E) -> Self {
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
        while self.element.entity_dofs(edim, 0).unwrap().len() == 0 {
            edim += 1;
        }
        let cell_entities = self
            .grid
            .topology()
            .connectivity(self.grid.topology().dim(), edim);
        for i in 0..self
            .grid
            .topology()
            .entity_count(self.grid.topology().dim())
        {
            let vs = cell_entities.row(i).unwrap();
            let mut c = 0;
            while c < colouring.len() {
                let mut found = false;
                for cell in &colouring[c] {
                    let cell_vs = cell_entities.row(*cell).unwrap();
                    for v in vs {
                        if cell_vs.contains(v) {
                            found = true;
                            break;
                        }
                    }
                    if found {
                        break;
                    }
                }
                if !found {
                    colouring[c].push(i);
                    break;
                }
                c += 1;
            }
            if c == colouring.len() {
                colouring.push(vec![i]);
            }
        }
        colouring
    }
}

impl<'a, E: FiniteElement> FunctionSpace<'a> for SerialFunctionSpace<'a, E> {
    type DofMap = SerialDofMap;
    type Grid = SerialGrid;
    type FiniteElement = E;

    fn dofmap(&self) -> &Self::DofMap {
        &self.dofmap
    }
    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self) -> &E {
        self.element
    }
}

#[cfg(test)]
mod test {
    use crate::function_space::*;
    use bempp_element::element::create_element;
    use bempp_grid::shapes::regular_sphere;
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::{Continuity, ElementFamily};

    #[test]
    fn test_colouring_p1() {
        let grid = regular_sphere(2);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);
        let colouring = space.compute_cell_colouring();
        let c20 = grid.topology().connectivity(2, 0);
        let mut n = 0;
        for i in &colouring {
            n += i.len()
        }
        assert_eq!(n, grid.topology().entity_count(2));
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
                        for v0 in c20.row(*cell0).unwrap() {
                            for v1 in c20.row(*cell1).unwrap() {
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
        let grid = regular_sphere(2);
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
        assert_eq!(n, grid.topology().entity_count(2));
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
        let grid = regular_sphere(2);
        let element = create_element(
            ElementFamily::RaviartThomas,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);
        let colouring = space.compute_cell_colouring();
        let c21 = grid.topology().connectivity(2, 1);
        let mut n = 0;
        for i in &colouring {
            n += i.len()
        }
        assert_eq!(n, grid.topology().entity_count(2));
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
                        for e0 in c21.row(*cell0).unwrap() {
                            for e1 in c21.row(*cell1).unwrap() {
                                assert!(e0 != e1);
                            }
                        }
                    }
                }
            }
        }
    }
}
