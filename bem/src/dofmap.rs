use bempp_traits::grid::Grid;
use bempp_traits::element::FiniteElement;

pub trait DofMap {
    /// Get the DOF numbers on the local process associated with the given entity
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize];

    /// Get the global DOF numbers associated with the given entity
    fn get_global_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize];
    
}

pub struct SerialDofMap {
}

impl SerialDofMap {
    pub fn new(grid: impl Grid, element: impl FiniteElement) -> Self {
        Self { }
    }

}

impl DofMap for SerialDofMap {
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        &[]
    }
    fn get_global_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        self.get_local_dof_numbers(entity_dim, entity_number)
    }
}



#[cfg(test)]
mod test {
    use crate::dofmap::*;
    use bempp_grid::shapes::regular_sphere;
    use bempp_element::element::LagrangeElementTriangleDegree0;

    #[test]
    fn test_dofmap() {
        let grid = regular_sphere(2);
        let element = LagrangeElementTriangleDegree0 { };
        let dofmap = SerialDofMap::new(grid, element);
    }
}
