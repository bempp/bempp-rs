//! Boundary operator assembly
pub mod batched;
pub mod common;
pub mod fmm_tools;

#[cfg(test)]
mod test {
    use self::batched::BatchedAssembler;
    use super::*;
    use crate::function_space::SerialFunctionSpace;
    use bempp_element::element::{create_element, ElementFamily};
    use bempp_grid::shapes::regular_sphere;
    use bempp_traits::bem::DofMap;
    use bempp_traits::bem::FunctionSpace;
    use bempp_traits::element::Continuity;
    use bempp_traits::types::ReferenceCellType;
    use rlst_dense::rlst_dynamic_array2;

    #[test]
    fn test_laplace_single_layer() {
        let grid = regular_sphere::<f64>(1);
        let element0 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let element1 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space0 = SerialFunctionSpace::new(&grid, &element0);
        let space1 = SerialFunctionSpace::new(&grid, &element1);

        let mut matrix = rlst_dynamic_array2!(
            f64,
            [space1.dofmap().global_size(), space0.dofmap().global_size()]
        );
        let a = batched::LaplaceSingleLayerAssembler::default();
        a.assemble_into_dense::<128, _, _>(&mut matrix, &space0, &space1);
    }
}
