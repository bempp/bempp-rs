pub mod batched;
pub mod common;
pub mod fmm_tools;
use crate::assembly::batched::BatchedAssembler;
use crate::function_space::SerialFunctionSpace;
use bempp_traits::grid::GridType;
use rlst_dense::{
    array::Array, base_array::BaseArray, data_container::VectorContainer, types::RlstScalar,
};
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum BoundaryOperator {
    SingleLayer,
    DoubleLayer,
    AdjointDoubleLayer,
    Hypersingular,
    ElectricField,
    MagneticField,
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum PDEType {
    Laplace,
    Helmholtz(f64),
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum AssemblyType {
    Dense,
}

/// Assemble an operator into a dense matrix using batched parallelisation
pub fn assemble<
    'a,
    T: RlstScalar,
    TestGrid: GridType<T = T::Real> + Sync,
    TrialGrid: GridType<T = T::Real> + Sync,
>(
    output: &mut Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
    atype: AssemblyType,
    operator: BoundaryOperator,
    pde: PDEType,
    trial_space: &SerialFunctionSpace<'a, T, TrialGrid>,
    test_space: &SerialFunctionSpace<'a, T, TestGrid>,
) {
    match atype {
        AssemblyType::Dense => match pde {
            PDEType::Laplace => match operator {
                BoundaryOperator::SingleLayer => {
                    let a = batched::LaplaceSingleLayerAssembler::default();
                    a.assemble_into_dense::<128, TestGrid, TrialGrid>(
                        output,
                        trial_space,
                        test_space,
                    )
                }
                BoundaryOperator::DoubleLayer => {
                    let a = batched::LaplaceDoubleLayerAssembler::default();
                    a.assemble_into_dense::<128, TestGrid, TrialGrid>(
                        output,
                        trial_space,
                        test_space,
                    )
                }
                BoundaryOperator::AdjointDoubleLayer => {
                    let a = batched::LaplaceAdjointDoubleLayerAssembler::default();
                    a.assemble_into_dense::<128, TestGrid, TrialGrid>(
                        output,
                        trial_space,
                        test_space,
                    )
                }
                _ => {
                    panic!("Unsupported operator");
                }
            },
            _ => {
                panic!("Unsupported PDE");
            }
        },
    }
}

#[cfg(test)]
mod test {
    use crate::assembly::batched;
    use crate::assembly::*;
    use crate::function_space::SerialFunctionSpace;
    use approx::*;
    use bempp_element::element::{create_element, ElementFamily};
    use bempp_grid::{
        flat_triangle_grid::SerialFlatTriangleGrid, shapes::regular_sphere,
        traits_impl::WrappedGrid,
    };
    use bempp_traits::bem::DofMap;
    use bempp_traits::element::Continuity;
    use bempp_traits::types::ReferenceCellType;
    // use num::complex::Complex;
    use bempp_traits::bem::FunctionSpace;
    use rlst_dense::{rlst_dynamic_array2, traits::RandomAccessByRef};

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
        a.assemble_into_dense::<128, WrappedGrid<SerialFlatTriangleGrid<f64>>, WrappedGrid<SerialFlatTriangleGrid<f64>>>(&mut matrix, &space0, &space1);

        let mut matrix2 = rlst_dynamic_array2!(
            f64,
            [space1.dofmap().global_size(), space0.dofmap().global_size()]
        );

        assemble(
            &mut matrix2,
            AssemblyType::Dense,
            BoundaryOperator::SingleLayer,
            PDEType::Laplace,
            &space0,
            &space1,
        );

        for i in 0..space1.dofmap().global_size() {
            for j in 0..space0.dofmap().global_size() {
                assert_relative_eq!(
                    *matrix.get([i, j]).unwrap(),
                    *matrix2.get([i, j]).unwrap(),
                    epsilon = 0.0001
                );
            }
        }
    }
}
