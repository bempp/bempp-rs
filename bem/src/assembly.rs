pub mod batched;
pub mod cl_kernel;
use crate::function_space::SerialFunctionSpace;
use bempp_kernel::laplace_3d;
use bempp_tools::arrays::Mat;

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

/// Assemble an operator into a dense matrix using batched parallelisation
pub fn assemble_batched<'a>(
    // TODO: ouput should be `&mut impl ArrayAccess2D` once such a trait exists
    output: &mut Mat<f64>,
    operator: BoundaryOperator,
    pde: PDEType,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    match pde {
        PDEType::Laplace => match operator {
            BoundaryOperator::SingleLayer => {
                batched::assemble(
                    output,
                    &laplace_3d::Laplace3dKernel::new(),
                    false,
                    false,
                    trial_space,
                    test_space,
                );
            }
            _ => {
                panic!("Invalid operator");
            }
        },
        _ => {
            panic!("Invalid PDE");
        }
    };
}

pub fn assemble_cl<'a>(
    output: &mut Mat<f64>,
    operator: BoundaryOperator,
    pde: PDEType,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    match pde {
        PDEType::Laplace => match operator {
            BoundaryOperator::SingleLayer => {
                cl_kernel::assemble(
                    output,
                    &laplace_3d::Laplace3dKernel::new(),
                    false,
                    false,
                    trial_space,
                    test_space,
                );
            }
            _ => {
                panic!("Invalid operator");
            }
        },
        _ => {
            panic!("Invalid PDE");
        }
    };
}

#[cfg(test)]
mod test {
    use crate::assembly::batched;
    use crate::assembly::*;
    use crate::function_space::SerialFunctionSpace;
    use approx::*;
    use bempp_element::element::create_element;
    use bempp_grid::shapes::regular_sphere;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tools::arrays::zero_matrix;
    use bempp_traits::bem::DofMap;
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::{Continuity, ElementFamily};
    // use num::complex::Complex;
    use bempp_traits::bem::FunctionSpace;
    use rlst_dense::RandomAccessByRef;

    #[test]
    fn test_laplace_single_layer() {
        let grid = regular_sphere(1);
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

        let mut matrix =
            zero_matrix::<f64>((space1.dofmap().global_size(), space0.dofmap().global_size()));
        batched::assemble(
            &mut matrix,
            &Laplace3dKernel::new(),
            false,
            false,
            &space0,
            &space1,
        );

        let mut matrix2 =
            zero_matrix::<f64>((space1.dofmap().global_size(), space0.dofmap().global_size()));

        assemble_batched(
            &mut matrix2,
            BoundaryOperator::SingleLayer,
            PDEType::Laplace,
            &space0,
            &space1,
        );

        for i in 0..space1.dofmap().global_size() {
            for j in 0..space0.dofmap().global_size() {
                assert_relative_eq!(
                    *matrix.get(i, j).unwrap(),
                    *matrix2.get(i, j).unwrap(),
                    epsilon = 0.0001
                );
            }
        }
    }
}
