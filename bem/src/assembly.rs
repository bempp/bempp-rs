pub mod dense;
use crate::green;
use crate::green::{GreenParameters, Scalar};
use bempp_tools::arrays::Array2D;
use bempp_traits::bem::FunctionSpace;

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

// TODO: template over float type

/// Assemble an operator into a dense matrix
pub fn assemble_dense<'a, T: Scalar>(
    // TODO: ouput should be `&mut impl ArrayAccess2D` once such a trait exists
    output: &mut Array2D<T>,
    operator: BoundaryOperator,
    pde: PDEType,
    trial_space: &impl FunctionSpace<'a>,
    test_space: &impl FunctionSpace<'a>,
) {
    let kernel = match pde {
        PDEType::Laplace => match operator {
            BoundaryOperator::SingleLayer => green::laplace_green,
            BoundaryOperator::DoubleLayer => green::laplace_green_dy,
            BoundaryOperator::AdjointDoubleLayer => green::laplace_green_dx,
            BoundaryOperator::Hypersingular => green::laplace_green,
            _ => {
                panic!("Invalid operator");
            }
        },
        PDEType::Helmholtz(_) => match operator {
            BoundaryOperator::SingleLayer => green::helmholtz_green,
            BoundaryOperator::DoubleLayer => green::helmholtz_green_dy,
            BoundaryOperator::AdjointDoubleLayer => green::laplace_green_dx,
            BoundaryOperator::Hypersingular => green::helmholtz_green,
            _ => {
                panic!("Invalid operator");
            }
        },
    };
    let params = match pde {
        PDEType::Laplace => GreenParameters::None,
        PDEType::Helmholtz(k) => GreenParameters::Wavenumber(k),
    };
    let needs_trial_normal = match operator {
        BoundaryOperator::DoubleLayer => true,
        _ => false,
    };
    let needs_test_normal = match operator {
        BoundaryOperator::AdjointDoubleLayer => true,
        _ => false,
    };

    if operator == BoundaryOperator::Hypersingular {
        dense::hypersingular_assemble(output, kernel, &params, trial_space, test_space);
    } else {
        dense::assemble(
            output,
            kernel,
            &params,
            needs_trial_normal,
            needs_test_normal,
            trial_space,
            test_space,
        );
    }
}

#[cfg(test)]
mod test {
    use crate::assembly::dense;
    use crate::assembly::*;
    use crate::function_space::SerialFunctionSpace;
    use crate::green::{helmholtz_green, laplace_green, GreenParameters};
    use approx::*;
    use bempp_element::element::create_element;
    use bempp_grid::shapes::regular_sphere;
    use bempp_tools::arrays::Array2D;
    use bempp_traits::arrays::Array2DAccess;
    use bempp_traits::bem::DofMap;
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::ElementFamily;
    use num::complex::Complex;

    #[test]
    fn test_laplace_single_layer() {
        let grid = regular_sphere(1);
        let element0 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            true,
        );
        let element1 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            false,
        );
        let space0 = SerialFunctionSpace::new(&grid, &element0);
        let space1 = SerialFunctionSpace::new(&grid, &element1);

        let mut matrix =
            Array2D::<f64>::new((space1.dofmap().global_size(), space0.dofmap().global_size()));
        dense::assemble(
            &mut matrix,
            laplace_green,
            &GreenParameters::None,
            false,
            false,
            &space0,
            &space1,
        );

        let mut matrix2 =
            Array2D::<f64>::new((space1.dofmap().global_size(), space0.dofmap().global_size()));

        assemble_dense(
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

    #[test]
    fn test_helmholtz_single_layer() {
        let grid = regular_sphere(1);
        let element0 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            true,
        );
        let element1 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            false,
        );
        let space0 = SerialFunctionSpace::new(&grid, &element0);
        let space1 = SerialFunctionSpace::new(&grid, &element1);

        let mut matrix = Array2D::<Complex<f64>>::new((
            space1.dofmap().global_size(),
            space0.dofmap().global_size(),
        ));
        dense::assemble(
            &mut matrix,
            helmholtz_green,
            &GreenParameters::Wavenumber(2.5),
            false,
            false,
            &space0,
            &space1,
        );

        let mut matrix2 = Array2D::<Complex<f64>>::new((
            space1.dofmap().global_size(),
            space0.dofmap().global_size(),
        ));

        assemble_dense(
            &mut matrix2,
            BoundaryOperator::SingleLayer,
            PDEType::Helmholtz(2.5),
            &space0,
            &space1,
        );

        for i in 0..space1.dofmap().global_size() {
            for j in 0..space0.dofmap().global_size() {
                assert_relative_eq!(
                    matrix.get(i, j).unwrap().re,
                    matrix2.get(i, j).unwrap().re,
                    epsilon = 0.0001
                );
                assert_relative_eq!(
                    matrix.get(i, j).unwrap().im,
                    matrix2.get(i, j).unwrap().im,
                    epsilon = 0.0001
                );
            }
        }
    }
}
