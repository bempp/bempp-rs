//! Boundary operator assembly
pub mod boundary;
pub(crate) mod common;
pub mod fmm_tools;
pub mod kernels;
pub mod potential;

#[cfg(test)]
mod test {
    use super::*;
    use crate::function::SerialFunctionSpace;
    use crate::traits::{BoundaryAssembly, FunctionSpace};
    use cauchy::{c32, c64};
    use ndelement::ciarlet::CiarletElement;
    use ndelement::ciarlet::LagrangeElementFamily;
    use ndelement::types::{Continuity, ReferenceCellType};
    use ndgrid::{
        grid::serial::{SingleElementGrid, SingleElementGridBuilder},
        shapes::regular_sphere,
        traits::Builder,
        types::RealScalar,
    };
    use paste::paste;
    use rlst::{rlst_dynamic_array2, MatrixInverse, RlstScalar};

    fn quadrilateral_grid<T: RealScalar + MatrixInverse>() -> SingleElementGrid<T, CiarletElement<T>>
    {
        let mut b = SingleElementGridBuilder::<T>::new(3, (ReferenceCellType::Quadrilateral, 1));
        for j in 0..4 {
            for i in 0..4 {
                b.add_point(
                    4 * j + i,
                    &[
                        num::cast::<usize, T>(i).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
                        num::cast::<usize, T>(j).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
                        num::cast::<f64, T>(0.0).unwrap(),
                    ],
                );
            }
        }
        for j in 0..3 {
            for i in 0..3 {
                b.add_cell(
                    3 * j + i,
                    &[4 * j + i, 4 * j + i + 1, 4 * j + i + 4, 4 * j + i + 5],
                );
            }
        }
        b.create_grid()
    }

    /*
    fn mixed_grid<T: Float + RlstScalar<Real = T>>() -> MixedGrid<T>
    where
        for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>:
            MatrixInverse,
    {
        let mut b = MixedGridBuilder::<3, T>::new(());
        for j in 0..4 {
            for i in 0..4 {
                b.add_point(
                    4 * j + i,
                    [
                        num::cast::<usize, T>(i).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
                        num::cast::<usize, T>(j).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
                        num::cast::<f64, T>(0.0).unwrap(),
                    ],
                );
            }
        }
        for j in 0..3 {
            b.add_cell(
                j,
                (
                    vec![4 * j, 4 * j + 1, 4 * j + 4, 4 * j + 5],
                    ReferenceCellType::Quadrilateral,
                    1,
                ),
            );
        }
        for j in 0..3 {
            b.add_cell(
                3 + 2 * j,
                (
                    vec![4 * j + 1, 4 * j + 2, 4 * j + 6],
                    ReferenceCellType::Triangle,
                    1,
                ),
            );
            b.add_cell(
                4 + 2 * j,
                (
                    vec![4 * j + 1, 4 * j + 6, 4 * j + 5],
                    ReferenceCellType::Triangle,
                    1,
                ),
            );
        }
        for j in 0..3 {
            b.add_cell(
                9 + j,
                (
                    vec![4 * j + 2, 4 * j + 3, 4 * j + 6, 4 * j + 7],
                    ReferenceCellType::Quadrilateral,
                    1,
                ),
            );
        }
        b.create_grid()
    }
    */

    macro_rules! example_grid {
        (Triangle, $dtype:ident) => {
            regular_sphere(0)
        };
        (Quadrilateral, $dtype:ident) => {
            quadrilateral_grid::<<$dtype as RlstScalar>::Real>()
        }; //(Mixed, $dtype:ident) => {
           //    mixed_grid::<<$dtype as RlstScalar>::Real>()
           //};
    }
    macro_rules! create_assembler {
        (Laplace, Hypersingular, $dtype:ident) => {
            paste! {
                boundary::HypersingularAssembler::<[<$dtype>], _, _>::new_laplace()
            }
        };
        (Helmholtz, Hypersingular, $dtype:ident) => {
            paste! {
                boundary::HypersingularAssembler::<[<$dtype>], _, _>::new_helmholtz(3.0)
            }
        };
        (Laplace, $operator:ident, $dtype:ident) => {
            paste! {
                boundary::BoundaryAssembler::<[<$dtype>], _, _>::[<new_laplace_ $operator>]()
            }
        };
        (Helmholtz, $operator:ident, $dtype:ident) => {
            paste! {
                boundary::BoundaryAssembler::<[<$dtype>], _, _>::[<new_helmholtz_ $operator>](3.0)
            }
        };
    }
    macro_rules! test_assembly {

        ($(($dtype:ident, $pde:ident, $operator:ident, $cell:ident)),+) => {

        $(
            paste! {

                #[test]
                fn [<test_assembly_ $pde:lower _ $operator:lower _ $cell:lower _ $dtype>]() {

                    let grid = example_grid!($cell, $dtype);
                    let element = LagrangeElementFamily::<[<$dtype>]>::new(0, Continuity::Discontinuous);
                    let space = SerialFunctionSpace::new(&grid, &element);

                    let ndofs = space.global_size();
                    let mut matrix = rlst_dynamic_array2!([<$dtype>], [ndofs, ndofs]);

                    let a = create_assembler!($pde, $operator, $dtype);
                    a.assemble_into_dense(&mut matrix, &space, &space);
                }

            }
        )*
        };
    }

    test_assembly!(
        (f64, Laplace, single_layer, Triangle),
        (f32, Laplace, single_layer, Triangle),
        //(c64, Laplace, single_layer, Triangle),
        //(c32, Laplace, single_layer, Triangle),
        (f64, Laplace, double_layer, Triangle),
        (f32, Laplace, double_layer, Triangle),
        //(c64, Laplace, double_layer, Triangle),
        //(c32, Laplace, double_layer, Triangle),
        (f64, Laplace, adjoint_double_layer, Triangle),
        (f32, Laplace, adjoint_double_layer, Triangle),
        //(c64, Laplace, adjoint_double_layer, Triangle),
        //(c32, Laplace, adjoint_double_layer, Triangle),
        (f64, Laplace, hypersingular, Triangle),
        (f32, Laplace, hypersingular, Triangle),
        //(c64, Laplace, hypersingular, Triangle),
        //(c32, Laplace, hypersingular, Triangle),
        (c64, Helmholtz, single_layer, Triangle),
        (c32, Helmholtz, single_layer, Triangle),
        (c64, Helmholtz, double_layer, Triangle),
        (c32, Helmholtz, double_layer, Triangle),
        (c64, Helmholtz, adjoint_double_layer, Triangle),
        (c32, Helmholtz, adjoint_double_layer, Triangle),
        (c64, Helmholtz, hypersingular, Triangle),
        (c32, Helmholtz, hypersingular, Triangle),
        (f64, Laplace, single_layer, Quadrilateral),
        (f32, Laplace, single_layer, Quadrilateral),
        //(c64, Laplace, single_layer, Quadrilateral),
        //(c32, Laplace, single_layer, Quadrilateral),
        (f64, Laplace, double_layer, Quadrilateral),
        (f32, Laplace, double_layer, Quadrilateral),
        //(c64, Laplace, double_layer, Quadrilateral),
        //(c32, Laplace, double_layer, Quadrilateral),
        (f64, Laplace, adjoint_double_layer, Quadrilateral),
        (f32, Laplace, adjoint_double_layer, Quadrilateral),
        //(c64, Laplace, adjoint_double_layer, Quadrilateral),
        //(c32, Laplace, adjoint_double_layer, Quadrilateral),
        (f64, Laplace, hypersingular, Quadrilateral),
        (f32, Laplace, hypersingular, Quadrilateral),
        //(c64, Laplace, hypersingular, Quadrilateral),
        //(c32, Laplace, hypersingular, Quadrilateral),
        (c64, Helmholtz, single_layer, Quadrilateral),
        (c32, Helmholtz, single_layer, Quadrilateral),
        (c64, Helmholtz, double_layer, Quadrilateral),
        (c32, Helmholtz, double_layer, Quadrilateral),
        (c64, Helmholtz, adjoint_double_layer, Quadrilateral),
        (c32, Helmholtz, adjoint_double_layer, Quadrilateral),
        (c64, Helmholtz, hypersingular, Quadrilateral),
        (c32, Helmholtz, hypersingular, Quadrilateral) //(f64, Laplace, single_layer, Mixed),
                                                       //(f32, Laplace, single_layer, Mixed),
                                                       //(c64, Laplace, single_layer, Mixed),
                                                       //(c32, Laplace, single_layer, Mixed),
                                                       //(f64, Laplace, double_layer, Mixed),
                                                       //(f32, Laplace, double_layer, Mixed),
                                                       //(c64, Laplace, double_layer, Mixed),
                                                       //(c32, Laplace, double_layer, Mixed),
                                                       //(f64, Laplace, adjoint_double_layer, Mixed),
                                                       //(f32, Laplace, adjoint_double_layer, Mixed),
                                                       //(c64, Laplace, adjoint_double_layer, Mixed),
                                                       //(c32, Laplace, adjoint_double_layer, Mixed),
                                                       //(f64, Laplace, hypersingular, Mixed),
                                                       //(f32, Laplace, hypersingular, Mixed),
                                                       //(c64, Laplace, hypersingular, Mixed),
                                                       //(c32, Laplace, hypersingular, Mixed),
                                                       //(c64, Helmholtz, single_layer, Mixed),
                                                       //(c32, Helmholtz, single_layer, Mixed),
                                                       //(c64, Helmholtz, double_layer, Mixed),
                                                       //(c32, Helmholtz, double_layer, Mixed),
                                                       //(c64, Helmholtz, adjoint_double_layer, Mixed),
                                                       //(c32, Helmholtz, adjoint_double_layer, Mixed),
                                                       //(c64, Helmholtz, hypersingular, Mixed),
                                                       //(c32, Helmholtz, hypersingular, Mixed)
    );
}
