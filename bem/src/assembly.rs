//! Boundary operator assembly
pub mod batched;
pub mod common;
pub mod fmm_tools;

#[cfg(test)]
mod test {
    use super::*;
    use super::batched::BatchedAssembler;
    use crate::function_space::SerialFunctionSpace;
    use bempp_element::element::lagrange;
    use bempp_grid::{shapes::regular_sphere, single_element_grid::{SerialSingleElementGrid, SerialSingleElementGridBuilder}, traits_impl::WrappedGrid};
    use bempp_traits::{element::Continuity, bem::FunctionSpace, types::ReferenceCellType, grid::Builder};
    use rlst::{rlst_dynamic_array2, RlstScalar, Array, BaseArray, VectorContainer, MatrixInverse};
    use cauchy::{c32, c64};
    use paste::paste;
    use num::Float;

    fn quadrilateral_grid<T: Float + RlstScalar<Real=T>>() -> WrappedGrid<SerialSingleElementGrid<T>>
    where for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse
    {
        let mut b = SerialSingleElementGridBuilder::<3, T>::new((ReferenceCellType::Quadrilateral, 1));
        for j in 0..4 {
            for i in 0..4 {
                b.add_point(4*j + i, [
num::cast::<i32, T>(i).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
num::cast::<i32, T>(j).unwrap() / num::cast::<f64, T>(3.0).unwrap(), num::cast::<f64, T>(0.0).unwrap()]);
            }
        }
        for j in 0..3 {
            for i in 0..3 {
                b.add_cell(3*j + i, vec![4*j + i, 4*j + i + 1, 4*j + i + 4, 4*j + i + 5]);
            }
        }
        b.create_grid()
    }

    macro_rules! example_grid {
        (Triangle, $dtype:ident) => { regular_sphere(0) };
        (Quadrilateral, $dtype:ident) => { quadrilateral_grid::<<$dtype as RlstScalar>::Real>() };
    }
    macro_rules! create_assembler {
        (Laplace, $operator:ident, $dtype:ident) => {paste!{
            batched::[<Laplace $operator Assembler>]::<128, [<$dtype>]>::default()
        }};
        (Helmholtz, $operator:ident, $dtype:ident) => {paste! {
            batched::[<Helmholtz $operator Assembler>]::<128, [<$dtype>]>::new(3.0)
        }};
    }
    macro_rules! test_assembly {

        ($(($dtype:ident, $pde:ident, $operator:ident, $cell:ident)),+) => {

        $(
            paste! {

                #[test]
                fn [<test_assembly_ $pde:lower _ $operator:lower _ $cell:lower _ $dtype>]() {

                    let grid = example_grid!($cell, $dtype);
                    let element = lagrange::create(ReferenceCellType::[<$cell>], 0, Continuity::Discontinuous);
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
        (f64, Laplace, SingleLayer, Triangle),
        (f32, Laplace, SingleLayer, Triangle),
        (c64, Laplace, SingleLayer, Triangle),
        (c32, Laplace, SingleLayer, Triangle),
        (f64, Laplace, DoubleLayer, Triangle),
        (f32, Laplace, DoubleLayer, Triangle),
        (c64, Laplace, DoubleLayer, Triangle),
        (c32, Laplace, DoubleLayer, Triangle),
        (f64, Laplace, AdjointDoubleLayer, Triangle),
        (f32, Laplace, AdjointDoubleLayer, Triangle),
        (c64, Laplace, AdjointDoubleLayer, Triangle),
        (c32, Laplace, AdjointDoubleLayer, Triangle),
        (f64, Laplace, Hypersingular, Triangle),
        (f32, Laplace, Hypersingular, Triangle),
        (c64, Laplace, Hypersingular, Triangle),
        (c32, Laplace, Hypersingular, Triangle),
        (f64, Laplace, SingleLayer, Quadrilateral),
        (f32, Laplace, SingleLayer, Quadrilateral),
        (c64, Laplace, SingleLayer, Quadrilateral),
        (c32, Laplace, SingleLayer, Quadrilateral),
        (f64, Laplace, DoubleLayer, Quadrilateral),
        (f32, Laplace, DoubleLayer, Quadrilateral),
        (c64, Laplace, DoubleLayer, Quadrilateral),
        (c32, Laplace, DoubleLayer, Quadrilateral),
        (f64, Laplace, AdjointDoubleLayer, Quadrilateral),
        (f32, Laplace, AdjointDoubleLayer, Quadrilateral),
        (c64, Laplace, AdjointDoubleLayer, Quadrilateral),
        (c32, Laplace, AdjointDoubleLayer, Quadrilateral),
        (f64, Laplace, Hypersingular, Quadrilateral),
        (f32, Laplace, Hypersingular, Quadrilateral),
        (c64, Laplace, Hypersingular, Quadrilateral),
        (c32, Laplace, Hypersingular, Quadrilateral),
        (c64, Helmholtz, SingleLayer, Triangle),
        (c32, Helmholtz, SingleLayer, Triangle),
        (c64, Helmholtz, DoubleLayer, Triangle),
        (c32, Helmholtz, DoubleLayer, Triangle),
        (c64, Helmholtz, AdjointDoubleLayer, Triangle),
        (c32, Helmholtz, AdjointDoubleLayer, Triangle),
        (c64, Helmholtz, Hypersingular, Triangle),
        (c32, Helmholtz, Hypersingular, Triangle),
        (c64, Helmholtz, SingleLayer, Quadrilateral),
        (c32, Helmholtz, SingleLayer, Quadrilateral),
        (c64, Helmholtz, DoubleLayer, Quadrilateral),
        (c32, Helmholtz, DoubleLayer, Quadrilateral),
        (c64, Helmholtz, AdjointDoubleLayer, Quadrilateral),
        (c32, Helmholtz, AdjointDoubleLayer, Quadrilateral),
        (c64, Helmholtz, Hypersingular, Quadrilateral),
        (c32, Helmholtz, Hypersingular, Quadrilateral)
    );

}
