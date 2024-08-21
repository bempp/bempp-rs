//! Assembly of boundary operators
mod assemblers;
pub(crate) mod cell_pair_assemblers;
mod integrands;

pub use assemblers::BoundaryAssembler;

#[cfg(test)]
mod test {
    use super::*;
    use crate::function::SerialFunctionSpace;
    use crate::traits::{BoundaryAssembly, FunctionSpace};
    use approx::*;
    use ndelement::ciarlet::LagrangeElementFamily;
    use ndelement::types::Continuity;
    use ndgrid::shapes::regular_sphere;
    use rlst::rlst_dynamic_array2;
    use rlst::RandomAccessByRef;

    #[test]
    fn test_singular_dp0() {
        let grid = regular_sphere::<f64>(0);
        let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
        let assembler = BoundaryAssembler::<f64, _, _>::new_laplace_single_layer();
        assembler.assemble_singular_into_dense(&mut matrix, &space, &space);
        let csr = assembler.assemble_singular_into_csr(&space, &space);

        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_singular_p1() {
        let grid = regular_sphere::<f64>(0);
        let element = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
        let assembler = BoundaryAssembler::<f64, _, _>::new_laplace_single_layer();
        assembler.assemble_singular_into_dense(&mut matrix, &space, &space);
        let csr = assembler.assemble_singular_into_csr(&space, &space);

        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_singular_dp0_p1() {
        let grid = regular_sphere::<f64>(0);
        let element0 = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
        let element1 = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space0 = SerialFunctionSpace::new(&grid, &element0);
        let space1 = SerialFunctionSpace::new(&grid, &element1);

        let ndofs0 = space0.global_size();
        let ndofs1 = space1.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs1, ndofs0]);
        let assembler = BoundaryAssembler::<f64, _, _>::new_laplace_single_layer();
        assembler.assemble_singular_into_dense(&mut matrix, &space0, &space1);
        let csr = assembler.assemble_singular_into_csr(&space0, &space1);
        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }
}
