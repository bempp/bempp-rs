//! Assemble a nonsingular dense operator.

pub use green_kernels::traits::Kernel;
use rlst::prelude::*;

/// Assembler for a single test/trial element pair.
pub trait RegularElementAssembler {
    /// Scalar data type.
    type T: RlstScalar;
    /// Structure that holds test element data.
    type TestElementData: Sized;
    /// Structure that holds trial element data.
    type TrialElementData: Sized;

    /// Assemble a single element pair with the given data.
    fn assemble<'a>(
        &self,
        test_element_data: &'a Self::TestElementData,
        trial_element_data: &'a Self::TrialElementData,
        result: &'a mut [Self::T],
    );
}

pub struct DefaultScalarElementData<T: RlstScalar> {
    /// Number of basis functions on element.
    pub number_of_basis_functions: usize,
    /// Number of physical integration points.
    ///
    /// The ordering of the points is of the form [x1, y1, z1, x2, y2, z2]...
    pub number_of_integration_points: usize,
    /// Values of element basis functions
    pub function_values: Vec<T::Real>,
    /// The integration element for each point.
    pub integration_elements: Vec<T::Real>,
    /// The integration points on the physical element.
    pub integration_points: Vec<T::Real>,
    /// The integration weight for each point.
    pub integration_weights: Vec<T::Real>,
}

/// Default element assembler for scalar single-layer kernels.
pub struct DefaultScalarSingleLayerRegularElementAssembler<'a, T: RlstScalar, K: Kernel<T = T>> {
    kernel: &'a K,
}

impl<'a, T: RlstScalar, K: Kernel<T = T>> RegularElementAssembler
    for DefaultScalarSingleLayerRegularElementAssembler<'a, T, K>
{
    type T = T;

    type TestElementData = DefaultScalarElementData<T>;

    type TrialElementData = DefaultScalarElementData<T>;

    fn assemble<'b>(
        &self,
        test_element_data: &'b Self::TestElementData,
        trial_element_data: &'b Self::TrialElementData,
        result: &'b mut [Self::T],
    ) {
        let number_of_test_points = test_element_data.number_of_integration_points;
        let number_of_trial_points = trial_element_data.number_of_integration_points;

        let number_of_test_functions = test_element_data.number_of_basis_functions;
        let number_of_trial_functions = trial_element_data.number_of_basis_functions;

        let mut kernel_matrix =
            rlst_dynamic_array2!(T, [number_of_trial_points, number_of_test_points]);

        let mut result = rlst_array_from_slice_mut2!(
            result,
            [
                test_element_data.number_of_basis_functions,
                trial_element_data.number_of_basis_functions
            ]
        );

        let test_functions = rlst_array_from_slice2!(
            &test_element_data.function_values,
            [number_of_test_points, number_of_test_functions]
        );
        let trial_functions = rlst_array_from_slice2!(
            &trial_element_data.function_values,
            [number_of_trial_points, number_of_trial_functions]
        );

        let test_weights = rlst_array_from_slice1!(
            &test_element_data.integration_weights,
            [number_of_test_points]
        );

        let trial_weights = rlst_array_from_slice1!(
            &trial_element_data.integration_weights,
            [number_of_trial_points]
        );

        // Evaluate the kernel matrix

        self.kernel.assemble_st(
            green_kernels::types::EvalType::Value,
            &trial_element_data.integration_points,
            &test_element_data.integration_points,
            kernel_matrix.data_mut(),
        );

        // Multiply the kernel matrix with integration weights

        kernel_matrix.rank1_cmp_product_inplace(trial_weights.cast(), test_weights.cast());

        // For each test/trial function pair multiply as rank 1 update the corresponding values into the kernel matrix and sum up.

        for (trial_function_values, mut res_col) in
            itertools::izip!(trial_functions.col_iter(), result.col_iter_mut())
        {
            for (test_function_values, res_value) in
                itertools::izip!(test_functions.col_iter(), res_col.iter_mut())
            {
                *res_value = kernel_matrix
                    .view()
                    .rank1_cmp_product(
                        trial_function_values.view().cast(),
                        test_function_values.view().cast(),
                    )
                    .sum();
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::grid::flat_triangle_grid::FlatTriangleGridBuilder;
    use crate::traits::grid::{CellType, GeometryType, Grid, PointType};

    #[test]
    fn test_laplace_single_element() {
        let mut grid_builder = FlatTriangleGridBuilder::<f64>::new();

        grid_builder.add_point(1, [0.0, 0.0, 0.0]);
        grid_builder.add_point(2, [1.0, 0.0, 0.0]);
        grid_builder.add_point(3, [1.0, 1.0, 0.0]);

        grid_builder.add_point(4, [0.0, 0.0, 1.0]);
        grid_builder.add_point(5, [1.0, 0.0, 1.0]);
        grid_builder.add_point(6, [1.0, 1.0, 1.0]);

        grid_builder.add_cell(1, [1, 2, 3]);
        grid_builder.add_cell(2, [4, 5, 6]);

        let grid = grid_builder.create_grid();
    }
}
