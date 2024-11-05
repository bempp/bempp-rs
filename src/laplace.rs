//! Laplace operators

/// Assemblers for Laplace problems.
pub mod assembler {
    use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
    use rlst::{rlst_dynamic_array2, DynamicArray, MatrixInverse, RlstScalar};

    use crate::{
        assembly::{
            boundary::{
                integrands::{
                    AdjointDoubleLayerBoundaryIntegrand, DoubleLayerBoundaryIntegrand,
                    HypersingularCurlCurlBoundaryIntegrand, SingleLayerBoundaryIntegrand,
                },
                BoundaryAssembler, BoundaryAssemblerOptions,
            },
            kernels::KernelEvaluator,
        },
        function::FunctionSpace,
    };

    /// Assembler for the Laplace single layer operator.
    pub fn laplace_single_layer<
        T: RlstScalar<Real = T> + MatrixInverse,
        Space: FunctionSpace<T = T> + Sync,
    >(
        trial_space: &Space,
        test_space: &Space,
        options: &BoundaryAssemblerOptions,
    ) -> DynamicArray<T, 2> {
        let nrows = trial_space.global_size();
        let ncols = test_space.global_size();

        let mut output = rlst_dynamic_array2!(T, [nrows, ncols]);

        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::Value);

        let assembler =
            BoundaryAssembler::new(SingleLayerBoundaryIntegrand::new(), kernel, options, 1, 0);

        assembler.assemble(&mut output, trial_space, test_space);

        output
    }

    /// Assembler for the Laplace double layer operator.
    pub fn laplace_double_layer<
        T: RlstScalar<Real = T> + MatrixInverse,
        Space: FunctionSpace<T = T> + Sync,
    >(
        trial_space: &Space,
        test_space: &Space,
        options: &BoundaryAssemblerOptions,
    ) -> DynamicArray<T, 2> {
        let nrows = trial_space.global_size();
        let ncols = test_space.global_size();

        let mut output = rlst_dynamic_array2!(T, [nrows, ncols]);

        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

        let assembler =
            BoundaryAssembler::new(DoubleLayerBoundaryIntegrand::new(), kernel, options, 4, 0);

        assembler.assemble(&mut output, trial_space, test_space);

        output
    }

    /// Assembler for the Laplace adjoint double layer operator.
    pub fn laplace_adjoint_double_layer<
        T: RlstScalar<Real = T> + MatrixInverse,
        Space: FunctionSpace<T = T> + Sync,
    >(
        trial_space: &Space,
        test_space: &Space,
        options: &BoundaryAssemblerOptions,
    ) -> DynamicArray<T, 2> {
        let nrows = trial_space.global_size();
        let ncols = test_space.global_size();

        let mut output = rlst_dynamic_array2!(T, [nrows, ncols]);

        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

        let assembler = BoundaryAssembler::new(
            AdjointDoubleLayerBoundaryIntegrand::new(),
            kernel,
            options,
            4,
            0,
        );

        assembler.assemble(&mut output, trial_space, test_space);

        output
    }

    /// Assembler for the Laplace hypersingular operator.
    pub fn laplace_hypersingular<
        T: RlstScalar<Real = T> + MatrixInverse,
        Space: FunctionSpace<T = T> + Sync,
    >(
        trial_space: &Space,
        test_space: &Space,
        options: &BoundaryAssemblerOptions,
    ) -> DynamicArray<T, 2> {
        let nrows = trial_space.global_size();
        let ncols = test_space.global_size();

        let mut output = rlst_dynamic_array2!(T, [nrows, ncols]);

        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

        let assembler = BoundaryAssembler::new(
            HypersingularCurlCurlBoundaryIntegrand::new(),
            kernel,
            options,
            4,
            1,
        );

        assembler.assemble(&mut output, trial_space, test_space);

        output
    }
}
