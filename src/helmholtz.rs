//! Helmholtz operators

/// Assemblers for Helmholtz problems
pub mod assembler {
    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::GreenKernelEvalType};
    use rlst::{rlst_dynamic_array2, DynamicArray, MatrixInverse, RlstScalar};

    use crate::{
        assembly::{
            boundary::{
                integrands::{
                    AdjointDoubleLayerBoundaryIntegrand, BoundaryIntegrandSum,
                    BoundaryIntegrandTimesScalar, DoubleLayerBoundaryIntegrand,
                    HypersingularCurlCurlBoundaryIntegrand,
                    HypersingularNormalNormalBoundaryIntegrand, SingleLayerBoundaryIntegrand,
                },
                BoundaryAssembler, BoundaryAssemblerOptions,
            },
            kernels::KernelEvaluator,
        },
        function::FunctionSpace,
    };

    /// Assembler for the Helmholtz single layer operator.
    pub fn helmholtz_single_layer<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Space: FunctionSpace<T = T> + Sync,
    >(
        wavenumber: T::Real,
        trial_space: &Space,
        test_space: &Space,
        options: &BoundaryAssemblerOptions,
    ) -> DynamicArray<T, 2> {
        let nrows = trial_space.global_size();
        let ncols = test_space.global_size();

        let mut output = rlst_dynamic_array2!(T, [nrows, ncols]);

        let kernel = KernelEvaluator::new(
            Helmholtz3dKernel::new(wavenumber),
            GreenKernelEvalType::Value,
        );

        let assembler =
            BoundaryAssembler::new(SingleLayerBoundaryIntegrand::new(), kernel, options, 1, 0);

        assembler.assemble_into_dense(&mut output, trial_space, test_space);

        output
    }

    /// Assembler for the Helmholtz double layer operator.
    pub fn helmholtz_double_layer<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Space: FunctionSpace<T = T> + Sync,
    >(
        wavenumber: T::Real,
        trial_space: &Space,
        test_space: &Space,
        options: &BoundaryAssemblerOptions,
    ) -> DynamicArray<T, 2> {
        let nrows = trial_space.global_size();
        let ncols = test_space.global_size();

        let mut output = rlst_dynamic_array2!(T, [nrows, ncols]);

        let kernel = KernelEvaluator::new(
            Helmholtz3dKernel::new(wavenumber),
            GreenKernelEvalType::ValueDeriv,
        );

        let assembler =
            BoundaryAssembler::new(DoubleLayerBoundaryIntegrand::new(), kernel, options, 4, 0);

        assembler.assemble_into_dense(&mut output, trial_space, test_space);

        output
    }

    /// Assembler for the Helmholtz adjoint double layer operator.
    pub fn helmholtz_adjoint_double_layer<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Space: FunctionSpace<T = T> + Sync,
    >(
        wavenumber: T::Real,
        trial_space: &Space,
        test_space: &Space,
        options: &BoundaryAssemblerOptions,
    ) -> DynamicArray<T, 2> {
        let nrows = trial_space.global_size();
        let ncols = test_space.global_size();

        let mut output = rlst_dynamic_array2!(T, [nrows, ncols]);

        let kernel = KernelEvaluator::new(
            Helmholtz3dKernel::new(wavenumber),
            GreenKernelEvalType::ValueDeriv,
        );

        let assembler = BoundaryAssembler::new(
            AdjointDoubleLayerBoundaryIntegrand::new(),
            kernel,
            options,
            4,
            0,
        );

        assembler.assemble_into_dense(&mut output, trial_space, test_space);

        output
    }

    /// Assembler for the Helmholtz hypersingular operator.
    pub fn helmholtz_hypersingular<
        T: RlstScalar<Complex = T> + MatrixInverse,
        Space: FunctionSpace<T = T> + Sync,
    >(
        wavenumber: T::Real,
        trial_space: &Space,
        test_space: &Space,
        options: &BoundaryAssemblerOptions,
    ) -> DynamicArray<T, 2> {
        let nrows = trial_space.global_size();
        let ncols = test_space.global_size();

        let mut output = rlst_dynamic_array2!(T, [nrows, ncols]);

        let kernel = KernelEvaluator::new(
            Helmholtz3dKernel::new(wavenumber),
            GreenKernelEvalType::ValueDeriv,
        );

        let integrand = BoundaryIntegrandSum::new(
            HypersingularCurlCurlBoundaryIntegrand::new(),
            BoundaryIntegrandTimesScalar::new(
                num::cast::<T::Real, T>(-wavenumber.powi(2)).unwrap(),
                HypersingularNormalNormalBoundaryIntegrand::new(),
            ),
        );

        let assembler = BoundaryAssembler::new(integrand, kernel, options, 4, 1);

        assembler.assemble_into_dense(&mut output, trial_space, test_space);

        output
    }
}
