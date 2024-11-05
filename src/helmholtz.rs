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
    pub fn helmholtz_single_layer<T: RlstScalar<Complex = T> + MatrixInverse>(
        wavenumber: T::Real,
        options: &BoundaryAssemblerOptions,
    ) -> BoundaryAssembler<T, SingleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>> {
        let kernel = KernelEvaluator::new(
            Helmholtz3dKernel::new(wavenumber),
            GreenKernelEvalType::Value,
        );

        BoundaryAssembler::new(SingleLayerBoundaryIntegrand::new(), kernel, options, 1, 0)
    }

    /// Assembler for the Helmholtz double layer operator.
    pub fn helmholtz_double_layer<T: RlstScalar<Complex = T> + MatrixInverse>(
        wavenumber: T::Real,
        options: &BoundaryAssemblerOptions,
    ) -> BoundaryAssembler<T, DoubleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>> {
        let kernel = KernelEvaluator::new(
            Helmholtz3dKernel::new(wavenumber),
            GreenKernelEvalType::ValueDeriv,
        );

        BoundaryAssembler::new(DoubleLayerBoundaryIntegrand::new(), kernel, options, 4, 0)
    }

    /// Assembler for the Helmholtz adjoint double layer operator.
    pub fn helmholtz_adjoint_double_layer<T: RlstScalar<Complex = T> + MatrixInverse>(
        wavenumber: T::Real,
        options: &BoundaryAssemblerOptions,
    ) -> BoundaryAssembler<T, AdjointDoubleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>> {
        let kernel = KernelEvaluator::new(
            Helmholtz3dKernel::new(wavenumber),
            GreenKernelEvalType::ValueDeriv,
        );

        BoundaryAssembler::new(
            AdjointDoubleLayerBoundaryIntegrand::new(),
            kernel,
            options,
            4,
            0,
        )
    }

    /// Assembler for the Helmholtz hypersingular operator.
    pub fn helmholtz_hypersingular<T: RlstScalar<Complex = T> + MatrixInverse>(
        wavenumber: T::Real,
        options: &BoundaryAssemblerOptions,
    ) -> BoundaryAssembler<
        T,
        BoundaryIntegrandSum<
            T,
            HypersingularCurlCurlBoundaryIntegrand<T>,
            BoundaryIntegrandTimesScalar<T, HypersingularNormalNormalBoundaryIntegrand<T>>,
        >,
        Helmholtz3dKernel<T>,
    > {
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

        BoundaryAssembler::new(integrand, kernel, options, 4, 1)
    }
}
