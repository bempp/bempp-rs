//! Helmholtz operators

/// Assemblers for Helmholtz problems
pub mod assembler {

    /// Helmholtz single layer assembler type.
    pub type HelmholtzSingleLayer3dAssembler<'o, T> =
        BoundaryAssembler<'o, T, SingleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>>;

    /// Helmholtz double layer assembler type.
    pub type HelmholtzDoubleLayer3dAssembler<'o, T> =
        BoundaryAssembler<'o, T, DoubleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>>;

    /// Helmholtz adjoint double layer assembler type.
    pub type HelmholtzAdjointDoubleLayer3dAssembler<'o, T> =
        BoundaryAssembler<'o, T, AdjointDoubleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>>;

    /// Helmholtz hypersingular double layer assembler type.
    pub type HelmholtzHypersingular3dAssembler<'o, T> = BoundaryAssembler<
        'o,
        T,
        BoundaryIntegrandSum<
            T,
            HypersingularCurlCurlBoundaryIntegrand<T>,
            BoundaryIntegrandTimesScalar<T, HypersingularNormalNormalBoundaryIntegrand<T>>,
        >,
        Helmholtz3dKernel<T>,
    >;

    use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::GreenKernelEvalType};
    use rlst::{MatrixInverse, RlstScalar};

    use crate::boundary_assemblers::{
        helpers::KernelEvaluator,
        integrands::{
            AdjointDoubleLayerBoundaryIntegrand, BoundaryIntegrandSum,
            BoundaryIntegrandTimesScalar, DoubleLayerBoundaryIntegrand,
            HypersingularCurlCurlBoundaryIntegrand, HypersingularNormalNormalBoundaryIntegrand,
            SingleLayerBoundaryIntegrand,
        },
        BoundaryAssembler, BoundaryAssemblerOptions,
    };

    /// Assembler for the Helmholtz single layer operator.
    pub fn helmholtz_single_layer<T: RlstScalar<Complex = T> + MatrixInverse>(
        wavenumber: T::Real,
        options: &BoundaryAssemblerOptions,
    ) -> HelmholtzSingleLayer3dAssembler<T> {
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
    ) -> HelmholtzDoubleLayer3dAssembler<T> {
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
    ) -> HelmholtzAdjointDoubleLayer3dAssembler<T> {
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
    ) -> HelmholtzHypersingular3dAssembler<T> {
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
