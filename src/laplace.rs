//! Laplace operators

/// Assemblers for Laplace problems.
pub mod assembler {
    use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
    use rlst::{MatrixInverse, RlstScalar};

    use crate::boundary_assemblers::{
        helpers::KernelEvaluator,
        integrands::{
            AdjointDoubleLayerBoundaryIntegrand, DoubleLayerBoundaryIntegrand,
            HypersingularCurlCurlBoundaryIntegrand, SingleLayerBoundaryIntegrand,
        },
        BoundaryAssembler, BoundaryAssemblerOptions,
    };

    /// Laplace single layer assembler type.
    pub type LaplaceSingleLayer3dAssembler<'o, T> =
        BoundaryAssembler<'o, T, SingleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>>;

    /// Laplace double layer assembler type.
    pub type LaplaceDoubleLayer3dAssembler<'o, T> =
        BoundaryAssembler<'o, T, DoubleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>>;

    /// Laplace adjoint double layer assembler type.
    pub type LaplaceAdjointDoubleLayer3dAssembler<'o, T> =
        BoundaryAssembler<'o, T, AdjointDoubleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>>;

    /// Laplace hypersingular double layer assembler type.
    pub type LaplaceHypersingular3dAssembler<'o, T> =
        BoundaryAssembler<'o, T, HypersingularCurlCurlBoundaryIntegrand<T>, Laplace3dKernel<T>>;

    /// Assembler for the Laplace single layer operator.
    pub fn laplace_single_layer<T: RlstScalar<Real = T> + MatrixInverse>(
        options: &BoundaryAssemblerOptions,
    ) -> LaplaceSingleLayer3dAssembler<T> {
        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::Value);
        BoundaryAssembler::new(SingleLayerBoundaryIntegrand::new(), kernel, options, 1, 0)
    }

    /// Assembler for the Laplace double layer operator.
    pub fn laplace_double_layer<T: RlstScalar<Real = T> + MatrixInverse>(
        options: &BoundaryAssemblerOptions,
    ) -> LaplaceDoubleLayer3dAssembler<T> {
        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

        BoundaryAssembler::new(DoubleLayerBoundaryIntegrand::new(), kernel, options, 4, 0)
    }

    /// Assembler for the Laplace adjoint double layer operator.
    pub fn laplace_adjoint_double_layer<T: RlstScalar<Real = T> + MatrixInverse>(
        options: &BoundaryAssemblerOptions,
    ) -> LaplaceAdjointDoubleLayer3dAssembler<T> {
        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

        BoundaryAssembler::new(
            AdjointDoubleLayerBoundaryIntegrand::new(),
            kernel,
            options,
            4,
            0,
        )
    }

    /// Assembler for the Laplace hypersingular operator.
    pub fn laplace_hypersingular<T: RlstScalar<Real = T> + MatrixInverse>(
        options: &BoundaryAssemblerOptions,
    ) -> LaplaceHypersingular3dAssembler<T> {
        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

        BoundaryAssembler::new(
            HypersingularCurlCurlBoundaryIntegrand::new(),
            kernel,
            options,
            4,
            1,
        )
    }
}
