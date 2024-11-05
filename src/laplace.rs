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
    pub fn laplace_single_layer<T: RlstScalar<Real = T> + MatrixInverse>(
        options: &BoundaryAssemblerOptions,
    ) -> BoundaryAssembler<T, SingleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>> {
        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::Value);
        BoundaryAssembler::new(SingleLayerBoundaryIntegrand::new(), kernel, options, 1, 0)
    }

    /// Assembler for the Laplace double layer operator.
    pub fn laplace_double_layer<T: RlstScalar<Real = T> + MatrixInverse>(
        options: &BoundaryAssemblerOptions,
    ) -> BoundaryAssembler<T, DoubleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>> {
        let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

        BoundaryAssembler::new(DoubleLayerBoundaryIntegrand::new(), kernel, options, 4, 0)
    }

    /// Assembler for the Laplace adjoint double layer operator.
    pub fn laplace_adjoint_double_layer<T: RlstScalar<Real = T> + MatrixInverse>(
        options: &BoundaryAssemblerOptions,
    ) -> BoundaryAssembler<T, AdjointDoubleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>> {
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
    ) -> BoundaryAssembler<T, HypersingularCurlCurlBoundaryIntegrand<T>, Laplace3dKernel<T>> {
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
