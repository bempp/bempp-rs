//! Interface to kifmm library

use std::cell::RefCell;

use bempp_distributed_tools::{self, permutation::DataPermutation};

use green_kernels::laplace_3d::Laplace3dKernel;
use kifmm::{
    traits::{
        fftw::{Dft, DftType},
        field::{
            SourceToTargetTranslation, SourceToTargetTranslationMetadata, SourceTranslation,
            TargetTranslation,
        },
        fmm::{DataAccessMulti, EvaluateMulti},
        general::{
            multi_node::GhostExchange,
            single_node::{AsComplex, Epsilon},
        },
    },
    tree::SortKind,
    ChargeHandler, Evaluate, FftFieldTranslation, KiFmm, KiFmmMulti, MultiNodeBuilder,
};
use mpi::traits::{Communicator, Equivalence};
use num::Float;
use rlst::{
    operator::interface::DistributedArrayVectorSpace, rlst_dynamic_array1, AsApply, Element,
    IndexLayout, MatrixSvd, OperatorBase, RawAccess, RawAccessMut, RlstScalar,
};

/// This structure instantiates an FMM evaluator.
pub struct KiFmmEvaluator<
    'a,
    C: Communicator,
    T: RlstScalar + Equivalence,
    SourceLayout: IndexLayout<Comm = C>,
    TargetLayout: IndexLayout<Comm = C>,
> where
    T::Real: Equivalence,
    T: DftType<InputType = T, OutputType = <T as AsComplex>::ComplexType>,
    T: Dft + AsComplex + Epsilon + MatrixSvd + Float,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: SourceToTargetTranslationMetadata,
{
    domain_space: &'a DistributedArrayVectorSpace<'a, SourceLayout, T>,
    range_space: &'a DistributedArrayVectorSpace<'a, TargetLayout, T>,
    source_permutation: DataPermutation<'a, SourceLayout>,
    target_permutation: DataPermutation<'a, TargetLayout>,
    n_permuted_sources: usize,
    n_permuted_targets: usize,
    fmm: RefCell<KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>>,
}

impl<
        'a,
        C: Communicator,
        T: RlstScalar + Equivalence,
        SourceLayout: IndexLayout<Comm = C>,
        TargetLayout: IndexLayout<Comm = C>,
    > std::fmt::Debug for KiFmmEvaluator<'a, C, T, SourceLayout, TargetLayout>
where
    T::Real: Equivalence,
    T: DftType<InputType = T, OutputType = <T as AsComplex>::ComplexType>,
    T: Dft + AsComplex + Epsilon + MatrixSvd + Float,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: SourceToTargetTranslationMetadata,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KiFmmEvaluator with {} sources and {} targets",
            self.domain_space.index_layout().number_of_global_indices(),
            self.range_space.index_layout().number_of_global_indices()
        )
    }
}

impl<
        'a,
        C: Communicator,
        T: RlstScalar<Real = T> + Equivalence,
        SourceLayout: IndexLayout<Comm = C>,
        TargetLayout: IndexLayout<Comm = C>,
    > KiFmmEvaluator<'a, C, T, SourceLayout, TargetLayout>
where
    T::Real: Equivalence,
    T: DftType<InputType = T, OutputType = <T as AsComplex>::ComplexType>,
    T: Dft + AsComplex + Epsilon + MatrixSvd + Float,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: SourceToTargetTranslationMetadata,
{
    /// Instantiate a new KiFmm Evaluator.
    pub fn new(
        sources: &[T::Real],
        targets: &[T::Real],
        local_tree_depth: usize,
        global_tree_depth: usize,
        expansion_order: usize,
        domain_space: &'a DistributedArrayVectorSpace<'a, SourceLayout, T>,
        range_space: &'a DistributedArrayVectorSpace<'a, TargetLayout, T>,
    ) -> Self {
        // We want that both layouts have the same communicator.
        assert!(std::ptr::addr_eq(domain_space.comm(), range_space.comm()));

        assert_eq!(
            sources.len() % 3,
            0,
            "Source vector length must be a multiple of 3."
        );
        assert_eq!(
            targets.len() % 3,
            0,
            "Target vector length must be a multiple of 3."
        );

        // The length of the source vector must be 3 times the length of the local source indices.
        assert_eq!(
            sources.len(),
            3 * domain_space.index_layout().number_of_local_indices(),
            "Number of sources ({}) does not match number of local indices ({}).",
            sources.len() / 3,
            domain_space.index_layout().number_of_local_indices(),
        );

        // The length of the target vector must be 3 times the length of the local target indices.
        assert_eq!(
            targets.len(),
            3 * range_space.index_layout().number_of_local_indices(),
            "Number of targets ({}) does not match number of local indices ({}).",
            targets.len() / 3,
            range_space.index_layout().number_of_local_indices(),
        );

        let simple_comm = domain_space.index_layout().comm().duplicate();
        let cell = RefCell::new(
            MultiNodeBuilder::new(false)
                .tree(
                    &simple_comm,
                    sources,
                    targets,
                    local_tree_depth as u64,
                    global_tree_depth as u64,
                    true,
                    SortKind::Samplesort { n_samples: 100 },
                )
                .unwrap()
                .parameters(
                    expansion_order,
                    Laplace3dKernel::<T::Real>::default(),
                    FftFieldTranslation::<T>::new(None),
                )
                .unwrap()
                .build()
                .unwrap(),
        );

        let source_indices = cell.borrow().tree.source_tree.global_indices.clone();

        let target_indices = cell.borrow().tree.target_tree.global_indices.clone();

        let source_permutation = DataPermutation::new(domain_space.index_layout(), &source_indices);
        let target_permutation = DataPermutation::new(range_space.index_layout(), &target_indices);

        KiFmmEvaluator {
            domain_space,
            range_space,
            source_permutation,
            target_permutation,
            n_permuted_sources: source_indices.len(),
            n_permuted_targets: target_indices.len(),
            fmm: cell,
        }
    }
}

impl<
        'a,
        C: Communicator,
        T: RlstScalar<Real = T> + Equivalence,
        SourceLayout: IndexLayout<Comm = C>,
        TargetLayout: IndexLayout<Comm = C>,
    > OperatorBase for KiFmmEvaluator<'a, C, T, SourceLayout, TargetLayout>
where
    T::Real: Equivalence,
    T: DftType<InputType = T, OutputType = <T as AsComplex>::ComplexType>,
    T: Dft + AsComplex + Epsilon + MatrixSvd + Float,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: SourceToTargetTranslationMetadata,
{
    type Domain = DistributedArrayVectorSpace<'a, SourceLayout, T>;

    type Range = DistributedArrayVectorSpace<'a, TargetLayout, T>;

    fn domain(&self) -> &Self::Domain {
        self.domain_space
    }

    fn range(&self) -> &Self::Range {
        self.range_space
    }
}

impl<
        'a,
        C: Communicator,
        T: RlstScalar<Real = T> + Equivalence,
        SourceLayout: IndexLayout<Comm = C>,
        TargetLayout: IndexLayout<Comm = C>,
    > AsApply for KiFmmEvaluator<'a, C, T, SourceLayout, TargetLayout>
where
    T::Real: Equivalence,
    T: DftType<InputType = T, OutputType = <T as AsComplex>::ComplexType>,
    T: Dft + AsComplex + Epsilon + MatrixSvd + Float,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: SourceToTargetTranslationMetadata,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: SourceToTargetTranslation,
    KiFmm<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: Evaluate,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: SourceTranslation,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: TargetTranslation,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: DataAccessMulti<Scalar = T>,
    KiFmmMulti<T, Laplace3dKernel<T>, FftFieldTranslation<T>>: GhostExchange,
{
    fn apply_extended(
        &self,
        alpha: <Self::Range as rlst::LinearSpace>::F,
        x: &<Self::Domain as rlst::LinearSpace>::E,
        beta: <Self::Range as rlst::LinearSpace>::F,
        y: &mut <Self::Range as rlst::LinearSpace>::E,
    ) -> rlst::RlstResult<()> {
        let mut x_permuted = rlst_dynamic_array1![T, [self.n_permuted_sources]];
        let mut y_permuted = rlst_dynamic_array1![T, [self.n_permuted_targets]];

        // This is the result vector of the FMM when permuted back into proper ordering.
        let mut y_result = rlst_dynamic_array1![
            T,
            [self.range_space.index_layout().number_of_local_indices()]
        ];

        // Now forward permute the source vector.

        self.source_permutation
            .forward_permute(x.view().local().data(), x_permuted.data_mut());

        // Multiply with the vector alpha

        x_permuted.scale_inplace(alpha);

        // We can now attach the charges to the kifmm.

        self.fmm
            .borrow_mut()
            .attach_charges_ordered(x_permuted.data())
            .unwrap();

        // Now we can evaluate the FMM.

        self.fmm.borrow_mut().evaluate().unwrap();

        // Save it into y_permuted.
        y_permuted
            .data_mut()
            .copy_from_slice(self.fmm.borrow().potentials().unwrap().as_slice());

        // Now permute back the result.
        self.target_permutation
            .backward_permute(y_permuted.data(), y_result.data_mut());

        // Scale the vector y with beta before we add the result.
        y.scale_inplace(beta);
        // Now add the result.
        y.view_mut().local_mut().sum_into(y_result.r());

        Ok(())
    }
}
