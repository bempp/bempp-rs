// Compare dense and FMM Green's function evaluators.

use bempp::greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator;
use bempp_distributed_tools::IndexLayoutFromLocalCounts;
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use mpi::traits::Communicator;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rlst::{
    operator::interface::DistributedArrayVectorSpace, AsApply, Element, LinearSpace, NormedSpace,
};

fn main() {
    // Number of points per process.
    let npoints = 10000;
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    // Create random sources and targets.

    let sources = (0..3 * npoints)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect::<Vec<f64>>();

    let targets = sources.clone();

    // Initialise MPI

    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    // Initalise the index layout.

    let index_layout = IndexLayoutFromLocalCounts::new(npoints, &world);

    // Create the vector space.

    let space = DistributedArrayVectorSpace::<_, f64>::new(&index_layout);

    // Create a random vector of charges.

    let mut charges = space.zero();

    charges
        .view_mut()
        .local_mut()
        .fill_from_equally_distributed(&mut rng);

    // Create the dense evaluator.

    let dense_evaluator = bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::new(
        &sources,
        &targets,
        GreenKernelEvalType::Value,
        false,
        Laplace3dKernel::default(),
        &space,
        &space,
    );

    // Create the FMM evaluator.

    let fmm_evaluator = KiFmmEvaluator::new(&sources, &targets, 3, 1, 5, &space, &space);

    // Apply the dense evaluator.

    let output_dense = dense_evaluator.apply(&charges);

    // Apply the FMM evaluator

    let output_fmm = fmm_evaluator.apply(&charges);

    // Compare the results.

    let dense_norm = space.norm(&output_dense);

    let rel_diff = space.norm(&output_dense.sum(&output_fmm.neg())) / dense_norm;

    if world.rank() == 0 {
        println!("The relative error is: {}", rel_diff);
    }
}
