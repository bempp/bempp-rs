// Compare dense and FMM Green's function evaluators.

use bempp::greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator;
use bempp_distributed_tools::IndexLayoutFromLocalCounts;
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use mpi::traits::{Communicator, Root};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rlst::{
    operator::interface::DistributedArrayVectorSpace, rlst_dynamic_array1, AsApply, Element,
    LinearSpace, NormedSpace, RawAccessMut,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    let rank = world.rank() as usize;

    // Number of points per process.
    let npoints = 10000;

    // Seed the random number generator with a different seed for each process.
    let mut rng = ChaCha8Rng::seed_from_u64(world.rank() as u64);

    // Create random sources and targets.

    let sources = (0..3 * npoints)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect::<Vec<f64>>();

    let targets = sources.clone();

    // Initialise MPI

    // Initalise the index layout.

    let index_layout = IndexLayoutFromLocalCounts::new(npoints, &world);

    // Create the vector space.

    let space = DistributedArrayVectorSpace::<_, f64>::new(&index_layout);

    // Create a random vector of charges.

    let mut charges = space.zero();

    // charges
    //     .view_mut()
    //     .local_mut()
    //     .fill_from_equally_distributed(&mut rng);

    charges.view_mut().local_mut().set_one();

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

    let rel_diff = space.norm(
        &space
            .new_from(&output_dense)
            .sum(&space.new_from(&output_fmm).neg()),
    ) / dense_norm;

    if world.rank() == 0 {
        println!("The relative error is: {}", rel_diff);
    }

    // We now gather back the data to the root process and repeat the calculation on just a single node.

    if rank != 0 {
        world.process_at_rank(0).gather_into(&sources);
        world.process_at_rank(0).gather_into(&targets);
        charges.view().gather_to_rank(0);
        output_fmm.view().gather_to_rank(0);
        output_dense.view().gather_to_rank(0);
    } else {
        let mut gathered_sources = vec![0.0; 3 * npoints * size];
        let mut gathered_targets = vec![0.0; 3 * npoints * size];
        let mut gathered_charges = rlst_dynamic_array1!(f64, [npoints * size]);
        let mut gathered_output_fmm = rlst_dynamic_array1!(f64, [npoints * size]);
        let mut gathered_output_dense = rlst_dynamic_array1!(f64, [npoints * size]);

        world
            .this_process()
            .gather_into_root(&sources, &mut gathered_sources);

        world
            .this_process()
            .gather_into_root(&targets, &mut gathered_targets);

        charges.view().gather_to_rank_root(gathered_charges.r_mut());
        output_fmm
            .view()
            .gather_to_rank_root(gathered_output_fmm.r_mut());
        output_dense
            .view()
            .gather_to_rank_root(gathered_output_dense.r_mut());

        // Now we have everything on root. Let's create a self communicator just on root.

        let root_comm = mpi::topology::SimpleCommunicator::self_comm();

        let index_layout_root = IndexLayoutFromLocalCounts::new(npoints * size, &root_comm);
        let space_root = DistributedArrayVectorSpace::<_, f64>::new(&index_layout_root);
        let evaluator_dense_on_root =
            bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::new(
                &gathered_sources,
                &gathered_targets,
                GreenKernelEvalType::Value,
                false,
                Laplace3dKernel::default(),
                &space_root,
                &space_root,
            );
        let fmm_evaluator_on_root = KiFmmEvaluator::new(
            &gathered_sources,
            &gathered_targets,
            3,
            1,
            5,
            &space_root,
            &space_root,
        );

        // Create the charge vector on root.

        let mut charges_on_root = space_root.zero();

        charges_on_root
            .view_mut()
            .local_mut()
            .data_mut()
            .copy_from_slice(gathered_charges.data_mut());

        // Now apply the fmm evaluator
        let fmm_result_on_root = fmm_evaluator_on_root.apply(&charges_on_root);
        // Now apply the dense evaluator
        let dense_result_on_root = evaluator_dense_on_root.apply(&charges_on_root);

        // Now compare the dense result on root with the global dense result.

        let dense_rel_diff = (gathered_output_dense.r() - dense_result_on_root.view().local().r())
            .norm_2()
            / gathered_output_dense.r().norm_2();

        println!(
            "Dense difference between MPI and single node: {}",
            dense_rel_diff
        );

        let fmm_rel_diff = (gathered_output_fmm.r() - fmm_result_on_root.view().local().r())
            .norm_2()
            / gathered_output_fmm.r().norm_2();

        println!(
            "FMM difference between MPI and single node: {}",
            fmm_rel_diff
        );
    }
}
