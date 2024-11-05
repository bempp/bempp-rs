use bempp::assembly::boundary::integrands::SingleLayerBoundaryIntegrand;
use bempp::assembly::boundary::{BoundaryAssembler, BoundaryAssemblerOptions};
use bempp::assembly::kernels::KernelEvaluator;
use bempp::function::FunctionSpace;
use bempp::function::SerialFunctionSpace;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::types::GreenKernelEvalType;
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::{Continuity, ReferenceCellType};
use ndgrid::shapes::regular_sphere;

pub fn assembly_parts_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");
    group.sample_size(20);

    for i in 3..5 {
        let grid = regular_sphere(i);
        let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);

        let space = SerialFunctionSpace::new(&grid, &element);
        let mut options = BoundaryAssemblerOptions::default();
        options.set_regular_quadrature_degree(ReferenceCellType::Triangle, 16);
        options.set_singular_quadrature_degree(
            (ReferenceCellType::Triangle, ReferenceCellType::Triangle),
            4,
        );
        options.set_batch_size(128);

        let assembler = BoundaryAssembler::<f64, _, _>::new(
            SingleLayerBoundaryIntegrand::new(),
            KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::Value),
            &options,
            1,
            0,
        );

        group.bench_function(
            format!(
                "Assembly of singular terms of {}x{} matrix",
                space.global_size(),
                space.global_size()
            ),
            |b| b.iter(|| black_box(assembler.assemble_singular_into_csr(&space, &space))),
        );
    }
    group.finish();
}

criterion_group!(benches, assembly_parts_benchmark);
criterion_main!(benches);
