use bempp::assembly::boundary::BoundaryAssembler;
use bempp::function::SerialFunctionSpace;
use bempp::traits::{BoundaryAssembly, FunctionSpace};
use criterion::{criterion_group, criterion_main, Criterion};
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::{Continuity, ReferenceCellType};
use ndgrid::shapes::regular_sphere;
use rlst::rlst_dynamic_array2;

pub fn assembly_parts_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");
    group.sample_size(20);

    for i in 3..5 {
        let grid = regular_sphere(i);
        let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);

        let space = SerialFunctionSpace::new(&grid, &element);
        let mut matrix = rlst_dynamic_array2!(f64, [space.global_size(), space.global_size()]);

        let colouring = space.cell_colouring();
        let mut a = BoundaryAssembler::<f64, _, _>::new_laplace_single_layer();
        a.quadrature_degree(ReferenceCellType::Triangle, 16);
        a.singular_quadrature_degree(
            (ReferenceCellType::Triangle, ReferenceCellType::Triangle),
            4,
        );
        a.batch_size(128);

        group.bench_function(
            &format!(
                "Assembly of singular terms of {}x{} matrix",
                space.global_size(),
                space.global_size()
            ),
            |b| b.iter(|| a.assemble_singular_into_dense(&mut matrix, &space, &space)),
        );
        group.bench_function(
            &format!(
                "Assembly of non-singular terms of {}x{} matrix",
                space.global_size(),
                space.global_size()
            ),
            |b| {
                b.iter(|| {
                    a.assemble_nonsingular_into_dense(
                        &mut matrix,
                        &space,
                        &space,
                        &colouring,
                        &colouring,
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, assembly_parts_benchmark);
criterion_main!(benches);
