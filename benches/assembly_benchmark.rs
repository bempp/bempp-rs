use bempp::bem::assembly::{batched, batched::BatchedAssembler};
use bempp::bem::function_space::SerialFunctionSpace;
use bempp::element::ciarlet::LagrangeElementFamily;
use bempp::grid::shapes::regular_sphere;
use bempp::traits::bem::FunctionSpace;
use bempp::traits::element::Continuity;
use criterion::{criterion_group, criterion_main, Criterion};
use rlst::rlst_dynamic_array2;

extern crate blas_src;
extern crate lapack_src;

pub fn assembly_parts_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");
    group.sample_size(20);

    for i in 3..5 {
        let grid = regular_sphere(i);
        let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);

        let space = SerialFunctionSpace::new(&grid, &element);
        let mut matrix = rlst_dynamic_array2!(f64, [space.global_size(), space.global_size()]);

        let colouring = space.cell_colouring();
        let a = batched::LaplaceSingleLayerAssembler::<128, f64>::default();

        group.bench_function(
            &format!(
                "Assembly of singular terms of {}x{} matrix",
                space.global_size(),
                space.global_size()
            ),
            |b| b.iter(|| a.assemble_singular_into_dense(&mut matrix, 4, &space, &space)),
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
                        16,
                        16,
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