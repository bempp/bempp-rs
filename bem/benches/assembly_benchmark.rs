use bempp_bem::assembly::{assemble_batched, BoundaryOperator, PDEType};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::create_element;
use bempp_grid::shapes::regular_sphere;
use bempp_tools::arrays::Array2D;
use bempp_traits::bem::DofMap;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily};
use criterion::{criterion_group, criterion_main, Criterion};

pub fn assembly_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");
    group.sample_size(20);

    for i in 3..5 {
        let grid = regular_sphere(i);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );

        let space = SerialFunctionSpace::new(&grid, &element);
        let mut matrix =
            Array2D::<f64>::new((space.dofmap().global_size(), space.dofmap().global_size()));

        group.bench_function(
            &format!(
                "Assembly of {}x{} matrix",
                space.dofmap().global_size(),
                space.dofmap().global_size()
            ),
            |b| {
                b.iter(|| {
                    assemble_batched(
                        &mut matrix,
                        BoundaryOperator::SingleLayer,
                        PDEType::Laplace,
                        &space,
                        &space,
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, assembly_benchmark);
criterion_main!(benches);
