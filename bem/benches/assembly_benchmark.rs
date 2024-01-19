use bempp_bem::assembly::batched;
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::create_element;
use bempp_grid::shapes::regular_sphere;
use bempp_kernel::laplace_3d;
use bempp_traits::bem::DofMap;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily};
use criterion::{criterion_group, criterion_main, Criterion};
use rlst_dense::rlst_dynamic_array2;

pub fn assembly_parts_benchmark(c: &mut Criterion) {
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
        let mut matrix = rlst_dynamic_array2!(
            f64,
            [space.dofmap().global_size(), space.dofmap().global_size()]
        );

        let colouring = space.compute_cell_colouring();

        group.bench_function(
            &format!(
                "Assembly of singular terms of {}x{} matrix",
                space.dofmap().global_size(),
                space.dofmap().global_size()
            ),
            |b| {
                b.iter(|| {
                    batched::assemble_singular_into_dense::<4, 128>(
                        &mut matrix,
                        &laplace_3d::Laplace3dKernel::new(),
                        &space,
                        &space,
                        &colouring,
                        &colouring,
                    )
                })
            },
        );
        group.bench_function(
            &format!(
                "Assembly of non-singular terms of {}x{} matrix",
                space.dofmap().global_size(),
                space.dofmap().global_size()
            ),
            |b| {
                b.iter(|| {
                    batched::assemble_nonsingular::<16, 16, 128>(
                        &mut matrix,
                        &laplace_3d::Laplace3dKernel::new(),
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
