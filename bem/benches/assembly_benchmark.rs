use bempp_bem::assembly::{assemble_batched, batched, BoundaryOperator, PDEType};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::create_element;
use bempp_grid::shapes::regular_sphere;
use bempp_kernel::laplace_3d;
use bempp_tools::arrays::zero_matrix;
use bempp_traits::bem::DofMap;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily};
use criterion::{criterion_group, criterion_main, Criterion};

pub fn full_assembly_benchmark(c: &mut Criterion) {
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
        let mut matrix = zero_matrix([space.dofmap().global_size(), space.dofmap().global_size()]);

        group.bench_function(
            &format!(
                "Full assembly of {}x{} matrix",
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
        let mut matrix = zero_matrix([space.dofmap().global_size(), space.dofmap().global_size()]);

        let colouring = space.compute_cell_colouring();

        group.bench_function(
            &format!(
                "Assembly of singular terms of {}x{} matrix",
                space.dofmap().global_size(),
                space.dofmap().global_size()
            ),
            |b| {
                b.iter(|| {
                    batched::assemble_singular(
                        &mut matrix,
                        &laplace_3d::Laplace3dKernel::new(),
                        false,
                        false,
                        &space,
                        &space,
                        &colouring,
                        &colouring,
                        128,
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
                    batched::assemble_nonsingular::<16, 16>(
                        &mut matrix,
                        &laplace_3d::Laplace3dKernel::new(),
                        &space,
                        &space,
                        &colouring,
                        &colouring,
                        128,
                    )
                })
            },
        );
    }
    group.finish();
}

// criterion_group!(benches, full_assembly_benchmark, assembly_parts_benchmark);
criterion_group!(benches, assembly_parts_benchmark);
criterion_main!(benches);
