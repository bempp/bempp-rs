//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
use approx::assert_relative_eq;
#[cfg(feature = "mpi")]
use bempp::{
    assembly::batched,
    assembly::batched::BatchedAssembler,
    element::ciarlet::LagrangeElementFamily,
    function::{ParallelFunctionSpace, SerialFunctionSpace},
    grid::{
        flat_triangle_grid::{FlatTriangleGrid, FlatTriangleGridBuilder},
        mixed_grid::{MixedGrid, MixedGridBuilder},
        parallel_grid::ParallelGrid,
        single_element_grid::{SingleElementGrid, SingleElementGridBuilder},
    },
    traits::{
        element::Continuity,
        function::FunctionSpace,
        grid::{Builder, CellType, GeometryType, GridType, ParallelBuilder, PointType},
        types::{Ownership, ReferenceCellType},
    },
};
#[cfg(feature = "mpi")]
use mpi::{
    environment::Universe,
    request::WaitGuard,
    traits::{Communicator, Destination, Source},
};
#[cfg(feature = "mpi")]
use rlst::CsrMatrix;
#[cfg(feature = "mpi")]
use std::collections::HashMap;

extern crate blas_src;
extern crate lapack_src;

#[cfg(feature = "mpi")]
fn create_flat_triangle_grid_data(b: &mut FlatTriangleGridBuilder<f64>, n: usize) {
    for y in 0..n {
        for x in 0..n {
            b.add_point(
                y * n + x,
                [x as f64 / (n - 1) as f64, y as f64 / (n - 1) as f64, 0.0],
            );
        }
    }

    for i in 0..n - 1 {
        for j in 0..n - 1 {
            b.add_cell(
                2 * i * (n - 1) + j,
                [j * n + i, j * n + i + 1, j * n + i + n + 1],
            );
            b.add_cell(
                2 * i * (n - 1) + j + 1,
                [j * n + i, j * n + i + n + 1, j * n + i + n],
            );
        }
    }
}

#[cfg(feature = "mpi")]
fn example_flat_triangle_grid<C: Communicator>(
    comm: &C,
    n: usize,
) -> ParallelGrid<'_, C, FlatTriangleGrid<f64>> {
    let rank = comm.rank();
    let size = comm.size();

    let mut b = FlatTriangleGridBuilder::<f64>::new(());

    if rank == 0 {
        create_flat_triangle_grid_data(&mut b, n);

        let ncells = 2 * (n - 1).pow(2);

        let mut owners = HashMap::new();
        let mut c = 0;
        for r in 0..size {
            let end = if r + 1 == size {
                ncells
            } else {
                (r + 1) as usize * ncells / size as usize
            };
            while c < end {
                owners.insert(c, r as usize);
                c += 1;
            }
        }
        b.create_parallel_grid(comm, &owners)
    } else {
        b.receive_parallel_grid(comm, 0)
    }
}

#[cfg(feature = "mpi")]
fn example_flat_triangle_grid_serial(n: usize) -> FlatTriangleGrid<f64> {
    let mut b = FlatTriangleGridBuilder::<f64>::new(());
    create_flat_triangle_grid_data(&mut b, n);
    b.create_grid()
}

#[cfg(feature = "mpi")]
fn create_single_element_grid_data(b: &mut SingleElementGridBuilder<3, f64>, n: usize) {
    for y in 0..n {
        for x in 0..n {
            b.add_point(
                y * n + x,
                [x as f64 / (n - 1) as f64, y as f64 / (n - 1) as f64, 0.0],
            );
        }
    }

    for i in 0..n - 1 {
        for j in 0..n - 1 {
            b.add_cell(
                i * (n - 1) + j,
                vec![j * n + i, j * n + i + 1, j * n + i + n, j * n + i + n + 1],
            );
        }
    }
}

#[cfg(feature = "mpi")]
fn example_single_element_grid<C: Communicator>(
    comm: &C,
    n: usize,
) -> ParallelGrid<'_, C, SingleElementGrid<f64>> {
    let rank = comm.rank();
    let size = comm.size();

    let mut b = SingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Quadrilateral, 1));

    if rank == 0 {
        create_single_element_grid_data(&mut b, n);

        let ncells = (n - 1).pow(2);

        let mut owners = HashMap::new();
        let mut c = 0;
        for r in 0..size {
            let end = if r + 1 == size {
                ncells
            } else {
                (r + 1) as usize * ncells / size as usize
            };
            while c < end {
                owners.insert(c, r as usize);
                c += 1;
            }
        }
        b.create_parallel_grid(comm, &owners)
    } else {
        b.receive_parallel_grid(comm, 0)
    }
}

#[cfg(feature = "mpi")]
fn example_single_element_grid_serial(n: usize) -> SingleElementGrid<f64> {
    let mut b = SingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Quadrilateral, 1));
    create_single_element_grid_data(&mut b, n);
    b.create_grid()
}

#[cfg(feature = "mpi")]
fn create_mixed_grid_data(b: &mut MixedGridBuilder<3, f64>, n: usize) {
    for y in 0..n {
        for x in 0..n {
            b.add_point(
                y * n + x,
                [x as f64 / (n - 1) as f64, y as f64 / (n - 1) as f64, 0.0],
            );
        }
    }

    for i in 0..n - 1 {
        for j in 0..n - 1 {
            b.add_cell(
                i * (n - 1) + j,
                (
                    vec![j * n + i, j * n + i + 1, j * n + i + n, j * n + i + n + 1],
                    ReferenceCellType::Quadrilateral,
                    1,
                ),
            );
        }
    }
}

#[cfg(feature = "mpi")]
fn example_mixed_grid<C: Communicator>(comm: &C, n: usize) -> ParallelGrid<'_, C, MixedGrid<f64>> {
    let rank = comm.rank();
    let size = comm.size();

    let mut b = MixedGridBuilder::<3, f64>::new(());

    if rank == 0 {
        create_mixed_grid_data(&mut b, n);

        let ncells = (n - 1).pow(2);

        let mut owners = HashMap::new();
        let mut c = 0;
        for r in 0..size {
            let end = if r + 1 == size {
                ncells
            } else {
                (r + 1) as usize * ncells / size as usize
            };
            while c < end {
                owners.insert(c, r as usize);
                c += 1;
            }
        }
        b.create_parallel_grid(comm, &owners)
    } else {
        b.receive_parallel_grid(comm, 0)
    }
}

#[cfg(feature = "mpi")]
fn example_mixed_grid_serial(n: usize) -> MixedGrid<f64> {
    let mut b = MixedGridBuilder::<3, f64>::new(());
    create_mixed_grid_data(&mut b, n);
    b.create_grid()
}

#[cfg(feature = "mpi")]
fn test_parallel_flat_triangle_grid<C: Communicator>(comm: &C) {
    let rank = comm.rank();
    let size = comm.size();

    let n = 10;
    let grid = example_flat_triangle_grid(comm, n);

    let mut area = 0.0;
    for cell in grid.iter_all_cells() {
        if cell.ownership() == Ownership::Owned {
            area += cell.geometry().volume();
        }
    }
    if rank != 0 {
        mpi::request::scope(|scope| {
            let _sreq2 = WaitGuard::from(comm.process_at_rank(0).immediate_send(scope, &area));
        });
    } else {
        for p in 1..size {
            let (a, _status) = comm.process_at_rank(p).receive::<f64>();
            area += a;
        }
        assert_relative_eq!(area, 1.0, max_relative = 1e-10);
    }

    let mut nvertices = 0;
    for v in 0..grid.number_of_vertices() {
        if grid.vertex_from_index(v).ownership() == Ownership::Owned {
            nvertices += 1
        }
    }
    if rank != 0 {
        mpi::request::scope(|scope| {
            let _sreq2 = WaitGuard::from(comm.process_at_rank(0).immediate_send(scope, &nvertices));
        });
    } else {
        for p in 1..size {
            let (nv, _status) = comm.process_at_rank(p).receive::<usize>();
            nvertices += nv;
        }
        assert_eq!(nvertices, n * n);
    }
}

#[cfg(feature = "mpi")]
fn test_parallel_assembly_flat_triangle_grid<C: Communicator>(
    comm: &C,
    degree: usize,
    cont: Continuity,
) {
    let rank = comm.rank();
    let size = comm.size();

    let n = 10;
    let grid = example_flat_triangle_grid(comm, n);
    let element = LagrangeElementFamily::<f64>::new(degree, cont);
    let space = ParallelFunctionSpace::new(&grid, &element);

    let a = batched::LaplaceSingleLayerAssembler::<f64>::default();

    let matrix = a.parallel_assemble_singular_into_csr(&space, &space);

    if rank == 0 {
        // Gather sparse matrices onto process 0
        let mut rows = vec![];
        let mut cols = vec![];
        let mut data = vec![];

        let mut r = 0;
        for (i, index) in matrix.indices().iter().enumerate() {
            while i >= matrix.indptr()[r + 1] {
                r += 1;
            }
            rows.push(r);
            cols.push(*index);
            data.push(matrix.data()[i]);
        }
        for p in 1..size {
            let process = comm.process_at_rank(p);
            let (indices, _status) = process.receive_vec::<usize>();
            let (indptr, _status) = process.receive_vec::<usize>();
            let (subdata, _status) = process.receive_vec::<f64>();
            let mat = CsrMatrix::new(
                [indptr.len() + 1, indptr.len() + 1],
                indices,
                indptr,
                subdata,
            );

            let mut r = 0;
            for (i, index) in mat.indices().iter().enumerate() {
                while i >= mat.indptr()[r + 1] {
                    r += 1;
                }
                rows.push(r);
                cols.push(*index);
                data.push(mat.data()[i]);
            }
        }
        let full_matrix = CsrMatrix::from_aij(
            [space.global_size(), space.global_size()],
            &rows,
            &cols,
            &data,
        )
        .unwrap();

        // Compare to matrix assembled on just this process
        let serial_grid = example_flat_triangle_grid_serial(n);
        let serial_space = SerialFunctionSpace::new(&serial_grid, &element);
        let serial_matrix = a.assemble_singular_into_csr(&serial_space, &serial_space);

        for (i, j) in full_matrix.indices().iter().zip(serial_matrix.indices()) {
            assert_eq!(i, j);
        }
        for (i, j) in full_matrix.indptr().iter().zip(serial_matrix.indptr()) {
            assert_eq!(i, j);
        }
        for (i, j) in full_matrix.data().iter().zip(serial_matrix.data()) {
            assert_relative_eq!(i, j, epsilon = 1e-10);
        }
    } else {
        mpi::request::scope(|scope| {
            let _ = WaitGuard::from(
                comm.process_at_rank(0)
                    .immediate_send(scope, matrix.indices()),
            );
            let _ = WaitGuard::from(
                comm.process_at_rank(0)
                    .immediate_send(scope, matrix.indptr()),
            );
            let _ = WaitGuard::from(comm.process_at_rank(0).immediate_send(scope, matrix.data()));
        });
    }
}

#[cfg(feature = "mpi")]
fn test_parallel_assembly_single_element_grid<C: Communicator>(
    comm: &C,
    degree: usize,
    cont: Continuity,
) {
    let rank = comm.rank();
    let size = comm.size();

    let n = 10;
    let grid = example_single_element_grid(comm, n);
    let element = LagrangeElementFamily::<f64>::new(degree, cont);
    let space = ParallelFunctionSpace::new(&grid, &element);

    let a = batched::LaplaceSingleLayerAssembler::<f64>::default();

    let matrix = a.parallel_assemble_singular_into_csr(&space, &space);

    if rank == 0 {
        // Gather sparse matrices onto process 0
        let mut rows = vec![];
        let mut cols = vec![];
        let mut data = vec![];

        let mut r = 0;
        for (i, index) in matrix.indices().iter().enumerate() {
            while i >= matrix.indptr()[r + 1] {
                r += 1;
            }
            rows.push(r);
            cols.push(*index);
            data.push(matrix.data()[i]);
        }
        for p in 1..size {
            let process = comm.process_at_rank(p);
            let (indices, _status) = process.receive_vec::<usize>();
            let (indptr, _status) = process.receive_vec::<usize>();
            let (subdata, _status) = process.receive_vec::<f64>();
            let mat = CsrMatrix::new(
                [indptr.len() + 1, indptr.len() + 1],
                indices,
                indptr,
                subdata,
            );

            let mut r = 0;
            for (i, index) in mat.indices().iter().enumerate() {
                while i >= mat.indptr()[r + 1] {
                    r += 1;
                }
                rows.push(r);
                cols.push(*index);
                data.push(mat.data()[i]);
            }
        }
        let full_matrix = CsrMatrix::from_aij(
            [space.global_size(), space.global_size()],
            &rows,
            &cols,
            &data,
        )
        .unwrap();

        // Compare to matrix assembled on just this process
        let serial_grid = example_single_element_grid_serial(n);
        let serial_space = SerialFunctionSpace::new(&serial_grid, &element);
        let serial_matrix = a.assemble_singular_into_csr(&serial_space, &serial_space);

        for (i, j) in full_matrix.indices().iter().zip(serial_matrix.indices()) {
            assert_eq!(i, j);
        }
        for (i, j) in full_matrix.indptr().iter().zip(serial_matrix.indptr()) {
            assert_eq!(i, j);
        }
        for (i, j) in full_matrix.data().iter().zip(serial_matrix.data()) {
            assert_relative_eq!(i, j, epsilon = 1e-10);
        }
    } else {
        mpi::request::scope(|scope| {
            let _ = WaitGuard::from(
                comm.process_at_rank(0)
                    .immediate_send(scope, matrix.indices()),
            );
            let _ = WaitGuard::from(
                comm.process_at_rank(0)
                    .immediate_send(scope, matrix.indptr()),
            );
            let _ = WaitGuard::from(comm.process_at_rank(0).immediate_send(scope, matrix.data()));
        });
    }
}

#[cfg(feature = "mpi")]
fn test_parallel_assembly_mixed_grid<C: Communicator>(comm: &C, degree: usize, cont: Continuity) {
    let rank = comm.rank();
    let size = comm.size();

    let n = 10;
    let grid = example_mixed_grid(comm, n);
    let element = LagrangeElementFamily::<f64>::new(degree, cont);
    let space = ParallelFunctionSpace::new(&grid, &element);

    let a = batched::LaplaceSingleLayerAssembler::<f64>::default();

    let matrix = a.parallel_assemble_singular_into_csr(&space, &space);

    if rank == 0 {
        // Gather sparse matrices onto process 0
        let mut rows = vec![];
        let mut cols = vec![];
        let mut data = vec![];

        let mut r = 0;
        for (i, index) in matrix.indices().iter().enumerate() {
            while i >= matrix.indptr()[r + 1] {
                r += 1;
            }
            rows.push(r);
            cols.push(*index);
            data.push(matrix.data()[i]);
        }
        for p in 1..size {
            let process = comm.process_at_rank(p);
            let (indices, _status) = process.receive_vec::<usize>();
            let (indptr, _status) = process.receive_vec::<usize>();
            let (subdata, _status) = process.receive_vec::<f64>();
            let mat = CsrMatrix::new(
                [indptr.len() + 1, indptr.len() + 1],
                indices,
                indptr,
                subdata,
            );

            let mut r = 0;
            for (i, index) in mat.indices().iter().enumerate() {
                while i >= mat.indptr()[r + 1] {
                    r += 1;
                }
                rows.push(r);
                cols.push(*index);
                data.push(mat.data()[i]);
            }
        }
        let full_matrix = CsrMatrix::from_aij(
            [space.global_size(), space.global_size()],
            &rows,
            &cols,
            &data,
        )
        .unwrap();

        // Compare to matrix assembled on just this process
        let serial_grid = example_mixed_grid_serial(n);
        let serial_space = SerialFunctionSpace::new(&serial_grid, &element);
        let serial_matrix = a.assemble_singular_into_csr(&serial_space, &serial_space);

        for (i, j) in full_matrix.indices().iter().zip(serial_matrix.indices()) {
            assert_eq!(i, j);
        }
        for (i, j) in full_matrix.indptr().iter().zip(serial_matrix.indptr()) {
            assert_eq!(i, j);
        }
        for (i, j) in full_matrix.data().iter().zip(serial_matrix.data()) {
            assert_relative_eq!(i, j, epsilon = 1e-10);
        }
    } else {
        mpi::request::scope(|scope| {
            let _ = WaitGuard::from(
                comm.process_at_rank(0)
                    .immediate_send(scope, matrix.indices()),
            );
            let _ = WaitGuard::from(
                comm.process_at_rank(0)
                    .immediate_send(scope, matrix.indptr()),
            );
            let _ = WaitGuard::from(comm.process_at_rank(0).immediate_send(scope, matrix.data()));
        });
    }
}

#[cfg(feature = "mpi")]
fn main() {
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        println!("Testing FlatTriangleGrid in parallel.");
    }
    test_parallel_flat_triangle_grid(&world);
    for degree in 0..4 {
        if rank == 0 {
            println!("Testing assembly with DP{degree} using FlatTriangleGrid in parallel.");
        }
        test_parallel_assembly_flat_triangle_grid(&world, degree, Continuity::Discontinuous);
        if rank == 0 {
            println!("Testing assembly with DP{degree} using SingleElementGrid in parallel.");
        }
        test_parallel_assembly_single_element_grid(&world, degree, Continuity::Discontinuous);
        if rank == 0 {
            println!("Testing assembly with DP{degree} using MixedGrid in parallel.");
        }
        test_parallel_assembly_mixed_grid(&world, degree, Continuity::Discontinuous);
    }
    for degree in 1..4 {
        if rank == 0 {
            println!("Testing assembly with P{degree} using FlatTriangleGrid in parallel.");
        }
        test_parallel_assembly_flat_triangle_grid(&world, degree, Continuity::Continuous);
        if rank == 0 {
            println!("Testing assembly with P{degree} using SingleElementGrid in parallel.");
        }
        test_parallel_assembly_single_element_grid(&world, degree, Continuity::Continuous);
        if rank == 0 {
            println!("Testing assembly with P{degree} using MixedGrid in parallel.");
        }
        test_parallel_assembly_mixed_grid(&world, degree, Continuity::Continuous);
    }
}
#[cfg(not(feature = "mpi"))]
fn main() {}
