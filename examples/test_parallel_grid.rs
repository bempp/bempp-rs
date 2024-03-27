//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
use approx::assert_relative_eq;
#[cfg(feature = "mpi")]
use bempp::{grid::{flat_triangle_grid::{FlatTriangleGrid, FlatTriangleGridBuilder}, parallel_grid::ParallelGrid},
traits::{
    grid::{Builder, CellType, GeometryType, GridType, ParallelBuilder, PointType},
    element::Continuity, 
    types::Ownership,
},
    element::ciarlet::LagrangeElementFamily,
    bem::{function_space::{SerialFunctionSpace, ParallelFunctionSpace}, assembly::batched, assembly::batched::BatchedAssembler},
    
};
#[cfg(feature = "mpi")]
use mpi::{
    environment::Universe,
    request::WaitGuard,
    traits::{Communicator, Destination, Source},
};
#[cfg(feature = "mpi")]
use std::collections::HashMap;

extern crate blas_src;
extern crate lapack_src;

#[cfg(feature = "mpi")]
fn example_flat_triangle_grid<'a, C: Communicator>(comm: &'a C, n: usize) -> ParallelGrid<'a, C, FlatTriangleGrid<f64>> {
    let rank = comm.rank();
    let size = comm.size();

    let mut b = FlatTriangleGridBuilder::<f64>::new(());

    if rank == 0 {
        for y in 0..n {
            for x in 0..n {
                b.add_point(
                    y * n + x,
                    [x as f64 / (n - 1) as f64, y as f64 / (n - 1) as f64, 0.0],
                );
            }
        }

        let ncells = 2 * (n - 1).pow(2);
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
            let _sreq2 =
                WaitGuard::from(comm.process_at_rank(0).immediate_send(scope, &nvertices));
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
fn test_parallel_assembly_flat_triangle_grid<C: Communicator>(comm: &C) {
    let rank = comm.rank();
    let size = comm.size();

    let grid = example_flat_triangle_grid(comm, 10);
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Continuous);
    let space = ParallelFunctionSpace::new(&grid, &element);

    let a = batched::LaplaceSingleLayerAssembler::<f64>::default();

    println!("[{rank}] Lagrange single layer matrix (singular part) as CSR matrix");
    let singular_sparse_matrix = a.assemble_singular_into_csr(&space, &space);
    println!("[{rank}] indices: {:?}", singular_sparse_matrix.indices());
    println!("[{rank}] indptr: {:?}", singular_sparse_matrix.indptr());
    println!("[{rank}] data: {:?}", singular_sparse_matrix.data());
    println!();
}

#[cfg(feature = "mpi")]
fn main() {
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    if rank == 1 {
        println!("Testing FlatTriangleGrid in parallel.");
    }
    test_parallel_flat_triangle_grid(&world);
    if rank == 1 {
        println!("Testing assembly using FlatTriangleGrid in parallel.");
    }
    test_parallel_assembly_flat_triangle_grid(&world);
}
#[cfg(not(feature = "mpi"))]
fn main() {}
