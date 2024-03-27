//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
fn test_parallel_grid() {
    use approx::*;
    //use bempp::grid::parallel_grid::ParallelGrid;
    use bempp::grid::flat_triangle_grid::FlatTriangleGridBuilder;
    use bempp::traits::{
        grid::{Builder, CellType, GeometryType, GridType, ParallelBuilder, PointType},
        types::Ownership,
    };
    use mpi::{
        environment::Universe,
        request::WaitGuard,
        traits::{Communicator, Destination, Source},
    };
    use std::collections::HashMap;

    extern crate blas_src;
    extern crate lapack_src;

    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let size = world.size();

    let n = 10;

    let mut b = FlatTriangleGridBuilder::<f64>::new(());

    let grid = if rank == 0 {
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
        b.create_parallel_grid(&world, &owners)
    } else {
        b.receive_parallel_grid(&world, 0)
    };

    let mut area = 0.0;
    for cell in grid.iter_all_cells() {
        if cell.ownership() == Ownership::Owned {
            area += cell.geometry().volume();
        }
    }
    if rank != 0 {
        mpi::request::scope(|scope| {
            let _sreq2 = WaitGuard::from(world.process_at_rank(0).immediate_send(scope, &area));
        });
    } else {
        for p in 1..size {
            let (a, _status) = world.process_at_rank(p as i32).receive::<f64>();
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
                WaitGuard::from(world.process_at_rank(0).immediate_send(scope, &nvertices));
        });
    } else {
        for p in 1..size {
            let (nv, _status) = world.process_at_rank(p as i32).receive::<usize>();
            nvertices += nv;
        }
        assert_eq!(nvertices, n * n);
    }
}

#[cfg(feature = "mpi")]
fn main() {
    test_parallel_grid()
}
#[cfg(not(feature = "mpi"))]
fn main() {}
