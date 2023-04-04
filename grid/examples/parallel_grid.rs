//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
use bempp_grid::parallel_grid::ParallelGrid;
#[cfg(feature = "mpi")]
use bempp_tools::arrays::{AdjacencyList, Array2D};
#[cfg(feature = "mpi")]
use bempp_traits::cell::ReferenceCellType;
#[cfg(feature = "mpi")]
use bempp_traits::grid::{Geometry, Grid, Ownership, Topology};
#[cfg(feature = "mpi")]
use mpi::{environment::Universe, request::WaitGuard, topology::Communicator, traits::*};

#[cfg(feature = "mpi")]
use approx::*;

#[cfg(feature = "mpi")]
fn test_parallel_grid() {
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let size = world.size();

    let n = 10;

    let grid = if rank == 0 {
        let mut pts = Array2D::new((n * n, 3));
        let mut i = 0;
        for y in 0..n {
            for x in 0..n {
                *pts.get_mut(i, 0).unwrap() = x as f64 / (n - 1) as f64;
                *pts.get_mut(i, 1).unwrap() = y as f64 / (n - 1) as f64;
                *pts.get_mut(i, 2).unwrap() = 0.0;
                i += 1;
            }
        }

        let mut cells = AdjacencyList::<usize>::new();
        let mut cell_types = vec![];
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                cells.add_row(&vec![j * n + i, j * n + i + 1, j * n + i + n + 1]);
                cells.add_row(&vec![j * n + i, j * n + i + n + 1, j * n + i + n]);
                cell_types.push(ReferenceCellType::Triangle);
                cell_types.push(ReferenceCellType::Triangle);
            }
        }

        let mut owners = vec![0; cells.num_rows()];
        let mut c = 0;
        for r in 0..size {
            let end = if r + 1 == size {
                cells.num_rows()
            } else {
                (r + 1) as usize * cells.num_rows() / size as usize
            };
            while c < end {
                owners[c] = r as usize;
                c += 1;
            }
        }

        ParallelGrid::new(&world, pts, cells, cell_types, owners)
    } else {
        ParallelGrid::new_subprocess(&world, 0)
    };

    let mut area = 0.0;
    for (ti, gi) in grid
        .topology()
        .index_map()
        .iter()
        .zip(grid.geometry().index_map().iter())
    {
        if grid.topology().entity_ownership(2, *ti) == Ownership::Owned {
            let vs = grid.geometry().cell_vertices(*gi).unwrap();
            let v0 = grid.geometry().point(vs[0]).unwrap();
            let v1 = grid.geometry().point(vs[1]).unwrap();
            let v2 = grid.geometry().point(vs[2]).unwrap();
            let edge1 = vec![v1[0] - v0[0], v1[1] - v0[1]];
            let edge2 = vec![v2[0] - v0[0], v2[1] - v0[1]];
            area += ((edge1[0] * edge2[1] - edge1[1] * edge2[0]) / 2.0).abs();
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
    for v in 0..grid.topology().entity_count(0) {
        if grid.topology().entity_ownership(0, v) == Ownership::Owned {
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
