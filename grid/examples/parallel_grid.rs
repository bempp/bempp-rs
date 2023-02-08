//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
use mpi::{environment::Universe, request::WaitGuard, topology::Communicator, traits::*};
#[cfg(feature = "mpi")]
use solvers_tools::arrays::{AdjacencyList, Array2D};

#[cfg(feature = "mpi")]
pub struct GridPiece<'a, C: Communicator> {
    pub comm: &'a C,
    pub cells: AdjacencyList<usize>,
    pub cell_owners: Vec<Ownership>,
    pub points: Array2D<f64>,
    pub point_owners: Vec<Ownership>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum Ownership {
    Owned,
    Ghost(usize, usize),
}

#[cfg(feature = "mpi")]
fn create_grid_part<'a, C: Communicator>(
    comm: &'a C,
    sizes: &Vec<usize>,
    flat_cells: &Vec<usize>,
    vertices: &Vec<usize>,
    ghosts: &Vec<usize>,
    ghost_owners: &Vec<i32>,
    ghost_indices: &Vec<usize>,
    coordinates: &Vec<f64>,
) -> GridPiece<'a, C> {
    let gdim = coordinates.len() / vertices.len();
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;

    // TODO: set this all up in process 1

    let mut cell_owners = vec![];
    let mut cells = AdjacencyList::new();
    let mut i = 0;
    for s in sizes {
        cells.add_row(&flat_cells[i..i + s]);
        cell_owners.push(Ownership::Owned);
        i += s;
    }

    let mut point_owners = vec![];
    let mut points = Array2D::<f64>::new((vertices.len() + ghosts.len(), gdim));
    let mut i = 0;
    for v in 0..vertices.len() {
        for d in 0..gdim {
            unsafe {
                *points.get_unchecked_mut(v, d) = coordinates[i];
            }
            point_owners.push(Ownership::Owned);
            i += 1;
        }
    }

    let mut ghosts_to_send = vec![vec![]; size];
    for (g, (o, i)) in ghosts.iter().zip(ghost_owners.iter().zip(ghost_indices)) {
        println!("{} {} {}", g, o, i);
        ghosts_to_send[*o as usize].push(*i);
        point_owners.push(Ownership::Ghost(*o as usize, *i));
    }
    for p in 0..size {
        if p != rank {
            mpi::request::scope(|scope| {
                let _sreq = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &ghosts_to_send[p][..]),
                );
            });
        }
    }
    let mut coords_to_send = vec![vec![]; size];
    for p in 0..size {
        if p != rank {
            let (data, status) = comm.process_at_rank(p as i32).receive_vec::<usize>();
            for i in data {
                for d in 0..gdim {
                    coords_to_send[p].push(unsafe { *points.get_unchecked(i, d) });
                }
            }
        }
    }
    // TODO: send cells adjacent to ghost points, and add their vertices to ghosts
    for p in 0..size {
        if p != rank {
            mpi::request::scope(|scope| {
                let _sreq = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &coords_to_send[p][..]),
                );
            });
        }
    }
    for p in 0..size {
        if p != rank {
            let (data, status) = comm.process_at_rank(p as i32).receive_vec::<f64>();
            let mut i = 0;
            for v in 0..data.len() / gdim {
                for d in 0..gdim {
                    unsafe {
                        *points.get_unchecked_mut(v, d) = data[i];
                    }
                    i += 1;
                }
            }
        }
    }

    // Communicate ghost cells

    GridPiece {
        comm: &comm,
        cells: cells,
        cell_owners: cell_owners,
        points: points,
        point_owners: point_owners,
    }
}

#[cfg(feature = "mpi")]
fn create_parallel_grid<'a, C: Communicator>(
    comm: &'a C,
    points: &Array2D<f64>,
    cells: &AdjacencyList<usize>,
    owners: &Vec<i32>,
) -> GridPiece<'a, C> {
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;

    let mut cells_per_proc = vec![vec![]; size];
    let mut sizes_per_proc = vec![vec![]; size];
    let mut v_owners = vec![(-1, 0); points.shape().0];
    let mut vertices_per_proc = vec![vec![]; size];
    let mut ghosts_per_proc = vec![vec![]; size];
    let mut ghost_owners_per_proc = vec![vec![]; size];
    let mut ghost_indices_per_proc = vec![vec![]; size];
    let mut coordinates_per_proc = vec![vec![]; size];

    for (c, o) in cells.iter_rows().zip(owners.iter()) {
        for v in c {
            cells_per_proc[*o as usize].push(*v);
            // TODO: could parts of this ghost data be computed on different processes?
            if v_owners[*v].0 == -1 {
                v_owners[*v] = (*o, vertices_per_proc[*o as usize].len());
                vertices_per_proc[*o as usize].push(*v);
                for x in unsafe { points.row_unchecked(*v) } {
                    coordinates_per_proc[*o as usize].push(*x)
                }
            } else if !vertices_per_proc[*o as usize].contains(v)
                && !ghosts_per_proc[*o as usize].contains(v)
            {
                ghosts_per_proc[*o as usize].push(*v);
                ghost_owners_per_proc[*o as usize].push(v_owners[*v].0);
                ghost_indices_per_proc[*o as usize].push(v_owners[*v].1);
            }
        }
        sizes_per_proc[*o as usize].push(c.len());
    }

    mpi::request::scope(|scope| {
        for p in 1..size {
            let _sreq = WaitGuard::from(
                comm.process_at_rank(p as i32)
                    .immediate_send(scope, &sizes_per_proc[p][..]),
            );
            let _sreq2 = WaitGuard::from(
                comm.process_at_rank(p as i32)
                    .immediate_send(scope, &cells_per_proc[p][..]),
            );
            let _sreq3 = WaitGuard::from(
                comm.process_at_rank(p as i32)
                    .immediate_send(scope, &vertices_per_proc[p][..]),
            );
            let _sreq3 = WaitGuard::from(
                comm.process_at_rank(p as i32)
                    .immediate_send(scope, &ghosts_per_proc[p][..]),
            );
            let _sreq3 = WaitGuard::from(
                comm.process_at_rank(p as i32)
                    .immediate_send(scope, &ghost_owners_per_proc[p][..]),
            );
            let _sreq3 = WaitGuard::from(
                comm.process_at_rank(p as i32)
                    .immediate_send(scope, &ghost_indices_per_proc[p][..]),
            );
            let _sreq3 = WaitGuard::from(
                comm.process_at_rank(p as i32)
                    .immediate_send(scope, &coordinates_per_proc[p][..]),
            );
        }
    });

    create_grid_part(
        &comm,
        &sizes_per_proc[rank],
        &cells_per_proc[rank],
        &vertices_per_proc[rank],
        &ghosts_per_proc[rank],
        &ghost_owners_per_proc[rank],
        &ghost_indices_per_proc[rank],
        &coordinates_per_proc[rank],
    )
    // Decide who owns each vertex, send this info out

    // send vertex coordinates
}

fn receive_grid_data<'a, C: Communicator>(comm: &'a C, root_rank: i32) -> GridPiece<'a, C> {
    let root_process = comm.process_at_rank(root_rank);
    let (sizes_received, status) = root_process.receive_vec::<usize>();
    let (cells_received, status) = root_process.receive_vec::<usize>();
    let (vertices_received, status) = root_process.receive_vec::<usize>();
    let (ghosts_received, status) = root_process.receive_vec::<usize>();
    let (ghost_owners_received, status) = root_process.receive_vec::<i32>();
    let (ghost_indices_received, status) = root_process.receive_vec::<usize>();
    let (coordinates_received, status) = root_process.receive_vec::<f64>();

    create_grid_part(
        &comm,
        &sizes_received,
        &cells_received,
        &vertices_received,
        &ghosts_received,
        &ghost_owners_received,
        &ghost_indices_received,
        &coordinates_received,
    )
}

#[cfg(feature = "mpi")]
fn test_parallel_grid() {
    // Setup an MPI environment
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();

    // let comm = world.duplicate();

    let rank = world.rank();
    let size = world.size();

    // This test only works on 2 processes
    //if world.size() != 2 { return; }

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
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                cells.add_row(&vec![j * n + i, j * n + i + 1, j * n + i + n + 1]);
                cells.add_row(&vec![j * n + i, j * n + i + n + 1, j * n + i + n]);
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
                owners[c] = r;
                c += 1;
            }
        }

        create_parallel_grid(&world, &pts, &cells, &owners)
    } else {
        receive_grid_data(&world, 0)
    };

    println!("{} {}", rank, grid.cells.num_rows());
    /*
    // Setup tree parameters
    // let adaptive = false;
    let adaptive = true;
    let n_crit = Some(50);
    // let n_crit: Option<_> = None;
    let depth: Option<_> = None;
    // let depth = Some(3);
    let n_points = 10000;
    let k: Option<_> = None;

    let points = points_fixture(n_points);

    let mut tree = MultiNodeTree::new(&comm, k, &points, adaptive, n_crit, depth);

    tree.create_let();

    println!(
        "rank {:?} has {:?} leaves",
        tree.world.rank(),
        tree.leaves.len()
    );
    */
}

#[cfg(feature = "mpi")]
fn main() {
    test_parallel_grid()
}
#[cfg(not(feature = "mpi"))]
fn main() {}
