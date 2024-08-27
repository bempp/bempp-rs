//? mpirun -n {{NPROCESSES}} --features "mpi"

#[cfg(feature = "mpi")]
use approx::assert_relative_eq;
#[cfg(feature = "mpi")]
use bempp::{
    assembly::boundary::BoundaryAssembler,
    function::{ParallelFunctionSpace, SerialFunctionSpace},
    traits::{BoundaryAssembly, FunctionSpace, ParallelBoundaryAssembly},
};
#[cfg(feature = "mpi")]
use itertools::izip;
#[cfg(feature = "mpi")]
use mpi::{
    collective::CommunicatorCollectives,
    environment::Universe,
    request::WaitGuard,
    traits::{Communicator, Destination, Source},
};
#[cfg(feature = "mpi")]
use ndelement::{
    ciarlet::{CiarletElement, LagrangeElementFamily},
    types::{Continuity, ReferenceCellType},
};
#[cfg(feature = "mpi")]
use ndgrid::{
    grid::parallel::ParallelGrid,
    traits::{Builder, Entity, Grid, ParallelBuilder},
    SingleElementGrid, SingleElementGridBuilder,
};
#[cfg(feature = "mpi")]
use rlst::{CsrMatrix, Shape};
#[cfg(feature = "mpi")]
use std::collections::{hash_map::Entry, HashMap};

#[cfg(feature = "mpi")]
fn create_single_element_grid_data(b: &mut SingleElementGridBuilder<f64>, n: usize) {
    for y in 0..n {
        for x in 0..n {
            b.add_point(
                y * n + x,
                &[x as f64 / (n - 1) as f64, y as f64 / (n - 1) as f64, 0.0],
            );
        }
    }

    for i in 0..n - 1 {
        for j in 0..n - 1 {
            b.add_cell(
                i * (n - 1) + j,
                &[j * n + i, j * n + i + 1, j * n + i + n, j * n + i + n + 1],
            );
        }
    }
}

#[cfg(feature = "mpi")]
fn example_single_element_grid<C: Communicator>(
    comm: &C,
    n: usize,
) -> ParallelGrid<'_, C, SingleElementGrid<f64, CiarletElement<f64>>> {
    let rank = comm.rank();

    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Quadrilateral, 1));

    if rank == 0 {
        create_single_element_grid_data(&mut b, n);
        b.create_parallel_grid(comm)
    } else {
        b.receive_parallel_grid(comm, 0)
    }
}

#[cfg(feature = "mpi")]
fn example_single_element_grid_serial(n: usize) -> SingleElementGrid<f64, CiarletElement<f64>> {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Quadrilateral, 1));
    create_single_element_grid_data(&mut b, n);
    b.create_grid()
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

    let a = BoundaryAssembler::<f64, _, _>::new_laplace_single_layer();

    let matrix = a.parallel_assemble_singular_into_csr(&space, &space);

    if rank == 0 {
        // Compute the same matrix on a single process
        let serial_grid = example_single_element_grid_serial(n);
        let serial_space = SerialFunctionSpace::new(&serial_grid, &element);
        let serial_matrix = a.assemble_singular_into_csr(&serial_space, &serial_space);

        // Dofs associated with each cell (by cell id)
        let mut serial_dofmap = HashMap::new();
        for cell in serial_grid.entity_iter(2) {
            serial_dofmap.insert(
                cell.id().unwrap(),
                serial_space
                    .cell_dofs(cell.local_index())
                    .unwrap()
                    .iter()
                    .map(|i| serial_space.global_dof_index(*i))
                    .collect::<Vec<_>>(),
            );
        }
        let mut parallel_dofmap = HashMap::new();
        for cell in grid.entity_iter(2) {
            parallel_dofmap.insert(
                cell.id().unwrap(),
                space
                    .cell_dofs(cell.local_index())
                    .unwrap()
                    .iter()
                    .map(|i| space.global_dof_index(*i))
                    .collect::<Vec<_>>(),
            );
        }
        for p in 1..size {
            let process = comm.process_at_rank(p);
            let (cell_ids, _status) = process.receive_vec::<usize>();
            let (dofs, _status) = process.receive_vec::<usize>();
            let (dofs_len, _status) = process.receive_vec::<usize>();
            let mut start = 0;
            for (id, len) in izip!(cell_ids, dofs_len) {
                if let Entry::Vacant(e) = parallel_dofmap.entry(id) {
                    e.insert(dofs[start..start + len].to_vec());
                } else {
                    assert_eq!(parallel_dofmap[&id], dofs[start..start + len]);
                }
                start += len;
            }
        }

        let mut index_map = vec![0; serial_space.global_size()];

        for (id, dofs) in parallel_dofmap {
            for (i, j) in izip!(&serial_dofmap[&id], dofs) {
                index_map[j] = *i;
            }
        }

        // Gather sparse matrices onto process 0
        let mut rows = vec![];
        let mut cols = vec![];
        let mut data = vec![];

        let mut r = 0;
        for (i, index) in matrix.indices().iter().enumerate() {
            while i >= matrix.indptr()[r + 1] {
                r += 1;
            }
            rows.push(index_map[r]);
            cols.push(index_map[*index]);
            data.push(matrix.data()[i]);
        }

        for p in 1..size {
            let process = comm.process_at_rank(p);
            let (indices, _status) = process.receive_vec::<usize>();
            let (indptr, _status) = process.receive_vec::<usize>();
            let (subdata, _status) = process.receive_vec::<f64>();
            let mat = CsrMatrix::new(matrix.shape(), indices, indptr, subdata);

            let mut r = 0;
            for (i, index) in mat.indices().iter().enumerate() {
                while i >= mat.indptr()[r + 1] {
                    r += 1;
                }
                rows.push(index_map[r]);
                cols.push(index_map[*index]);
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
        let mut cell_ids = vec![];
        let mut dofs = vec![];
        let mut dofs_len = vec![];
        for cell in grid.entity_iter(2) {
            cell_ids.push(cell.id().unwrap());
            let cell_dofs = space
                .cell_dofs(cell.local_index())
                .unwrap()
                .iter()
                .map(|i| space.global_dof_index(*i))
                .collect::<Vec<_>>();
            dofs.extend_from_slice(&cell_dofs);
            dofs_len.push(cell_dofs.len());
        }

        mpi::request::scope(|scope| {
            let root = comm.process_at_rank(0);
            // TODO: send this:
            let _ = WaitGuard::from(root.immediate_send(scope, &cell_ids));
            let _ = WaitGuard::from(root.immediate_send(scope, &dofs));
            let _ = WaitGuard::from(root.immediate_send(scope, &dofs_len));
            let _ = WaitGuard::from(root.immediate_send(scope, matrix.indices()));
            let _ = WaitGuard::from(root.immediate_send(scope, matrix.indptr()));
            let _ = WaitGuard::from(root.process_at_rank(0).immediate_send(scope, matrix.data()));
        });
    }
}

#[cfg(feature = "mpi")]
fn main() {
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    for degree in 0..4 {
        if rank == 0 {
            println!("Testing assembly with DP{degree} using SingleElementGrid in parallel.");
        }
        test_parallel_assembly_single_element_grid(&world, degree, Continuity::Discontinuous);
        world.barrier();
    }
    for degree in 1..4 {
        if rank == 0 {
            println!("Testing assembly with P{degree} using SingleElementGrid in parallel.");
        }
        test_parallel_assembly_single_element_grid(&world, degree, Continuity::Standard);
        world.barrier();
    }
}
#[cfg(not(feature = "mpi"))]
fn main() {}
