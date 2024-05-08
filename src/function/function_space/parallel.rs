//! Parallel function space

use crate::element::ciarlet::CiarletElement;
use crate::function::{function_space::assign_dofs, SerialFunctionSpace};
use crate::traits::{
    element::ElementFamily,
    function::{FunctionSpace, FunctionSpaceInParallel},
    grid::{GridType, ParallelGridType},
    types::{Ownership, ReferenceCellType},
};
use mpi::{
    point_to_point::{Destination, Source},
    request::WaitGuard,
    topology::Communicator,
};
use rlst::RlstScalar;
use std::collections::HashMap;

/// The local function space on a process
pub struct LocalFunctionSpace<'a, T: RlstScalar, GridImpl: GridType<T = T::Real>> {
    serial_space: SerialFunctionSpace<'a, T, GridImpl>,
    global_size: usize,
    global_dof_numbers: Vec<usize>,
    ownership: Vec<Ownership>,
}

impl<'a, T: RlstScalar, GridImpl: GridType<T = T::Real>> FunctionSpace
    for LocalFunctionSpace<'a, T, GridImpl>
{
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<T>;

    fn grid(&self) -> &Self::Grid {
        self.serial_space.grid
    }
    fn element(&self, cell_type: ReferenceCellType) -> &CiarletElement<T> {
        self.serial_space.element(cell_type)
    }
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        self.serial_space
            .get_local_dof_numbers(entity_dim, entity_number)
    }
    fn local_size(&self) -> usize {
        self.serial_space.local_size()
    }
    fn global_size(&self) -> usize {
        self.global_size
    }
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]> {
        self.serial_space.cell_dofs(cell)
    }
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>> {
        self.serial_space.cell_colouring()
    }
    fn global_dof_index(&self, local_dof_index: usize) -> usize {
        self.global_dof_numbers[local_dof_index]
    }
    fn ownership(&self, local_dof_index: usize) -> Ownership {
        self.ownership[local_dof_index]
    }
}

/// A parallel function space
pub struct ParallelFunctionSpace<
    'a,
    T: RlstScalar,
    GridImpl: ParallelGridType + GridType<T = T::Real>,
> {
    grid: &'a GridImpl,
    local_space: LocalFunctionSpace<'a, T, <GridImpl as ParallelGridType>::LocalGridType>,
}

impl<'a, T: RlstScalar, GridImpl: ParallelGridType + GridType<T = T::Real>>
    ParallelFunctionSpace<'a, T, GridImpl>
{
    /// Create new function space
    pub fn new(
        grid: &'a GridImpl,
        e_family: &impl ElementFamily<T = T, FiniteElement = CiarletElement<T>>,
    ) -> Self {
        let comm = grid.comm();
        let rank = comm.rank();
        let size = comm.size();

        // Create local space on current process
        let (cell_dofs, entity_dofs, dofmap_size, owner_data) =
            assign_dofs(rank as usize, grid, e_family);

        let mut elements = HashMap::new();
        for cell in grid.cell_types() {
            elements.insert(*cell, e_family.element(*cell));
        }

        // Assign global DOF numbers
        let mut global_dof_numbers = vec![0; dofmap_size];
        let mut ghost_indices = vec![vec![]; size as usize];
        let mut ghost_dims = vec![vec![]; size as usize];
        let mut ghost_entities = vec![vec![]; size as usize];
        let mut ghost_entity_dofs = vec![vec![]; size as usize];

        let local_offset = if rank == 0 {
            0
        } else {
            let (value, _status) = comm.process_at_rank(rank - 1).receive::<usize>();
            value
        };
        let mut dof_n = local_offset;
        for (i, ownership) in owner_data.iter().enumerate() {
            if ownership.0 == rank as usize {
                global_dof_numbers[i] = dof_n;
                dof_n += 1;
            } else {
                ghost_indices[ownership.0].push(i);
                ghost_dims[ownership.0].push(ownership.1);
                ghost_entities[ownership.0].push(ownership.2);
                ghost_entity_dofs[ownership.0].push(ownership.3);
            }
        }
        if rank < size - 1 {
            mpi::request::scope(|scope| {
                let _ =
                    WaitGuard::from(comm.process_at_rank(rank + 1).immediate_send(scope, &dof_n));
            });
        }

        let global_size = if rank == size - 1 {
            for p in 0..rank {
                mpi::request::scope(|scope| {
                    let _ = WaitGuard::from(comm.process_at_rank(p).immediate_send(scope, &dof_n));
                });
            }
            dof_n
        } else {
            let (gs, _status) = comm.process_at_rank(size - 1).receive::<usize>();
            gs
        };

        // Communicate information about ghosts
        // send requests for ghost info
        for p in 0..size {
            if p != rank {
                mpi::request::scope(|scope| {
                    let process = comm.process_at_rank(p);
                    let _ = WaitGuard::from(process.immediate_send(scope, &ghost_dims[p as usize]));
                    let _ =
                        WaitGuard::from(process.immediate_send(scope, &ghost_entities[p as usize]));
                    let _ = WaitGuard::from(
                        process.immediate_send(scope, &ghost_entity_dofs[p as usize]),
                    );
                });
            }
        }
        // accept requests and send ghost info
        for p in 0..size {
            if p != rank {
                let process = comm.process_at_rank(p);
                let (gdims, _status) = process.receive_vec::<usize>();
                let (gentities, _status) = process.receive_vec::<usize>();
                let (gentity_dofs, _status) = process.receive_vec::<usize>();
                let local_ghost_dofs = gdims
                    .iter()
                    .zip(gentities.iter().zip(&gentity_dofs))
                    .map(|(c, (e, d))| entity_dofs[*c][*e][*d])
                    .collect::<Vec<_>>();
                let global_ghost_dofs = local_ghost_dofs
                    .iter()
                    .map(|i| global_dof_numbers[*i])
                    .collect::<Vec<_>>();
                mpi::request::scope(|scope| {
                    let _ = WaitGuard::from(process.immediate_send(scope, &local_ghost_dofs));
                    let _ = WaitGuard::from(process.immediate_send(scope, &global_ghost_dofs));
                });
            }
        }

        // receive ghost info
        let mut ownership = vec![Ownership::Owned; dofmap_size];
        for p in 0..size {
            if p != rank {
                let process = comm.process_at_rank(p);
                let (local_ghost_dofs, _status) = process.receive_vec::<usize>();
                let (global_ghost_dofs, _status) = process.receive_vec::<usize>();
                for (i, (l, g)) in ghost_indices[p as usize]
                    .iter()
                    .zip(local_ghost_dofs.iter().zip(&global_ghost_dofs))
                {
                    global_dof_numbers[*i] = *g;
                    ownership[*i] = Ownership::Ghost(p as usize, *l);
                }
            }
        }

        println!("[{rank}] cell_dofs = {cell_dofs:?}");
        println!("[{rank}] entity_dofs = {entity_dofs:?}");
        println!("[{rank}] dofmap_size = {dofmap_size:?}");
        println!("[{rank}] owner_data = {owner_data:?}");
        println!("[{rank}] global_dof_numbers = {global_dof_numbers:?}");

        

        let serial_space = SerialFunctionSpace {
            grid: grid.local_grid(),
            elements,
            entity_dofs,
            cell_dofs,
            size: dofmap_size,
        };
        let local_space = LocalFunctionSpace {
            serial_space,
            global_size,
            global_dof_numbers,
            ownership,
        };

        Self { grid, local_space }
    }
}

impl<'a, T: RlstScalar, GridImpl: ParallelGridType + GridType<T = T::Real>> FunctionSpaceInParallel
    for ParallelFunctionSpace<'a, T, GridImpl>
{
    type ParallelGrid = GridImpl;
    type SerialSpace = LocalFunctionSpace<'a, T, <GridImpl as ParallelGridType>::LocalGridType>;

    /// Get the local space on the process
    fn local_space(
        &self,
    ) -> &LocalFunctionSpace<'a, T, <GridImpl as ParallelGridType>::LocalGridType> {
        &self.local_space
    }
}

impl<'a, T: RlstScalar, GridImpl: ParallelGridType + GridType<T = T::Real>> FunctionSpace
    for ParallelFunctionSpace<'a, T, GridImpl>
{
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<T>;

    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self, cell_type: ReferenceCellType) -> &CiarletElement<T> {
        self.local_space.element(cell_type)
    }
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        self.local_space
            .get_local_dof_numbers(entity_dim, entity_number)
    }
    fn local_size(&self) -> usize {
        self.local_space.local_size()
    }
    fn global_size(&self) -> usize {
        self.local_space.global_size()
    }
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]> {
        self.local_space.cell_dofs(cell)
    }
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>> {
        self.local_space.cell_colouring()
    }
    fn global_dof_index(&self, local_dof_index: usize) -> usize {
        self.local_space.global_dof_index(local_dof_index)
    }
    fn ownership(&self, local_dof_index: usize) -> Ownership {
        self.local_space.ownership(local_dof_index)
    }
}
