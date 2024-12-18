//! Parallel function space

use crate::function::{function_space::assign_dofs, FunctionSpace, MPIFunctionSpace};
use mpi::{
    point_to_point::{Destination, Source},
    request::WaitGuard,
    topology::Communicator,
};
use ndelement::traits::ElementFamily;
use ndelement::types::ReferenceCellType;
use ndelement::{ciarlet::CiarletElement, traits::FiniteElement};
use ndgrid::{
    traits::{Entity, Grid, ParallelGrid, Topology},
    types::Ownership,
};
use rlst::{MatrixInverse, RlstScalar};
use std::collections::HashMap;

/// The local function space on a process
pub struct LocalFunctionSpace<
    'a,
    T: RlstScalar + MatrixInverse,
    GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
> {
    grid: &'a GridImpl,
    elements: HashMap<ReferenceCellType, CiarletElement<T>>,
    entity_dofs: [Vec<Vec<usize>>; 4],
    cell_dofs: Vec<Vec<usize>>,
    local_size: usize,
    global_size: usize,
    global_dof_numbers: Vec<usize>,
    ownership: Vec<Ownership>,
}

impl<
        'a,
        T: RlstScalar + MatrixInverse,
        GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
    > LocalFunctionSpace<'a, T, GridImpl>
{
    // /// Create new function space
    // pub fn new(
    //     grid: &'a GridImpl,
    //     e_family: &impl ElementFamily<
    //         T = T,
    //         FiniteElement = CiarletElement<T>,
    //         CellType = ReferenceCellType,
    //     >,
    //     global_size: usize,
    //     global_dof_numbers: Vec<usize>,
    //     ownership: Vec<Ownership>,
    // ) -> Self {
    //     let (cell_dofs, entity_dofs, local_size, _) = assign_dofs(0, grid, e_family);

    //     let mut elements = HashMap::new();
    //     for cell in grid.entity_types(2) {
    //         elements.insert(*cell, e_family.element(*cell));
    //     }

    //     Self {
    //         grid,
    //         elements,
    //         entity_dofs,
    //         cell_dofs,
    //         local_size,
    //         global_size,
    //         global_dof_numbers,
    //         ownership,
    //     }
    // }
}

impl<
        T: RlstScalar + MatrixInverse,
        GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
    > FunctionSpace for LocalFunctionSpace<'_, T, GridImpl>
{
    type T = T;
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<T>;
    type LocalSpace<'b>
        = LocalFunctionSpace<'b, T, GridImpl>
    where
        Self: 'b;

    fn local_space(&self) -> &Self::LocalSpace<'_> {
        self
    }
    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self, cell_type: ReferenceCellType) -> &CiarletElement<T> {
        &self.elements[&cell_type]
    }
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        &self.entity_dofs[entity_dim][entity_number]
    }
    fn is_serial(&self) -> bool {
        false
    }
    fn local_size(&self) -> usize {
        self.local_size
    }
    fn global_size(&self) -> usize {
        self.global_size
    }
    unsafe fn cell_dofs_unchecked(&self, cell: usize) -> &[usize] {
        self.cell_dofs.get_unchecked(cell)
    }
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]> {
        if cell < self.cell_dofs.len() {
            Some(unsafe { self.cell_dofs_unchecked(cell) })
        } else {
            None
        }
    }
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>> {
        let mut colouring = HashMap::new();
        //: HashMap<ReferenceCellType, Vec<Vec<usize>>>
        for cell in self.grid.entity_types(2) {
            colouring.insert(*cell, vec![]);
        }
        let mut edim = 0;
        while self.elements[&self.grid.entity_types(2)[0]]
            .entity_dofs(edim, 0)
            .unwrap()
            .is_empty()
        {
            edim += 1;
        }

        let mut entity_colours = vec![
            vec![];
            if edim == 0 {
                self.grid.entity_count(ReferenceCellType::Point)
            } else if edim == 1 {
                self.grid.entity_count(ReferenceCellType::Interval)
            } else if edim == 2 && self.grid.topology_dim() == 2 {
                self.grid
                    .entity_types(2)
                    .iter()
                    .map(|&i| self.grid.entity_count(i))
                    .sum::<usize>()
            } else {
                unimplemented!();
            }
        ];

        for cell in self.grid.entity_iter(2) {
            let cell_type = cell.entity_type();
            let indices = cell.topology().sub_entity_iter(edim).collect::<Vec<_>>();

            let c = {
                let mut c = 0;
                while c < colouring[&cell_type].len() {
                    let mut found = false;
                    for v in &indices {
                        if entity_colours[*v].contains(&c) {
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        break;
                    }
                    c += 1;
                }
                c
            };
            if c == colouring[&cell_type].len() {
                for ct in self.grid.entity_types(2) {
                    colouring.get_mut(ct).unwrap().push(if *ct == cell_type {
                        vec![cell.local_index()]
                    } else {
                        vec![]
                    });
                }
            } else {
                colouring.get_mut(&cell_type).unwrap()[c].push(cell.local_index());
            }
            for v in &indices {
                entity_colours[*v].push(c);
            }
        }
        colouring
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
    C: Communicator,
    T: RlstScalar + MatrixInverse,
    GridImpl: ParallelGrid<C> + Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
> {
    grid: &'a GridImpl,
    local_space: LocalFunctionSpace<'a, T, <GridImpl as ParallelGrid<C>>::LocalGrid<'a>>,
}

impl<
        'a,
        C: Communicator,
        T: RlstScalar + MatrixInverse,
        GridImpl: ParallelGrid<C> + Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    > ParallelFunctionSpace<'a, C, T, GridImpl>
{
    /// Create new function space
    pub fn new(
        grid: &'a GridImpl,
        e_family: &impl ElementFamily<
            T = T,
            FiniteElement = CiarletElement<T>,
            CellType = ReferenceCellType,
        >,
    ) -> Self {
        let comm = grid.comm();
        let rank = comm.rank();
        let size = comm.size();

        // Create local space on current process
        let (cell_dofs, entity_dofs, dofmap_size, owner_data) =
            assign_dofs(rank as usize, grid.local_grid(), e_family);

        let mut elements = HashMap::new();
        for cell in grid.entity_types(grid.topology_dim()) {
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

        let local_space = LocalFunctionSpace {
            grid: grid.local_grid(),
            elements,
            entity_dofs,
            cell_dofs,
            local_size: dofmap_size,
            global_size,
            global_dof_numbers,
            ownership,
        };

        Self { grid, local_space }
    }
}

impl<
        C: Communicator,
        T: RlstScalar + MatrixInverse,
        GridImpl: ParallelGrid<C> + Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    > MPIFunctionSpace<C> for ParallelFunctionSpace<'_, C, T, GridImpl>
{
    fn comm(&self) -> &C {
        self.grid.comm()
    }
}

impl<
        'a,
        C: Communicator,
        T: RlstScalar + MatrixInverse,
        GridImpl: ParallelGrid<C> + Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    > FunctionSpace for ParallelFunctionSpace<'a, C, T, GridImpl>
{
    type T = T;
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<T>;
    type LocalSpace<'b>
        = LocalFunctionSpace<'b, T, <GridImpl as ParallelGrid<C>>::LocalGrid<'a>>
    where
        Self: 'b;

    fn local_space(
        &self,
    ) -> &LocalFunctionSpace<'_, T, <GridImpl as ParallelGrid<C>>::LocalGrid<'a>> {
        &self.local_space
    }

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
    fn is_serial(&self) -> bool {
        false
    }
    fn local_size(&self) -> usize {
        self.local_space.local_size()
    }
    fn global_size(&self) -> usize {
        self.local_space.global_size()
    }
    unsafe fn cell_dofs_unchecked(&self, cell: usize) -> &[usize] {
        self.local_space.cell_dofs_unchecked(cell)
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
