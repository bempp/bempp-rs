//! Parallel function space

use crate::function::{function_space::assign_dofs, FunctionSpace};
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
use std::{collections::HashMap, marker::PhantomData};
