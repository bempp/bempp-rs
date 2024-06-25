//! Mixed grid

use crate::grid::mixed_grid::{geometry::MixedGeometry, topology::MixedTopology};
use crate::grid::traits::Grid;
use log::warn;
use ndelement::ciarlet::lagrange;
use ndelement::reference_cell;
use ndelement::traits::FiniteElement;
use ndelement::types::Continuity;
use ndelement::types::ReferenceCellType;
use num::Float;
use rlst::RlstScalar;
use rlst::{
    dense::array::{views::ArrayViewMut, Array},
    BaseArray, MatrixInverse, VectorContainer,
};
use std::collections::HashMap;

/// A mixed grid
pub struct MixedGrid<T: Float + RlstScalar<Real = T>> {
    topology: MixedTopology,
    pub(crate) geometry: MixedGeometry<T>,
}

impl<T: Float + RlstScalar<Real = T>> MixedGrid<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    #[allow(clippy::too_many_arguments)]
    /// Create a mixed grid
    pub fn new(
        points: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        cells: &[usize],
        cell_types: &[ReferenceCellType],
        cell_degrees: &[usize],
        point_indices_to_ids: Vec<usize>,
        point_ids_to_indices: HashMap<usize, usize>,
        cell_indices_to_ids: Vec<usize>,
        edge_ids: Option<HashMap<[usize; 2], usize>>,
    ) -> Self {
        let mut element_info = vec![];
        let mut element_numbers = vec![];

        for (cell, degree) in cell_types.iter().zip(cell_degrees) {
            let info = (*cell, *degree);
            if !element_info.contains(&info) {
                element_info.push(info);
            }
            element_numbers.push(element_info.iter().position(|&i| i == info).unwrap());
        }

        let elements = element_info
            .iter()
            .map(|(i, j)| lagrange::create::<T>(*i, *j, Continuity::Continuous))
            .collect::<Vec<_>>();

        if elements.len() == 1 {
            warn!("Creating a mixed grid with only one element. Using a SingleElementGrid would be faster.");
        }

        let mut cell_vertices = vec![];

        let mut start = 0;
        for (cell_type, e_n) in cell_types.iter().zip(&element_numbers) {
            let nvertices = reference_cell::entity_counts(*cell_type)[0];
            let npoints = elements[*e_n].dim();
            cell_vertices.extend_from_slice(&cells[start..start + nvertices]);
            start += npoints;
        }

        let topology = MixedTopology::new(
            &cell_vertices,
            cell_types,
            &point_indices_to_ids,
            &cell_indices_to_ids,
            edge_ids,
        );
        let geometry = MixedGeometry::<T>::new(
            points,
            cells,
            elements,
            &element_numbers,
            point_indices_to_ids,
            point_ids_to_indices,
            &cell_indices_to_ids,
        );

        Self { topology, geometry }
    }
}

impl<T: Float + RlstScalar<Real = T>> Grid for MixedGrid<T> {
    type T = T;
    type Topology = MixedTopology;
    type Geometry = MixedGeometry<T>;

    fn topology(&self) -> &Self::Topology {
        &self.topology
    }

    fn geometry(&self) -> &Self::Geometry {
        &self.geometry
    }

    fn is_serial(&self) -> bool {
        true
    }
}
