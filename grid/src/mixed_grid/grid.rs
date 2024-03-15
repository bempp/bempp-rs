//! Mixed grid

use crate::mixed_grid::{geometry::SerialMixedGeometry, topology::SerialMixedTopology};
use crate::traits::Grid;
use bempp_element::element::lagrange;
use bempp_element::reference_cell;
use bempp_traits::element::{Continuity, FiniteElement};
use bempp_traits::types::ReferenceCellType;
use log::warn;
use num::Float;
use rlst_dense::types::RlstScalar;
use rlst_dense::{
    array::{views::ArrayViewMut, Array},
    base_array::BaseArray,
    data_container::VectorContainer,
    traits::MatrixInverse,
};
use std::collections::HashMap;

/// A mixed grid
pub struct SerialMixedGrid<T: Float + RlstScalar<Real = T>> {
    topology: SerialMixedTopology,
    pub(crate) geometry: SerialMixedGeometry<T>,
}

impl<T: Float + RlstScalar<Real = T>> SerialMixedGrid<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    /// Create a mixed grid
    pub fn new(
        points: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        cells: &[usize],
        cell_types: &[ReferenceCellType],
        cell_degrees: &[usize],
        point_indices_to_ids: Vec<usize>,
        point_ids_to_indices: HashMap<usize, usize>,
        cell_indices_to_ids: Vec<usize>,
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
            .map(|(i, j)| {
                lagrange::create::<T>(*i, *j, Continuity::Continuous)
            })
            .collect::<Vec<_>>();

        if elements.len() == 1 {
            warn!("Creating a mixed grid with only one element. Using a SerialSingleElementGrid would be faster.");
        }

        let mut cell_vertices = vec![];

        let mut start = 0;
        for (cell_type, e_n) in cell_types.iter().zip(&element_numbers) {
            let nvertices = reference_cell::entity_counts(*cell_type)[0];
            let npoints = elements[*e_n].dim();
            cell_vertices.extend_from_slice(&cells[start..start + nvertices]);
            start += npoints;
        }

        let topology = SerialMixedTopology::new(
            &cell_vertices,
            cell_types,
            &point_indices_to_ids,
            &cell_indices_to_ids,
        );
        let geometry = SerialMixedGeometry::<T>::new(
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

impl<T: Float + RlstScalar<Real = T>> Grid for SerialMixedGrid<T> {
    type T = T;
    type Topology = SerialMixedTopology;
    type Geometry = SerialMixedGeometry<T>;

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
