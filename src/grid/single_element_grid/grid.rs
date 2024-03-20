//! Single element grid

use crate::element::element::lagrange;
use crate::element::reference_cell;
use crate::grid::single_element_grid::{
    geometry::SerialSingleElementGeometry, topology::SerialSingleElementTopology,
};
use crate::grid::traits::Grid;
use crate::traits::element::{Continuity, FiniteElement};
use crate::traits::types::ReferenceCellType;
use log::warn;
use num::Float;
use rlst::RlstScalar;
use rlst::{
    dense::array::{views::ArrayViewMut, Array},
    BaseArray, MatrixInverse, VectorContainer,
};
use std::collections::HashMap;

/// A single element grid
pub struct SerialSingleElementGrid<T: Float + RlstScalar<Real = T>> {
    topology: SerialSingleElementTopology,
    pub(crate) geometry: SerialSingleElementGeometry<T>,
}

impl<T: Float + RlstScalar<Real = T>> SerialSingleElementGrid<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    /// Create a single element grid
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        points: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        cells: &[usize],
        cell_type: ReferenceCellType,
        cell_degree: usize,
        point_indices_to_ids: Vec<usize>,
        point_ids_to_indices: HashMap<usize, usize>,
        cell_indices_to_ids: Vec<usize>,
        cell_ids_to_indices: HashMap<usize, usize>,
    ) -> Self {
        if cell_type == ReferenceCellType::Triangle && cell_degree == 1 {
            warn!("Creating a single element grid with a P1 triangle. Using a FlatTriangleGrid would be faster.");
        }
        let element = lagrange::create::<T>(cell_type, cell_degree, Continuity::Continuous);

        let mut cell_vertices = vec![];

        let mut start = 0;
        let nvertices = reference_cell::entity_counts(cell_type)[0];
        let npoints = element.dim();
        while start < cells.len() {
            cell_vertices.extend_from_slice(&cells[start..start + nvertices]);
            start += npoints;
        }

        let topology = SerialSingleElementTopology::new(
            &cell_vertices,
            cell_type,
            &point_indices_to_ids,
            &cell_indices_to_ids,
        );
        let geometry = SerialSingleElementGeometry::<T>::new(
            points,
            cells,
            element,
            point_indices_to_ids,
            point_ids_to_indices,
            cell_indices_to_ids,
            cell_ids_to_indices,
        );
        Self { topology, geometry }
    }
}

impl<T: Float + RlstScalar<Real = T>> Grid for SerialSingleElementGrid<T> {
    type T = T;
    type Topology = SerialSingleElementTopology;
    type Geometry = SerialSingleElementGeometry<T>;

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
