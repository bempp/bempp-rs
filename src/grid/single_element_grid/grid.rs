//! Single element grid

use crate::element::ciarlet::lagrange;
use crate::element::reference_cell;
use crate::grid::single_element_grid::{
    geometry::SingleElementGeometry, topology::SingleElementTopology,
};
use crate::grid::traits::Grid;
use crate::traits::element::{Continuity, FiniteElement};
use crate::traits::types::{Ownership, ReferenceCellType};
use log::warn;
use num::Float;
use rlst::RlstScalar;
use rlst::{
    dense::array::{views::ArrayViewMut, Array},
    BaseArray, MatrixInverse, VectorContainer,
};
use std::collections::HashMap;

/// A single element grid
pub struct SingleElementGrid<T: Float + RlstScalar<Real = T>> {
    topology: SingleElementTopology,
    pub(crate) geometry: SingleElementGeometry<T>,
}

impl<T: Float + RlstScalar<Real = T>> SingleElementGrid<T>
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
        cell_ownership: Option<HashMap<usize, Ownership>>,
        vertex_ownership: Option<HashMap<usize, Ownership>>,
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

        let topology = SingleElementTopology::new(
            &cell_vertices,
            cell_type,
            &point_indices_to_ids,
            &cell_indices_to_ids,
            cell_ownership,
            vertex_ownership,
        );
        let geometry = SingleElementGeometry::<T>::new(
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

impl<T: Float + RlstScalar<Real = T>> Grid for SingleElementGrid<T> {
    type T = T;
    type Topology = SingleElementTopology;
    type Geometry = SingleElementGeometry<T>;

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
