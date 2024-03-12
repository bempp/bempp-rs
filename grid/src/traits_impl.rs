//! Implementing the grid traits from the topology and geometry traits used to store the grid data

use crate::traits::{Geometry, GeometryEvaluator, Grid, Topology};
use bempp_element::reference_cell;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{
    CellType, GeometryType, GridType, PointType, ReferenceMapType, TopologyType,
};
use bempp_traits::types::{CellLocalIndexPair, ReferenceCellType};
use num::Float;
use rlst_dense::types::RlstScalar;
use std::iter::Copied;
use std::marker::PhantomData;

/// A grid
pub struct WrappedGrid<GridImpl: Grid> {
    pub grid: GridImpl,
}

impl<GridImpl: Grid> Grid for WrappedGrid<GridImpl> {
    type T = GridImpl::T;
    type Topology = GridImpl::Topology;
    type Geometry = GridImpl::Geometry;

    fn topology(&self) -> &Self::Topology {
        self.grid.topology()
    }
    fn geometry(&self) -> &Self::Geometry {
        self.grid.geometry()
    }
    fn is_serial(&self) -> bool {
        self.grid.is_serial()
    }
}

/// A point
pub struct Point<'a, T: Float + RlstScalar<Real = T>, G: Geometry> {
    geometry: &'a G,
    index: usize,
    _t: PhantomData<T>,
}
/// A cell
pub struct Cell<'a, T: Float + RlstScalar<Real = T>, GridImpl: Grid> {
    grid: &'a WrappedGrid<GridImpl>,
    index: usize,
    _t: PhantomData<T>,
}
/// The topology of a cell
pub struct CellTopology<'a, GridImpl: Grid> {
    topology: &'a <GridImpl as Grid>::Topology,
    index: <<GridImpl as Grid>::Topology as Topology>::IndexType,
}
/// The geometry of a cell
pub struct CellGeometry<'a, T: Float + RlstScalar<Real = T>, GridImpl: Grid> {
    geometry: &'a <GridImpl as Grid>::Geometry,
    index: <<GridImpl as Grid>::Geometry as Geometry>::IndexType,
    _t: PhantomData<T>,
}
/// A reference to physical map
pub struct ReferenceMap<'a, GridImpl: Grid> {
    grid: &'a WrappedGrid<GridImpl>,
    evaluator: <<GridImpl as Grid>::Geometry as Geometry>::Evaluator<'a>,
}
/// An iterator over points
pub struct PointIterator<'a, GridImpl: Grid, Iter: std::iter::Iterator<Item = usize>> {
    iter: Iter,
    geometry: &'a <GridImpl as Grid>::Geometry,
}

impl<'a, GridImpl: Grid, Iter: std::iter::Iterator<Item = usize>> std::iter::Iterator
    for PointIterator<'a, GridImpl, Iter>
{
    type Item = Point<'a, <GridImpl as Grid>::T, <GridImpl as Grid>::Geometry>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.iter.next() {
            Some(Point {
                geometry: self.geometry,
                index,
                _t: PhantomData,
            })
        } else {
            None
        }
    }
}

impl<'a, T: Float + RlstScalar<Real = T>, G: Geometry<T = T>> PointType for Point<'a, T, G> {
    type T = T;
    fn coords(&self, data: &mut [Self::T]) {
        assert_eq!(data.len(), self.geometry.dim());
        for (dim, d) in data.iter_mut().enumerate() {
            *d = *self.geometry.coordinate(self.index, dim).unwrap();
        }
    }
    fn index(&self) -> usize {
        self.index
    }
    fn id(&self) -> usize {
        self.geometry.point_index_to_id(self.index)
    }
}

impl<'grid, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> CellType
    for Cell<'grid, T, GridImpl>
where
    GridImpl: 'grid,
{
    type Grid = WrappedGrid<GridImpl>;

    type Topology<'a> = CellTopology<'a, GridImpl> where Self: 'a;
    type Geometry<'a> = CellGeometry<'a, T, GridImpl> where Self: 'a;

    fn id(&self) -> usize {
        self.grid.geometry().point_index_to_id(self.index)
    }
    fn index(&self) -> usize {
        self.index
    }

    fn topology(&self) -> Self::Topology<'_> {
        CellTopology::<'_, GridImpl> {
            topology: self.grid.topology(),
            index: self.grid.topology().index_map()[self.index],
        }
    }

    fn grid(&self) -> &Self::Grid {
        self.grid
    }

    fn geometry(&self) -> Self::Geometry<'_> {
        CellGeometry::<'_, T, GridImpl> {
            geometry: self.grid.geometry(),
            index: self.grid.geometry().index_map()[self.index],
            _t: PhantomData,
        }
    }
}

impl<'grid, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> TopologyType
    for CellTopology<'grid, GridImpl>
where
    GridImpl: 'grid,
{
    type Grid = WrappedGrid<GridImpl>;
    type VertexIndexIter<'a> = Copied<std::slice::Iter<'a, usize>>
    where
        Self: 'a;
    type EdgeIndexIter<'a> = Self::VertexIndexIter<'a>
    where
        Self: 'a;
    type FaceIndexIter<'a> = Self::VertexIndexIter<'a>
    where
        Self: 'a;

    fn vertex_indices(&self) -> Self::VertexIndexIter<'_> {
        self.topology
            .cell_to_flat_entities(self.index, 0)
            .unwrap()
            .iter()
            .copied()
    }

    fn edge_indices(&self) -> Self::EdgeIndexIter<'_> {
        self.topology
            .cell_to_flat_entities(self.index, 1)
            .unwrap()
            .iter()
            .copied()
    }

    fn face_indices(&self) -> Self::FaceIndexIter<'_> {
        self.topology
            .cell_to_flat_entities(self.index, 2)
            .unwrap()
            .iter()
            .copied()
    }

    fn cell_type(&self) -> ReferenceCellType {
        self.topology.cell_type(self.index).unwrap()
    }
}

impl<'grid, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> GeometryType
    for CellGeometry<'grid, T, GridImpl>
where
    GridImpl: 'grid,
{
    type Grid = WrappedGrid<GridImpl>;

    type VertexIterator<'iter> =
        PointIterator<'iter, Self::Grid, Copied<std::slice::Iter<'iter, usize>>> where Self: 'iter;

    type PointIterator<'iter> = Self::VertexIterator<'iter> where Self: 'iter;

    fn physical_dimension(&self) -> usize {
        self.geometry.dim()
    }

    fn midpoint(&self, point: &mut [T]) {
        self.geometry.midpoint(self.index, point)
    }

    fn diameter(&self) -> T {
        self.geometry.diameter(self.index)
    }

    fn volume(&self) -> T {
        self.geometry.volume(self.index)
    }

    fn points(&self) -> Self::PointIterator<'_> {
        PointIterator {
            iter: self
                .geometry
                .cell_points(self.index)
                .unwrap()
                .iter()
                .copied(),
            geometry: self.geometry,
        }
    }
    fn vertices(&self) -> Self::VertexIterator<'_> {
        let cell_type = self.geometry.cell_element(self.index).unwrap().cell_type();
        let nvertices = reference_cell::entity_counts(cell_type)[0];
        PointIterator {
            iter: self.geometry.cell_points(self.index).unwrap()[..nvertices]
                .iter()
                .copied(),
            geometry: self.geometry,
        }
    }
}

impl<'a, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> ReferenceMapType
    for ReferenceMap<'a, GridImpl>
{
    type Grid = WrappedGrid<GridImpl>;

    fn domain_dimension(&self) -> usize {
        self.grid.topology().dim()
    }

    fn physical_dimension(&self) -> usize {
        self.grid.geometry().dim()
    }

    fn number_of_reference_points(&self) -> usize {
        self.evaluator.point_count()
    }

    fn reference_to_physical(
        &self,
        cell_index: usize,
        point_index: usize,
        value: &mut [<Self::Grid as GridType>::T],
    ) {
        self.evaluator.compute_point(cell_index, point_index, value);
    }

    fn jacobian(&self, cell_index: usize, point_index: usize, value: &mut [T]) {
        self.evaluator
            .compute_jacobian(cell_index, point_index, value);
    }

    fn normal(&self, cell_index: usize, point_index: usize, value: &mut [T]) {
        self.evaluator
            .compute_normal(cell_index, point_index, value);
    }
}

impl<'grid, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> GridType
    for WrappedGrid<GridImpl>
where
    GridImpl: 'grid,
{
    type T = T;

    type ReferenceMap<'a> = ReferenceMap<'a, GridImpl>
    where
        Self: 'a;

    type Point<'a> = Point<'a, T, GridImpl::Geometry> where Self: 'a;
    type Cell<'a> = Cell<'a, T, GridImpl> where Self: 'a;

    fn number_of_points(&self) -> usize {
        self.geometry().point_count()
    }
    fn number_of_edges(&self) -> usize {
        self.topology().entity_count(ReferenceCellType::Interval)
    }
    fn number_of_vertices(&self) -> usize {
        self.topology().entity_count(ReferenceCellType::Point)
    }
    fn number_of_cells(&self) -> usize {
        self.geometry().cell_count()
    }

    fn point_index_from_id(&self, id: usize) -> usize {
        id
    }
    fn point_id_from_index(&self, index: usize) -> usize {
        index
    }

    fn cell_index_from_id(&self, id: usize) -> usize {
        id
    }
    fn cell_id_from_index(&self, index: usize) -> usize {
        index
    }

    fn point_from_index(&self, index: usize) -> Self::Point<'_> {
        Self::Point {
            geometry: self.geometry(),
            index,
            _t: PhantomData,
        }
    }

    fn cell_from_index(&self, index: usize) -> Self::Cell<'_> {
        Self::Cell {
            grid: self,
            index,
            _t: PhantomData,
        }
    }

    fn reference_to_physical_map<'a>(
        &'a self,
        reference_points: &'a [Self::T],
    ) -> Self::ReferenceMap<'a> {
        Self::ReferenceMap {
            grid: self,
            evaluator: self.geometry().get_evaluator(reference_points),
        }
    }

    fn vertex_to_cells(&self, vertex_index: usize) -> &[CellLocalIndexPair<usize>] {
        self.topology()
            .entity_to_flat_cells(0, self.topology().vertex_flat_index_to_index(vertex_index))
            .unwrap()
    }

    fn edge_to_cells(&self, edge_index: usize) -> &[CellLocalIndexPair<usize>] {
        self.topology()
            .entity_to_flat_cells(1, self.topology().edge_flat_index_to_index(edge_index))
            .unwrap()
    }

    fn face_to_cells(&self, face_index: usize) -> &[CellLocalIndexPair<usize>] {
        self.topology()
            .entity_to_flat_cells(2, self.topology().face_flat_index_to_index(face_index))
            .unwrap()
    }

    fn is_serial(&self) -> bool {
        Grid::is_serial(self)
    }

    fn domain_dimension(&self) -> usize {
        self.grid.topology().dim()
    }

    fn physical_dimension(&self) -> usize {
        self.grid.geometry().dim()
    }
}
