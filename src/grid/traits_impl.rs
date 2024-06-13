// //! Implementing the grid traits from the topology and geometry traits used to store the grid data

// use crate::element::reference_cell;
// // use crate::grid::traits::{Geometry, GeometryEvaluator, Grid, Topology};
// use crate::traits::element::FiniteElement;
// use crate::traits::grid::{
//     CellType, EdgeType, GeometryType, Grid, PointType, ReferenceMapType, TopologyType,
// };
// use crate::traits::types::{CellLocalIndexPair, Ownership, ReferenceCellType};
// #[cfg(feature = "mpi")]
// use crate::{
//     grid::parallel_grid::{LocalGrid, ParallelGrid},
//     traits::grid::ParallelGridType,
// };
// #[cfg(feature = "mpi")]
// use mpi::traits::Communicator;
// use num::Float;
// use rlst::RlstScalar;
// use std::iter::Copied;
// use std::marker::PhantomData;

// /// A point
// pub struct Point<'a, T: Float + RlstScalar<Real = T>, G: Geometry> {
//     geometry: &'a G,
//     index: usize,
//     _t: PhantomData<T>,
// }
// /// A vertex
// pub struct Vertex<'a, T: Float + RlstScalar<Real = T>, G: Geometry, Top: Topology> {
//     geometry: &'a G,
//     topology: &'a Top,
//     index: usize,
//     gindex: usize,
//     tindex: usize,
//     _t: PhantomData<T>,
// }
// /// An edge
// pub struct Edge<'a, Top: Topology> {
//     topology: &'a Top,
//     index: usize,
// }
// /// A cell
// pub struct Cell<'a, T: Float + RlstScalar<Real = T>, GridImpl: Grid> {
//     grid: &'a GridImpl,
//     index: usize,
//     _t: PhantomData<T>,
// }
// /// The topology of a cell
// pub struct CellTopology<'a, GridImpl: Grid> {
//     topology: &'a <GridImpl as Grid>::Topology,
//     index: <<GridImpl as Grid>::Topology as Topology>::IndexType,
//     face_indices: Vec<usize>,
// }
// /// The geometry of a cell
// pub struct CellGeometry<'a, T: Float + RlstScalar<Real = T>, GridImpl: Grid> {
//     geometry: &'a <GridImpl as Grid>::Geometry,
//     index: <<GridImpl as Grid>::Geometry as Geometry>::IndexType,
//     _t: PhantomData<T>,
// }
// /// A reference to physical map
// pub struct ReferenceMap<'a, GridImpl: Grid> {
//     grid: &'a GridImpl,
//     evaluator: <<GridImpl as Grid>::Geometry as Geometry>::Evaluator<'a>,
// }
// /// An iterator over points
// pub struct PointIterator<'a, GridImpl: Grid, Iter: std::iter::Iterator<Item = usize>> {
//     iter: Iter,
//     geometry: &'a <GridImpl as Grid>::Geometry,
// }

// impl<'a, GridImpl: Grid, Iter: std::iter::Iterator<Item = usize>> std::iter::Iterator
//     for PointIterator<'a, GridImpl, Iter>
// {
//     type Item = Point<'a, <GridImpl as Grid>::T, <GridImpl as Grid>::Geometry>;

//     fn next(&mut self) -> Option<Self::Item> {
//         if let Some(index) = self.iter.next() {
//             Some(Point {
//                 geometry: self.geometry,
//                 index,
//                 _t: PhantomData,
//             })
//         } else {
//             None
//         }
//     }
// }

// impl<'a, T: Float + RlstScalar<Real = T>, G: Geometry<T = T>> PointType for Point<'a, T, G> {
//     type T = T;
//     fn coords(&self, data: &mut [Self::T]) {
//         assert_eq!(data.len(), self.geometry.dim());
//         for (dim, d) in data.iter_mut().enumerate() {
//             *d = *self.geometry.coordinate(self.index, dim).unwrap();
//         }
//     }
//     fn index(&self) -> usize {
//         self.index
//     }
//     fn id(&self) -> usize {
//         self.geometry.point_index_to_id(self.index)
//     }
//     fn ownership(&self) -> Ownership {
//         // TODO
//         Ownership::Owned
//     }
// }

// impl<'a, T: Float + RlstScalar<Real = T>, G: Geometry<T = T>, Top: Topology> PointType
//     for Vertex<'a, T, G, Top>
// {
//     type T = T;
//     fn coords(&self, data: &mut [Self::T]) {
//         assert_eq!(data.len(), self.geometry.dim());
//         for (dim, d) in data.iter_mut().enumerate() {
//             *d = *self.geometry.coordinate(self.gindex, dim).unwrap();
//         }
//     }
//     fn index(&self) -> usize {
//         self.index
//     }
//     fn id(&self) -> usize {
//         self.topology.vertex_index_to_id(self.tindex)
//     }
//     fn ownership(&self) -> Ownership {
//         self.topology.vertex_ownership(self.tindex)
//     }
// }

// impl<'a, Top: Topology> EdgeType for Edge<'a, Top> {
//     fn index(&self) -> usize {
//         self.index
//     }
//     fn ownership(&self) -> Ownership {
//         self.topology.edge_ownership(self.index)
//     }
// }

// impl<'grid, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> CellType
//     for Cell<'grid, T, GridImpl>
// where
//     GridImpl: 'grid,
// {
//     type Grid = GridImpl;

//     type Topology<'a> = CellTopology<'a, GridImpl> where Self: 'a;
//     type Geometry<'a> = CellGeometry<'a, T, GridImpl> where Self: 'a;

//     fn id(&self) -> usize {
//         self.grid.geometry().point_index_to_id(self.index)
//     }
//     fn index(&self) -> usize {
//         self.index
//     }

//     fn topology(&self) -> Self::Topology<'_> {
//         CellTopology::<'_, GridImpl> {
//             topology: self.grid.topology(),
//             index: self.grid.topology().index_map()[self.index],
//             face_indices: vec![self.index],
//         }
//     }

//     fn grid(&self) -> &Self::Grid {
//         self.grid
//     }

//     fn geometry(&self) -> Self::Geometry<'_> {
//         CellGeometry::<'_, T, GridImpl> {
//             geometry: self.grid.geometry(),
//             index: self.grid.geometry().index_map()[self.index],
//             _t: PhantomData,
//         }
//     }

//     fn ownership(&self) -> Ownership {
//         self.grid
//             .topology()
//             .cell_ownership(self.grid.topology().index_map()[self.index])
//     }
// }

// impl<'grid, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> TopologyType
//     for CellTopology<'grid, GridImpl>
// where
//     GridImpl: 'grid,
// {
//     type Grid = GridImpl;
//     type VertexIndexIter<'a> = Copied<std::slice::Iter<'a, usize>>
//     where
//         Self: 'a;
//     type EdgeIndexIter<'a> = Self::VertexIndexIter<'a>
//     where
//         Self: 'a;
//     type FaceIndexIter<'a> = Self::VertexIndexIter<'a>
//     where
//         Self: 'a;

//     fn vertex_indices(&self) -> Self::VertexIndexIter<'_> {
//         self.topology
//             .cell_to_entities(self.index, 0)
//             .unwrap()
//             .iter()
//             .copied()
//     }

//     fn edge_indices(&self) -> Self::EdgeIndexIter<'_> {
//         self.topology
//             .cell_to_entities(self.index, 1)
//             .unwrap()
//             .iter()
//             .copied()
//     }

//     fn face_indices(&self) -> Self::FaceIndexIter<'_> {
//         self.face_indices.iter().copied()
//     }

//     fn cell_type(&self) -> ReferenceCellType {
//         self.topology.cell_type(self.index).unwrap()
//     }
// }

// impl<'grid, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> GeometryType
//     for CellGeometry<'grid, T, GridImpl>
// where
//     GridImpl: 'grid,
// {
//     type Grid = GridImpl;

//     type VertexIterator<'iter> =
//         PointIterator<'iter, Self::Grid, Copied<std::slice::Iter<'iter, usize>>> where Self: 'iter;

//     type PointIterator<'iter> = Self::VertexIterator<'iter> where Self: 'iter;

//     fn physical_dimension(&self) -> usize {
//         self.geometry.dim()
//     }

//     fn midpoint(&self, point: &mut [T]) {
//         self.geometry.midpoint(self.index, point)
//     }

//     fn diameter(&self) -> T {
//         self.geometry.diameter(self.index)
//     }

//     fn volume(&self) -> T {
//         self.geometry.volume(self.index)
//     }

//     fn points(&self) -> Self::PointIterator<'_> {
//         PointIterator {
//             iter: self
//                 .geometry
//                 .cell_points(self.index)
//                 .unwrap()
//                 .iter()
//                 .copied(),
//             geometry: self.geometry,
//         }
//     }
//     fn vertices(&self) -> Self::VertexIterator<'_> {
//         let cell_type = self.geometry.cell_element(self.index).unwrap().cell_type();
//         let nvertices = reference_cell::entity_counts(cell_type)[0];
//         PointIterator {
//             iter: self.geometry.cell_points(self.index).unwrap()[..nvertices]
//                 .iter()
//                 .copied(),
//             geometry: self.geometry,
//         }
//     }
// }

// impl<'a, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> ReferenceMapType
//     for ReferenceMap<'a, GridImpl>
// {
//     type Grid = GridImpl;

//     fn domain_dimension(&self) -> usize {
//         self.grid.topology().dim()
//     }

//     fn physical_dimension(&self) -> usize {
//         self.grid.geometry().dim()
//     }

//     fn number_of_reference_points(&self) -> usize {
//         self.evaluator.point_count()
//     }

//     fn reference_to_physical(&self, cell_index: usize, value: &mut [<Self::Grid as Grid>::T]) {
//         self.evaluator.compute_points(cell_index, value);
//     }

//     fn jacobian(&self, cell_index: usize, value: &mut [T]) {
//         self.evaluator.compute_jacobians(cell_index, value);
//     }

//     fn normal(&self, cell_index: usize, value: &mut [T]) {
//         self.evaluator.compute_normals(cell_index, value);
//     }
// }

// impl<'grid, T: Float + RlstScalar<Real = T>, GridImpl: Grid<T = T>> Grid for GridImpl
// where
//     GridImpl: 'grid,
// {
//     type T = T;

//     type ReferenceMap<'a> = ReferenceMap<'a, GridImpl>
//     where
//         Self: 'a;

//     type Point<'a> = Point<'a, T, GridImpl::Geometry> where Self: 'a;
//     type Vertex<'a> = Vertex<'a, T, GridImpl::Geometry, GridImpl::Topology> where Self: 'a;
//     type Edge<'a> = Edge<'a, GridImpl::Topology> where Self: 'a;
//     type Cell<'a> = Cell<'a, T, GridImpl> where Self: 'a;

//     fn number_of_points(&self) -> usize {
//         self.geometry().point_count()
//     }
//     fn number_of_edges(&self) -> usize {
//         self.topology().entity_count(ReferenceCellType::Interval)
//     }
//     fn number_of_vertices(&self) -> usize {
//         self.topology().entity_count(ReferenceCellType::Point)
//     }
//     fn number_of_cells(&self) -> usize {
//         self.geometry().cell_count()
//     }

//     fn point_index_from_id(&self, id: usize) -> usize {
//         id
//     }
//     fn point_id_from_index(&self, index: usize) -> usize {
//         index
//     }

//     fn vertex_index_from_id(&self, id: usize) -> usize {
//         self.topology().vertex_id_to_index(id)
//     }
//     fn vertex_id_from_index(&self, index: usize) -> usize {
//         self.topology().vertex_index_to_id(index)
//     }

//     fn cell_index_from_id(&self, id: usize) -> usize {
//         id
//     }
//     fn cell_id_from_index(&self, index: usize) -> usize {
//         index
//     }

//     fn point_from_index(&self, index: usize) -> Self::Point<'_> {
//         Self::Point {
//             geometry: self.geometry(),
//             index,
//             _t: PhantomData,
//         }
//     }

//     fn vertex_from_index(&self, index: usize) -> Self::Vertex<'_> {
//         Self::Vertex {
//             geometry: self.geometry(),
//             topology: self.topology(),
//             index,
//             gindex: self.point_index_from_id(self.vertex_id_from_index(index)),
//             tindex: index,
//             _t: PhantomData,
//         }
//     }

//     fn edge_from_index(&self, index: usize) -> Self::Edge<'_> {
//         Self::Edge {
//             topology: self.topology(),
//             index,
//         }
//     }

//     fn cell_from_index(&self, index: usize) -> Self::Cell<'_> {
//         Self::Cell {
//             grid: self,
//             index,
//             _t: PhantomData,
//         }
//     }

//     fn reference_to_physical_map<'a>(
//         &'a self,
//         reference_points: &'a [Self::T],
//     ) -> Self::ReferenceMap<'a> {
//         Self::ReferenceMap {
//             grid: self,
//             evaluator: self.geometry().get_evaluator(reference_points),
//         }
//     }

//     fn vertex_to_cells(&self, vertex_index: usize) -> &[CellLocalIndexPair<usize>] {
//         self.topology()
//             .entity_to_flat_cells(0, vertex_index)
//             .unwrap()
//     }

//     fn edge_to_cells(&self, edge_index: usize) -> &[CellLocalIndexPair<usize>] {
//         self.topology().entity_to_flat_cells(1, edge_index).unwrap()
//     }

//     fn face_to_cells(&self, _face_index: usize) -> &[CellLocalIndexPair<usize>] {
//         unimplemented!();
//     }

//     fn is_serial(&self) -> bool {
//         Grid::is_serial(self)
//     }

//     fn domain_dimension(&self) -> usize {
//         self.topology().dim()
//     }

//     fn physical_dimension(&self) -> usize {
//         self.geometry().dim()
//     }

//     fn cell_types(&self) -> &[ReferenceCellType] {
//         self.topology().cell_types()
//     }
// }

// #[cfg(feature = "mpi")]
// impl<'comm, T: RlstScalar<Real = T> + Float, C: Communicator, G: Grid<T = T> + Grid<T = T>>
//     ParallelGridType for ParallelGrid<'comm, C, G>
// {
//     type Comm = C;
//     type LocalGridType = LocalGrid<G>;

//     fn comm(&self) -> &C {
//         self.comm
//     }

//     fn local_grid(&self) -> &LocalGrid<G> {
//         &self.local_grid
//     }
// }
