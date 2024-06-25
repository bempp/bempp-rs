//! Implementation of grid topology

use ndelement::reference_cell;
use crate::grid::traits::Topology;
use crate::traits::types::{CellLocalIndexPair, Ownership, ReferenceCellType};

use std::collections::HashMap;

fn all_equal<T: Eq>(a: &[T], b: &[T]) -> bool {
    if a.len() != b.len() {
        false
    } else {
        all_in(a, b)
    }
}

fn all_in<T: Eq>(a: &[T], b: &[T]) -> bool {
    for i in a {
        if !b.contains(i) {
            return false;
        }
    }
    true
}

type IndexType = (ReferenceCellType, usize);

/// Topology of a mixed grid
pub struct MixedTopology {
    dim: usize,
    index_map: Vec<IndexType>,
    reverse_index_map: HashMap<IndexType, usize>,
    entities_to_vertices: Vec<Vec<Vec<usize>>>,
    cells_to_entities: Vec<HashMap<ReferenceCellType, Vec<Vec<usize>>>>,
    entities_to_cells: Vec<Vec<Vec<CellLocalIndexPair<IndexType>>>>,
    entities_to_flat_cells: Vec<Vec<Vec<CellLocalIndexPair<usize>>>>,
    entity_types: Vec<Vec<ReferenceCellType>>,
    vertex_indices_to_ids: Vec<usize>,
    vertex_ids_to_indices: HashMap<usize, usize>,
    edge_indices_to_ids: Vec<usize>,
    edge_ids_to_indices: HashMap<usize, usize>,
    cell_indices_to_ids: HashMap<IndexType, usize>,
    cell_ids_to_indices: HashMap<usize, IndexType>,
}

unsafe impl Sync for MixedTopology {}

impl MixedTopology {
    /// Create a topology
    pub fn new(
        cells_input: &[usize],
        cell_types: &[ReferenceCellType],
        point_indices_to_ids: &[usize],
        grid_cell_indices_to_ids: &[usize],
        edge_ids: Option<HashMap<[usize; 2], usize>>,
    ) -> Self {
        let mut index_map = vec![(ReferenceCellType::Point, 0); cell_types.len()];
        let mut reverse_index_map = HashMap::new();
        let mut vertices = vec![];
        let dim = reference_cell::dim(cell_types[0]);

        let mut vertex_indices_to_ids = vec![];
        let mut vertex_ids_to_indices = HashMap::new();
        let mut edge_indices_to_ids = vec![];
        let mut edge_ids_to_indices = HashMap::new();
        let mut cell_indices_to_ids = HashMap::new();
        let mut cell_ids_to_indices = HashMap::new();

        let mut entity_types = vec![vec![]; 4];

        let mut entities_to_vertices = vec![vec![]; dim];
        let mut cells_to_entities = vec![HashMap::new(); dim + 1];
        let mut entities_to_cells = vec![vec![]; dim];
        let mut entities_to_flat_cells = vec![vec![]; dim + 1];

        for c in cell_types {
            if dim != reference_cell::dim(*c) {
                panic!("Grids with cells of mixed topological dimension not supported.");
            }
            for (dim0, etypes) in reference_cell::entity_types(*c).iter().enumerate() {
                for e in etypes {
                    if !entity_types[dim0].contains(e) {
                        entity_types[dim0].push(*e);

                        if dim0 == dim {
                            for ce in cells_to_entities.iter_mut() {
                                ce.insert(*e, vec![]);
                            }
                        }
                    }
                }
            }
        }

        // dim0 = dim, dim1 = 0
        for c in &entity_types[dim] {
            let n = reference_cell::entity_counts(*c)[0];
            let mut start = 0;
            for (i, ct) in cell_types.iter().enumerate() {
                if *ct == *c {
                    let cell = &cells_input[start..start + n];
                    let cell_i = (*c, cells_to_entities[0][c].len());
                    index_map[i] = cell_i;
                    reverse_index_map.insert(cell_i, i);

                    cell_indices_to_ids.insert(cell_i, grid_cell_indices_to_ids[i]);
                    cell_ids_to_indices.insert(grid_cell_indices_to_ids[i], cell_i);

                    let mut row = vec![];
                    for v in cell {
                        if !vertices.contains(v) {
                            entities_to_cells[0].push(vec![]);
                            entities_to_flat_cells[0].push(vec![]);
                            vertex_indices_to_ids.push(point_indices_to_ids[*v]);
                            vertex_ids_to_indices.insert(point_indices_to_ids[*v], vertices.len());
                            vertices.push(*v);
                        }
                        row.push(vertices.iter().position(|&r| r == *v).unwrap());
                    }

                    for (local_index, v) in row.iter().enumerate() {
                        entities_to_cells[0][*v].push(CellLocalIndexPair::new(cell_i, local_index));
                        entities_to_flat_cells[0][*v].push(CellLocalIndexPair::new(
                            reverse_index_map[&cell_i],
                            local_index,
                        ));
                    }

                    cells_to_entities[0].get_mut(c).unwrap().push(row);
                }
                start += reference_cell::entity_counts(*ct)[0];
            }
        }
        for i in 0..vertices.len() {
            entities_to_vertices[0].push(vec![i]);
        }

        let mut edge_indices = HashMap::new();
        if let Some(e) = &edge_ids {
            for (edge_i, (i, j)) in e.iter().enumerate() {
                let mut v0 = vertex_ids_to_indices[&i[0]];
                let mut v1 = vertex_ids_to_indices[&i[1]];
                if v0 > v1 {
                    std::mem::swap(&mut v0, &mut v1);
                }
                edge_indices.insert((v0, v1), edge_i);
                edge_indices_to_ids.push(*j);
                edge_ids_to_indices.insert(*j, edge_i);
                entities_to_vertices[1].push(vec![v0, v1]);
                entities_to_cells[1].push(vec![]);
            }
        }
        for cell_type in &entity_types[dim] {
            let ref_conn = &reference_cell::connectivity(*cell_type)[1];
            let ncells = cells_to_entities[0][cell_type].len();
            for cell_i in 0..ncells {
                cells_to_entities[1]
                    .get_mut(cell_type)
                    .unwrap()
                    .push(vec![]);
                let cell_index = (*cell_type, cell_i);
                for (local_index, rc) in ref_conn.iter().enumerate() {
                    let cell = &cells_to_entities[0][cell_type][cell_i];
                    let mut first = cell[rc[0][0]];
                    let mut second = cell[rc[0][1]];
                    if first > second {
                        std::mem::swap(&mut first, &mut second);
                    }
                    if let Some(edge_index) = edge_indices.get(&(first, second)) {
                        cells_to_entities[1].get_mut(cell_type).unwrap()[cell_i].push(*edge_index);
                        entities_to_cells[1][*edge_index]
                            .push(CellLocalIndexPair::new(cell_index, local_index));
                    } else {
                        if edge_ids.is_some() {
                            panic!("Missing id for edge");
                        }
                        let id = entities_to_vertices[1].len();
                        edge_indices.insert((first, second), id);
                        edge_indices_to_ids.push(id);
                        edge_ids_to_indices.insert(id, id);
                        cells_to_entities[1].get_mut(cell_type).unwrap()[cell_i]
                            .push(entities_to_vertices[1].len());
                        entities_to_cells[1]
                            .push(vec![CellLocalIndexPair::new(cell_index, local_index)]);
                        entities_to_vertices[1].push(vec![first, second]);
                    }
                }
            }
        }

        for d in 2..dim {
            for cell_type in &entity_types[dim] {
                let mut c_to_e = vec![];
                let mut c_to_e_flat = vec![];
                let ref_conn = &reference_cell::connectivity(*cell_type)[d];
                for (cell_i, cell) in cells_to_entities[0][cell_type].iter().enumerate() {
                    let mut entity_ids = vec![];
                    let mut entity_ids_flat = vec![];

                    for (local_index, rc) in ref_conn.iter().enumerate() {
                        let vertices = rc[0].iter().map(|x| cell[*x]).collect::<Vec<_>>();
                        let mut found = false;
                        for (entity_index, entity) in entities_to_vertices[d].iter().enumerate() {
                            if all_equal(entity, &vertices) {
                                entity_ids.push(entity_index);
                                entity_ids_flat.push(entity_index);
                                entities_to_cells[d][entity_index].push(CellLocalIndexPair::new(
                                    (*cell_type, cell_i),
                                    local_index,
                                ));

                                found = true;
                                break;
                            }
                        }
                        if !found {
                            entity_ids.push(entities_to_vertices[d].len());
                            entity_ids_flat.push(entities_to_vertices[d].len());
                            entities_to_cells[d].push(vec![CellLocalIndexPair::new(
                                (*cell_type, cell_i),
                                local_index,
                            )]);
                            entities_to_vertices[d].push(vertices);
                        }
                    }
                    c_to_e.push(entity_ids);
                    c_to_e_flat.push(entity_ids_flat);
                }
                *cells_to_entities[d].get_mut(cell_type).unwrap() = c_to_e;
            }
        }

        Self {
            dim,
            index_map,
            reverse_index_map,
            entities_to_vertices,
            cells_to_entities,
            entities_to_cells,
            entities_to_flat_cells,
            entity_types,
            vertex_indices_to_ids,
            vertex_ids_to_indices,
            edge_indices_to_ids,
            edge_ids_to_indices,
            cell_indices_to_ids,
            cell_ids_to_indices,
        }
    }
}

impl Topology for MixedTopology {
    type IndexType = IndexType;

    fn dim(&self) -> usize {
        self.dim
    }
    fn index_map(&self) -> &[IndexType] {
        &self.index_map
    }
    fn entity_count(&self, etype: ReferenceCellType) -> usize {
        let dim = reference_cell::dim(etype);
        if dim == 2 {
            if self.cells_to_entities[0].contains_key(&etype) {
                self.cells_to_entities[0][&etype].len()
            } else {
                0
            }
        } else {
            self.entities_to_cells[dim].len()
        }
    }
    fn entity_count_by_dim(&self, dim: usize) -> usize {
        self.entity_types[dim]
            .iter()
            .map(|e| self.entity_count(*e))
            .sum()
    }
    fn cell(&self, index: IndexType) -> Option<&[usize]> {
        if self.cells_to_entities[0].contains_key(&index.0)
            && index.1 < self.cells_to_entities[0][&index.0].len()
        {
            Some(&self.cells_to_entities[0][&index.0][index.1])
        } else {
            None
        }
    }
    fn cell_type(&self, index: IndexType) -> Option<ReferenceCellType> {
        if self.cells_to_entities[0].contains_key(&index.0)
            && index.1 < self.cells_to_entities[0][&index.0].len()
        {
            Some(index.0)
        } else {
            None
        }
    }

    fn entity_types(&self, dim: usize) -> &[ReferenceCellType] {
        &self.entity_types[dim]
    }

    fn cell_ownership(&self, _index: (ReferenceCellType, usize)) -> Ownership {
        Ownership::Owned
    }
    fn vertex_ownership(&self, _index: usize) -> Ownership {
        Ownership::Owned
    }
    fn edge_ownership(&self, _index: usize) -> Ownership {
        Ownership::Owned
    }

    fn entity_to_cells(
        &self,
        dim: usize,
        index: usize,
    ) -> Option<&[CellLocalIndexPair<IndexType>]> {
        if dim < self.dim && index < self.entities_to_cells[dim].len() {
            Some(&self.entities_to_cells[dim][index])
        } else {
            None
        }
    }

    fn entity_to_flat_cells(
        &self,
        dim: usize,
        index: usize,
    ) -> Option<&[CellLocalIndexPair<usize>]> {
        if dim < self.dim && index < self.entities_to_flat_cells[dim].len() {
            Some(&self.entities_to_flat_cells[dim][index])
        } else {
            None
        }
    }

    fn cell_to_entities(&self, index: IndexType, dim: usize) -> Option<&[usize]> {
        if dim < self.dim
            && self.cells_to_entities[dim].contains_key(&index.0)
            && index.1 < self.cells_to_entities[dim][&index.0].len()
        {
            Some(&self.cells_to_entities[dim][&index.0][index.1])
        } else {
            None
        }
    }

    fn entity_vertices(&self, dim: usize, index: usize) -> Option<&[usize]> {
        if index < self.entities_to_vertices[dim].len() {
            Some(&self.entities_to_vertices[dim][index])
        } else {
            None
        }
    }

    fn vertex_index_to_id(&self, index: usize) -> usize {
        self.vertex_indices_to_ids[index]
    }
    fn cell_index_to_id(&self, index: IndexType) -> usize {
        self.cell_indices_to_ids[&index]
    }
    fn vertex_id_to_index(&self, id: usize) -> usize {
        if self.vertex_ids_to_indices.contains_key(&id) {
            self.vertex_ids_to_indices[&id]
        } else {
            panic!("Vertex with id {} not found", id);
        }
    }
    fn edge_id_to_index(&self, id: usize) -> usize {
        self.edge_ids_to_indices[&id]
    }
    fn edge_index_to_id(&self, index: usize) -> usize {
        self.edge_indices_to_ids[index]
    }
    fn cell_id_to_index(&self, id: usize) -> IndexType {
        self.cell_ids_to_indices[&id]
    }
    fn face_index_to_flat_index(&self, index: IndexType) -> usize {
        self.reverse_index_map[&index]
    }
    fn face_flat_index_to_index(&self, index: usize) -> IndexType {
        self.index_map[index]
    }
    fn cell_types(&self) -> &[ReferenceCellType] {
        &self.entity_types[self.dim]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn example_topology() -> MixedTopology {
        //! A topology with a single cell type
        MixedTopology::new(
            &[0, 1, 2, 2, 1, 3],
            &[ReferenceCellType::Triangle; 2],
            &[0, 1, 2, 3],
            &[0, 1],
            None,
        )
    }

    fn example_topology_mixed() -> MixedTopology {
        //! A topology with a mixture of cell types
        MixedTopology::new(
            &[0, 1, 2, 3, 1, 4, 3],
            &[
                ReferenceCellType::Quadrilateral,
                ReferenceCellType::Triangle,
            ],
            &[0, 1, 2, 3, 4],
            &[0, 1],
            None,
        )
    }

    #[test]
    fn test_counts() {
        //! Test entity counts
        let t = example_topology();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.entity_count(ReferenceCellType::Point), 4);
        assert_eq!(t.entity_count(ReferenceCellType::Interval), 5);
        assert_eq!(t.entity_count(ReferenceCellType::Triangle), 2);
    }

    #[test]
    fn test_cell_to_entities_vertices() {
        //! Test cell vertices
        let t = example_topology();
        for (i, vertices) in [[0, 1, 2], [2, 1, 3]].iter().enumerate() {
            let c = t
                .cell_to_entities((ReferenceCellType::Triangle, i), 0)
                .unwrap();
            assert_eq!(c.len(), 3);
            assert_eq!(c[0], vertices[0]);
            assert_eq!(c[1], vertices[1]);
            assert_eq!(c[2], vertices[2]);
        }
    }
    #[test]
    fn test_cell_to_entities_edges() {
        //! Test cell edges
        let t = example_topology();
        for (i, edges) in [[0, 1, 2], [3, 4, 0]].iter().enumerate() {
            let c = t
                .cell_to_entities((ReferenceCellType::Triangle, i), 1)
                .unwrap();
            assert_eq!(c.len(), 3);
            assert_eq!(c[0], edges[0]);
            assert_eq!(c[1], edges[1]);
            assert_eq!(c[2], edges[2]);
        }
    }

    #[test]
    fn test_mixed_counts() {
        //! Test entity counts
        let t = example_topology_mixed();
        assert_eq!(t.dim(), 2);
        assert_eq!(t.entity_count(ReferenceCellType::Point), 5);
        assert_eq!(t.entity_count(ReferenceCellType::Interval), 6);
        assert_eq!(t.entity_count(ReferenceCellType::Triangle), 1);
        assert_eq!(t.entity_count(ReferenceCellType::Quadrilateral), 1);
    }

    #[test]
    fn test_mixed_cell_entities_vertices() {
        //! Test vertex-to-cell connectivity
        let t = example_topology_mixed();
        let c = t
            .cell_to_entities((ReferenceCellType::Quadrilateral, 0), 0)
            .unwrap();
        assert_eq!(c.len(), 4);
        // cell 0
        assert_eq!(c[0], 0);
        assert_eq!(c[1], 1);
        assert_eq!(c[2], 2);
        assert_eq!(c[3], 3);

        let c = t
            .cell_to_entities((ReferenceCellType::Triangle, 0), 0)
            .unwrap();
        assert_eq!(c.len(), 3);
        // cell 1
        assert_eq!(c[0], 1);
        assert_eq!(c[1], 4);
        assert_eq!(c[2], 3);
    }

    #[test]
    fn test_mixed_cell_entities_edges() {
        //! Test edge-to-cell connectivity
        let t = example_topology_mixed();
        let c = t
            .cell_to_entities((ReferenceCellType::Quadrilateral, 0), 1)
            .unwrap();

        assert_eq!(c.len(), 4);
        // cell 0
        assert_eq!(c[0], 0);
        assert_eq!(c[1], 1);
        assert_eq!(c[2], 2);
        assert_eq!(c[3], 3);

        let c = t
            .cell_to_entities((ReferenceCellType::Triangle, 0), 1)
            .unwrap();
        assert_eq!(c.len(), 3);
        // cell 1
        assert_eq!(c[0], 4);
        assert_eq!(c[1], 2);
        assert_eq!(c[2], 5);
    }
}
