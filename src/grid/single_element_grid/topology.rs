//! Implementation of grid topology

use crate::element::reference_cell;
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

/// Topology of a single element grid
pub struct SingleElementTopology {
    dim: usize,
    index_map: Vec<usize>,
    entities_to_vertices: Vec<Vec<Vec<usize>>>,
    cells_to_entities: Vec<Vec<Vec<usize>>>,
    entities_to_cells: Vec<Vec<Vec<CellLocalIndexPair<usize>>>>,
    entity_types: Vec<ReferenceCellType>,
    vertex_indices_to_ids: Vec<usize>,
    vertex_ids_to_indices: HashMap<usize, usize>,
    cell_indices_to_ids: Vec<usize>,
    cell_ids_to_indices: HashMap<usize, usize>,
    cell_types: [ReferenceCellType; 1],
}

unsafe impl Sync for SingleElementTopology {}

impl SingleElementTopology {
    /// Create a topology
    pub fn new(
        cells_input: &[usize],
        cell_type: ReferenceCellType,
        point_indices_to_ids: &[usize],
        grid_cell_indices_to_ids: &[usize],
    ) -> Self {
        let size = reference_cell::entity_counts(cell_type)[0];
        let ncells = cells_input.len() / size;

        let mut vertex_indices_to_ids = vec![];
        let mut vertex_ids_to_indices = HashMap::new();
        let mut cell_indices_to_ids = vec![];
        let mut cell_ids_to_indices = HashMap::new();

        let mut index_map = vec![0; ncells];
        let mut vertices = vec![];
        let dim = reference_cell::dim(cell_type);

        let entity_types = reference_cell::entity_types(cell_type)
            .iter()
            .filter(|t| !t.is_empty())
            .map(|t| t[0])
            .collect::<Vec<_>>();

        let mut cells_to_entities = vec![vec![vec![]; ncells]; dim + 1];
        let mut entities_to_cells = vec![vec![]; dim + 1];
        let mut entities_to_vertices = vec![vec![]; dim];

        entities_to_cells[dim] = vec![vec![]; ncells];

        let mut start = 0;
        for (cell_i, i) in index_map.iter_mut().enumerate() {
            let cell = &cells_input[start..start + size];
            *i = cell_i;
            cell_indices_to_ids.push(grid_cell_indices_to_ids[cell_i]);
            cell_ids_to_indices.insert(grid_cell_indices_to_ids[cell_i], cell_i);
            let mut row = vec![];
            for v in cell {
                if !vertices.contains(v) {
                    entities_to_cells[0].push(vec![]);
                    vertices.push(*v);
                    vertex_indices_to_ids.push(point_indices_to_ids[*v]);
                    vertex_ids_to_indices.insert(point_indices_to_ids[*v], *v);
                }
                row.push(vertices.iter().position(|&r| r == *v).unwrap());
            }

            for (local_index, v) in row.iter().enumerate() {
                entities_to_cells[0][*v].push(CellLocalIndexPair::new(cell_i, local_index));
            }
            entities_to_cells[dim][cell_i] = vec![CellLocalIndexPair::new(cell_i, 0)];

            cells_to_entities[0][cell_i] = row;
            cells_to_entities[dim][cell_i] = vec![cell_i];

            start += size;
        }

        for i in 0..vertices.len() {
            entities_to_vertices[0].push(vec![i]);
        }
        for d in 1..dim {
            let mut c_to_e = vec![];
            let ref_conn = &reference_cell::connectivity(cell_type)[d];
            for (cell_i, cell) in cells_to_entities[0].iter().enumerate() {
                let mut entity_ids = vec![];
                for (local_index, rc) in ref_conn.iter().enumerate() {
                    let vertices = rc[0].iter().map(|x| cell[*x]).collect::<Vec<_>>();
                    let mut found = false;
                    for (entity_index, entity) in entities_to_vertices[d].iter().enumerate() {
                        if all_equal(entity, &vertices) {
                            entity_ids.push(entity_index);
                            entities_to_cells[d][entity_index]
                                .push(CellLocalIndexPair::new(cell_i, local_index));
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        entity_ids.push(entities_to_vertices[d].len());
                        entities_to_cells[d]
                            .push(vec![CellLocalIndexPair::new(cell_i, local_index)]);
                        entities_to_vertices[d].push(vertices);
                    }
                }
                c_to_e.push(entity_ids);
            }
            cells_to_entities[d] = c_to_e;
        }

        Self {
            dim,
            index_map,
            entities_to_vertices,
            cells_to_entities,
            entities_to_cells,
            entity_types,
            vertex_indices_to_ids,
            vertex_ids_to_indices,
            cell_indices_to_ids,
            cell_ids_to_indices,
            cell_types: [cell_type],
        }
    }
}

impl Topology for SingleElementTopology {
    type IndexType = usize;

    fn dim(&self) -> usize {
        self.dim
    }
    fn index_map(&self) -> &[usize] {
        &self.index_map
    }
    fn entity_count(&self, etype: ReferenceCellType) -> usize {
        if self.entity_types.contains(&etype) {
            self.entities_to_cells[reference_cell::dim(etype)].len()
        } else {
            0
        }
    }
    fn entity_count_by_dim(&self, dim: usize) -> usize {
        self.entity_count(self.entity_types[dim])
    }
    fn cell(&self, index: usize) -> Option<&[usize]> {
        if index < self.cells_to_entities[self.dim].len() {
            Some(&self.cells_to_entities[self.dim][index])
        } else {
            None
        }
    }
    fn cell_type(&self, index: usize) -> Option<ReferenceCellType> {
        if index < self.cells_to_entities[self.dim].len() {
            Some(self.entity_types[self.dim])
        } else {
            None
        }
    }

    fn entity_types(&self, dim: usize) -> &[ReferenceCellType] {
        &self.entity_types[dim..dim + 1]
    }

    fn cell_ownership(&self, _index: usize) -> Ownership {
        Ownership::Owned
    }
    fn vertex_ownership(&self, _index: usize) -> Ownership {
        Ownership::Owned
    }

    fn cell_to_entities(&self, index: usize, dim: usize) -> Option<&[usize]> {
        if dim <= self.dim && index < self.cells_to_entities[dim].len() {
            Some(&self.cells_to_entities[dim][index])
        } else {
            None
        }
    }
    fn cell_to_flat_entities(&self, index: usize, dim: usize) -> Option<&[usize]> {
        self.cell_to_entities(index, dim)
    }
    fn entity_to_cells(&self, dim: usize, index: usize) -> Option<&[CellLocalIndexPair<usize>]> {
        if dim <= self.dim && index < self.entities_to_cells[dim].len() {
            Some(&self.entities_to_cells[dim][index])
        } else {
            None
        }
    }

    fn entity_to_flat_cells(
        &self,
        dim: usize,
        index: Self::IndexType,
    ) -> Option<&[CellLocalIndexPair<usize>]> {
        self.entity_to_cells(dim, index)
    }

    fn entity_vertices(&self, dim: usize, index: usize) -> Option<&[usize]> {
        if dim == self.dim {
            self.cell_to_entities(index, 0)
        } else if dim < self.dim && index < self.entities_to_vertices[dim].len() {
            Some(&self.entities_to_vertices[dim][index])
        } else {
            None
        }
    }

    fn vertex_index_to_id(&self, index: usize) -> usize {
        self.vertex_indices_to_ids[index]
    }
    fn cell_index_to_id(&self, index: usize) -> usize {
        self.cell_indices_to_ids[index]
    }
    fn vertex_id_to_index(&self, id: usize) -> usize {
        if self.vertex_ids_to_indices.contains_key(&id) {
            self.vertex_ids_to_indices[&id]
        } else {
            panic!("Vertex with id {} not found", id);
        }
    }
    fn cell_id_to_index(&self, id: usize) -> usize {
        self.cell_ids_to_indices[&id]
    }
    fn vertex_index_to_flat_index(&self, index: usize) -> usize {
        index
    }
    fn edge_index_to_flat_index(&self, index: usize) -> usize {
        index
    }
    fn face_index_to_flat_index(&self, index: usize) -> usize {
        index
    }
    fn vertex_flat_index_to_index(&self, index: usize) -> usize {
        index
    }
    fn edge_flat_index_to_index(&self, index: usize) -> usize {
        index
    }
    fn face_flat_index_to_index(&self, index: usize) -> usize {
        index
    }
    fn cell_types(&self) -> &[ReferenceCellType] {
        &self.cell_types
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn example_topology() -> SingleElementTopology {
        //! An example topology
        SingleElementTopology::new(
            &[0, 1, 2, 2, 1, 3],
            ReferenceCellType::Triangle,
            &[0, 1, 2, 3],
            &[0, 1],
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
    fn test_cell_entities_vertices() {
        //! Test cell vertices
        let t = example_topology();
        for (i, vertices) in [[0, 1, 2], [2, 1, 3]].iter().enumerate() {
            let c = t.cell_to_entities(i, 0).unwrap();
            assert_eq!(c.len(), 3);
            assert_eq!(c, vertices);
        }
    }

    #[test]
    fn test_cell_entities_edges() {
        //! Test cell edges
        let t = example_topology();
        for (i, edges) in [[0, 1, 2], [3, 4, 0]].iter().enumerate() {
            let c = t.cell_to_entities(i, 1).unwrap();
            assert_eq!(c.len(), 3);
            assert_eq!(c, edges);
        }
    }

    #[test]
    fn test_cell_entities_cells() {
        //! Test cells
        let t = example_topology();
        for i in 0..2 {
            let c = t.cell_to_entities(i, 2).unwrap();
            assert_eq!(c.len(), 1);
            assert_eq!(c[0], i);
        }
    }

    #[test]
    fn test_entities_to_cells_vertices() {
        //! Test vertex-to-cell connectivity
        let t = example_topology();
        let c_to_e = (0..t.entity_count(ReferenceCellType::Triangle))
            .map(|i| t.cell_to_entities(i, 0).unwrap())
            .collect::<Vec<_>>();
        let e_to_c = (0..t.entity_count(ReferenceCellType::Point))
            .map(|i| {
                t.entity_to_cells(0, i)
                    .unwrap()
                    .iter()
                    .map(|x| x.cell)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for (i, cell) in c_to_e.iter().enumerate() {
            for v in *cell {
                assert!(e_to_c[*v].contains(&i));
            }
        }
        for (i, cells) in e_to_c.iter().enumerate() {
            for c in cells {
                assert!(c_to_e[*c].contains(&i));
            }
        }
    }

    #[test]
    fn test_entities_to_cells_edges() {
        //! Test edge-to-cell connectivity
        let t = example_topology();
        let c_to_e = (0..t.entity_count(ReferenceCellType::Triangle))
            .map(|i| t.cell_to_entities(i, 1).unwrap())
            .collect::<Vec<_>>();
        let e_to_c = (0..t.entity_count(ReferenceCellType::Interval))
            .map(|i| {
                t.entity_to_cells(1, i)
                    .unwrap()
                    .iter()
                    .map(|x| x.cell)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for (i, cell) in c_to_e.iter().enumerate() {
            for v in *cell {
                assert!(e_to_c[*v].contains(&i));
            }
        }
        for (i, cells) in e_to_c.iter().enumerate() {
            for c in cells {
                assert!(c_to_e[*c].contains(&i));
            }
        }
    }
}
