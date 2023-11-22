//! Functions to create simple example grids

use crate::grid::SerialGrid;
use bempp_element::cell::Triangle;
use bempp_tools::arrays::{to_matrix, zero_matrix, AdjacencyList};
use bempp_traits::arrays::AdjacencyListAccess;
use bempp_traits::cell::{ReferenceCell, ReferenceCellType};
use bempp_traits::grid::{Geometry, Grid, Topology};
use rlst_common::traits::{RandomAccessByRef, RandomAccessMut};

/// Create a regular sphere
///
/// A regular sphere is created by starting with a regular octahedron. The shape is then refined `refinement_level` times.
/// Each time the grid is refined, each triangle is split into four triangles (by adding lines connecting the midpoints of
/// each edge). The new points are then scaled so that they are a distance of 1 from the origin.
pub fn regular_sphere(refinement_level: usize) -> SerialGrid {
    let mut g = SerialGrid::new(
        to_matrix(
            &[
                0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, -1.0,
            ],
            [6, 3],
        ),
        AdjacencyList::from_data(
            vec![
                0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4,
            ],
            vec![0, 3, 6, 9, 12, 15, 18, 21, 24],
        ),
        vec![ReferenceCellType::Triangle; 8],
    );
    let ref_e = Triangle {};
    for _level in 0..refinement_level {
        let nvertices_old = g.topology().entity_count(0);
        let ncells_old = g.topology().entity_count(2);
        let nvertices = g.topology().entity_count(0) + g.topology().entity_count(1);
        let mut coordinates = zero_matrix([nvertices, 3]);
        let mut cells = AdjacencyList::<usize>::new();

        for i in 0..ncells_old {
            let ti = g.topology().index_map()[i];
            let tedges = (0..3)
                .map(|x| unsafe { g.topology().connectivity(2, 1).row_unchecked(ti)[x] })
                .collect::<Vec<usize>>();
            let gi = g.geometry().index_map()[i];
            let tv = g.topology().cell(ti).unwrap();
            let gv = g.geometry().cell_vertices(gi).unwrap();
            for (j, gv_j) in gv.iter().enumerate() {
                for k in 0..g.geometry().dim() {
                    *coordinates.get_mut([tv[j], k]).unwrap() =
                        *g.geometry().coordinate(*gv_j, k).unwrap();
                }
            }

            for (j, tedges_j) in tedges.iter().enumerate() {
                let vs = ref_e.connectivity(1, j, 0).unwrap();
                let pt = (0..3)
                    .map(|k| {
                        (*coordinates.get([tv[vs[0]], k]).unwrap()
                            + *coordinates.get([tv[vs[1]], k]).unwrap())
                            / 2.0
                    })
                    .collect::<Vec<f64>>();

                let norm = (pt[0].powi(2) + pt[1].powi(2) + pt[2].powi(2)).sqrt();
                for (k, pt_k) in pt.iter().enumerate() {
                    *coordinates.get_mut([nvertices_old + tedges_j, k]).unwrap() = *pt_k / norm;
                }
            }

            cells.add_row(&[tv[0], nvertices_old + tedges[2], nvertices_old + tedges[1]]);
            cells.add_row(&[tv[1], nvertices_old + tedges[0], nvertices_old + tedges[2]]);
            cells.add_row(&[tv[2], nvertices_old + tedges[1], nvertices_old + tedges[0]]);
            cells.add_row(&[
                nvertices_old + tedges[0],
                nvertices_old + tedges[1],
                nvertices_old + tedges[2],
            ]);
        }
        let ncells = cells.num_rows();
        g = SerialGrid::new(
            coordinates,
            cells,
            vec![ReferenceCellType::Triangle; ncells],
        );
    }

    g
}

#[cfg(test)]
mod test {
    use crate::shapes::*;

    #[test]
    fn test_regular_sphere_0() {
        let _g = regular_sphere(0);
    }

    #[test]
    fn test_regular_spheres() {
        let _g1 = regular_sphere(1);
        let _g2 = regular_sphere(2);
        let _g3 = regular_sphere(3);
    }

    #[test]
    fn test_normal_is_outward() {
        for i in 0..3 {
            let g = regular_sphere(i);
            let points = to_matrix(&[1.0 / 3.0, 1.0 / 3.0], [1, 2]);

            let mut mapped_pt = zero_matrix([1, 3]);
            let mut normal = zero_matrix([1, 3]);

            for i in 0..g.geometry().cell_count() {
                g.geometry().compute_points(&points, i, &mut mapped_pt);
                g.geometry().compute_normals(&points, i, &mut normal);
                let dot = *mapped_pt.get([0, 0]).unwrap() * *normal.get([0, 0]).unwrap()
                    + *mapped_pt.get([0, 1]).unwrap() * *normal.get([0, 1]).unwrap()
                    + *mapped_pt.get([0, 2]).unwrap() * *normal.get([0, 2]).unwrap();
                assert!(dot > 0.0);
            }
        }
    }
}
