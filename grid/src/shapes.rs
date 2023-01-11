pub use crate::grid::SerialTriangle3DGrid;
pub use solvers_traits::grid::Geometry;
pub use solvers_traits::grid::Grid;
pub use solvers_traits::grid::Topology;

fn regular_sphere(refinement_level: usize) -> SerialTriangle3DGrid {
    let mut g = SerialTriangle3DGrid {
        coordinates: vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
            -1.0,
        ],
        cells: vec![
            0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4,
        ],
    };
    for level in 0..refinement_level {
        let nvertices = g.topology().entity_count(0) + g.topology().entity_count(1);
        let ncells = 4 * g.topology().entity_count(2);
        let mut coordinates = vec![0.0; 3 * nvertices];
        let mut cells = vec![0; 3 * ncells];

        for i in 0..g.coordinates.len() {
            coordinates[i] = g.coordinates[i];
        }
        for edge in 0..g.topology().entity_count(1) {
            let mut pt = [0.0, 0.0, 0.0];
            for j in 0..3 {
                for i in 0..2 {
                    pt[j] += g.coordinates[3 * g.topology().connectivity_1_0[2 * edge + i] + j]
                }
                pt[j] /= 2.0;
            }
            let norm = (pt[0].powi(2) + pt[1].powi(2) + pt[2].powi(2)).sqrt();

            for j in 0..3 {
                coordinates[3 * (g.topology().entity_count(0) + edge) + j] = pt[j] / norm;
            }
        }

        for triangle in 0..g.topology().entity_count(2) {
            let vs = &g.topology().cells[3 * triangle..3 * (triangle + 1)];
            let es = &g.topology().connectivity_2_1[3 * triangle..3 * (triangle + 1)];
            cells[12 * triangle] = vs[0];
            cells[12 * triangle + 1] = g.topology().entity_count(0) + es[2];
            cells[12 * triangle + 2] = g.topology().entity_count(0) + es[1];
            cells[12 * triangle + 3] = vs[1];
            cells[12 * triangle + 4] = g.topology().entity_count(0) + es[0];
            cells[12 * triangle + 5] = g.topology().entity_count(0) + es[2];
            cells[12 * triangle + 6] = vs[2];
            cells[12 * triangle + 7] = g.topology().entity_count(0) + es[1];
            cells[12 * triangle + 8] = g.topology().entity_count(0) + es[0];
            cells[12 * triangle + 9] = g.topology().entity_count(0) + es[0];
            cells[12 * triangle + 10] = g.topology().entity_count(0) + es[1];
            cells[12 * triangle + 11] = g.topology().entity_count(0) + es[2];
        }
        g = SerialTriangle3DGrid {
            coordinates: coordinates,
            cells: cells,
        };
    }

    g
}

#[cfg(test)]
mod test {
    use crate::shapes::*;
    use approx::*;

    #[test]
    fn test_regular_sphere_0() {
        let g = regular_sphere(0);
        let volume = g.cell_geometry(0).volume();
        for i in 0..g.topology().entity_count(2) {
            assert_relative_eq!(g.cell_geometry(i).volume(), volume);
        }
    }

    #[test]
    fn test_regular_sphere2() {
        let g1 = regular_sphere(1);
        let g2 = regular_sphere(2);
        let g3 = regular_sphere(3);
    }
}
