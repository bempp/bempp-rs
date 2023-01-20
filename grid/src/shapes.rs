pub use crate::grid::SerialTriangle3DGrid;
use solvers_tools::arrays::AdjacencyList;
use solvers_tools::arrays::Array2D;
pub use solvers_traits::grid::Geometry;
pub use solvers_traits::grid::Grid;
pub use solvers_traits::grid::Topology;

pub fn regular_sphere(refinement_level: usize) -> SerialTriangle3DGrid {
    let mut g = SerialTriangle3DGrid::new(
        Array2D::from_data(
            vec![
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
                0.0, -1.0,
            ],
            (6, 3),
        ),
        AdjacencyList::from_data(
            vec![
                0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4,
            ],
            vec![0, 3, 6, 9, 12, 15, 18, 21, 24],
        ),
    );
    for _level in 0..refinement_level {
        let nvertices = g.topology().entity_count(0) + g.topology().entity_count(1);
        let mut coordinates = Array2D::<f64>::new((nvertices, 3));
        let mut cells = AdjacencyList::<usize>::new();

        for i in 0..g.geometry().point_count() {
            for j in 0..g.geometry().dim() {
                *coordinates.get_mut(i, j).unwrap() = g.geometry().point(i).unwrap()[j];
            }
        }
        for edge in 0..g.topology().entity_count(1) {
            let mut pt = [0.0, 0.0, 0.0];
            for j in 0..3 {
                for i in 0..2 {
                    pt[j] += g
                        .geometry()
                        .point(g.topology().connectivity_1_0[2 * edge + i])
                        .unwrap()[j];
                }
                pt[j] /= 2.0;
            }
            let norm = (pt[0].powi(2) + pt[1].powi(2) + pt[2].powi(2)).sqrt();

            for j in 0..3 {
                *coordinates
                    .get_mut(g.topology().entity_count(0) + edge, j)
                    .unwrap() = pt[j] / norm;
            }
        }

        for triangle in 0..g.topology().entity_count(2) {
            let vs = g.topology().cell(triangle).unwrap();
            let es = &g.topology().connectivity_2_1[3 * triangle..3 * (triangle + 1)];
            cells.add_row(vec![
                vs[0],
                g.topology().entity_count(0) + es[2],
                g.topology().entity_count(0) + es[1],
            ]);
            cells.add_row(vec![
                vs[1],
                g.topology().entity_count(0) + es[0],
                g.topology().entity_count(0) + es[2],
            ]);
            cells.add_row(vec![
                vs[2],
                g.topology().entity_count(0) + es[1],
                g.topology().entity_count(0) + es[0],
            ]);
            cells.add_row(vec![
                g.topology().entity_count(0) + es[0],
                g.topology().entity_count(0) + es[1],
                g.topology().entity_count(0) + es[2],
            ]);
        }
        g = SerialTriangle3DGrid::new(coordinates, cells);
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
}
