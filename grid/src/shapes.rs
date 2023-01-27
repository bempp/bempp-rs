//! Functions to create simple example grids

pub use crate::grid::SerialGrid;
use solvers_tools::arrays::AdjacencyList;
use solvers_tools::arrays::Array2D;
use solvers_traits::cell::ReferenceCellType;
pub use solvers_traits::grid::Geometry;
pub use solvers_traits::grid::Grid;
pub use solvers_traits::grid::Topology;

/// Create a regular sphere
///
/// A regular sphere is created by starting with a regular octahedron. The shape is then refined `refinement_level` times.
/// Each time the grid is refined, each triangle is split into four triangles (by adding lines connecting the midpoints of
/// each edge). The new points are then scaled so that they are a distance of 1 from the origin.
pub fn regular_sphere(refinement_level: usize) -> SerialGrid {
    let mut g = SerialGrid::new(
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
        vec![ReferenceCellType::Triangle; 8],
    );
    for _level in 0..refinement_level {
        let nvertices = g.topology_mut().entity_count(0) + g.topology_mut().entity_count(1);
        let mut coordinates = Array2D::<f64>::new((nvertices, 3));
        let mut cells = AdjacencyList::<usize>::new();

        for i in 0..g.geometry().point_count() {
            let pt = g.geometry().point(i).unwrap();
            for (j, c) in pt.iter().enumerate() {
                *coordinates.get_mut(i, j).unwrap() = *c;
            }
        }
        for edge in 0..g.topology_mut().entity_count(1) {
            let mut pt = [0.0, 0.0, 0.0];
            for j in 0..3 {
                for i in 0..2 {
                    let n = *g.topology_mut().connectivity(1, 0).get(edge, i).unwrap();
                    pt[j] += g.geometry().point(n).unwrap()[j];
                }
                pt[j] /= 2.0;
            }
            let norm = (pt[0].powi(2) + pt[1].powi(2) + pt[2].powi(2)).sqrt();

            for j in 0..3 {
                *coordinates
                    .get_mut(g.topology_mut().entity_count(0) + edge, j)
                    .unwrap() = pt[j] / norm;
            }
        }

        let nvertices = g.topology_mut().entity_count(0);
        for triangle in 0..g.topology_mut().entity_count(2) {
            let vs = [
                g.topology().cell(triangle).unwrap()[0],
                g.topology().cell(triangle).unwrap()[1],
                g.topology().cell(triangle).unwrap()[2],
            ];
            let es = g.topology_mut().connectivity(2, 1).row(triangle).unwrap();
            cells.add_row(&[vs[0], nvertices + es[2], nvertices + es[1]]);
            cells.add_row(&[vs[1], nvertices + es[0], nvertices + es[2]]);
            cells.add_row(&[vs[2], nvertices + es[1], nvertices + es[0]]);
            cells.add_row(&[nvertices + es[0], nvertices + es[1], nvertices + es[2]]);
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
}
