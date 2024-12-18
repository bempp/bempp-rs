//! Definition of various test shapes.

use std::collections::HashMap;

use mpi::traits::{Communicator, Equivalence};
use ndelement::{ciarlet::CiarletElement, types::ReferenceCellType};
use ndgrid::{
    traits::{Builder, ParallelBuilder},
    types::RealScalar,
    ParallelGrid, SingleElementGrid, SingleElementGridBuilder,
};
use num::Float;

/// Create a regular sphere
///
/// A regular sphere is created by starting with a regular octahedron. The shape is then refined `refinement_level` times.
/// Each time the grid is refined, each triangle is split into four triangles (by adding lines connecting the midpoints of
/// each edge). The new points are then scaled so that they are a distance of 1 from the origin.
pub fn regular_sphere<T: RealScalar + Equivalence, C: Communicator>(
    refinement_level: u32,
    degree: usize,
    comm: &C,
) -> ParallelGrid<C, SingleElementGrid<T, CiarletElement<T>>> {
    if comm.rank() == 0 {
        let mut b = SingleElementGridBuilder::new_with_capacity(
            3,
            2 + usize::pow(4, refinement_level + 1),
            8 * usize::pow(4, refinement_level),
            (ReferenceCellType::Triangle, degree),
        );

        let mut points = Vec::<[T; 3]>::with_capacity(2 + usize::pow(4, refinement_level + 1));

        let zero = T::from(0.0).unwrap();
        let one = T::from(1.0).unwrap();
        let half = T::from(0.5).unwrap();

        points.push([zero, zero, one]);
        points.push([one, zero, zero]);
        points.push([zero, one, zero]);
        points.push([-one, zero, zero]);
        points.push([zero, -one, zero]);
        points.push([zero, zero, -one]);

        let mut point_n = 6;

        let mut cells = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4],
        ];
        let mut v = [[zero, zero, zero], [zero, zero, zero], [zero, zero, zero]];

        for level in 0..refinement_level {
            let mut edge_points = HashMap::new();
            let mut new_cells = Vec::with_capacity(8 * usize::pow(6, level));
            for c in &cells {
                for i in 0..3 {
                    for j in 0..3 {
                        v[i][j] = points[c[i]][j];
                    }
                }
                let edges = [[1, 2], [0, 2], [0, 1]]
                    .iter()
                    .map(|[i, j]| {
                        let mut pt_i = c[*i];
                        let mut pt_j = c[*j];
                        if pt_i > pt_j {
                            std::mem::swap(&mut pt_i, &mut pt_j);
                        }
                        *edge_points.entry((pt_i, pt_j)).or_insert_with(|| {
                            let v_i = v[*i];
                            let v_j = v[*j];
                            let mut new_pt = [
                                half * (v_i[0] + v_j[0]),
                                half * (v_i[1] + v_j[1]),
                                half * (v_i[2] + v_j[2]),
                            ];
                            let size = Float::sqrt(new_pt.iter().map(|&x| x * x).sum::<T>());
                            for i in new_pt.iter_mut() {
                                *i /= size;
                            }
                            points.push(new_pt);
                            let out = point_n;
                            point_n += 1;
                            out
                        })
                    })
                    .collect::<Vec<_>>();
                new_cells.push([c[0], edges[2], edges[1]]);
                new_cells.push([c[1], edges[0], edges[2]]);
                new_cells.push([c[2], edges[1], edges[0]]);
                new_cells.push([edges[0], edges[1], edges[2]]);
            }
            cells = new_cells;
        }
        for (i, v) in points.iter().enumerate() {
            b.add_point(i, v);
        }
        for (i, v) in cells.iter().enumerate() {
            b.add_cell(i, v);
        }

        b.create_parallel_grid(comm)
    } else {
        SingleElementGridBuilder::new(3, (ReferenceCellType::Triangle, degree))
            .receive_parallel_grid(comm, 0)
    }
}
