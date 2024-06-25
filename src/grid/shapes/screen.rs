//! Regular sphere grid

use crate::grid::flat_triangle_grid::{FlatTriangleGrid, FlatTriangleGridBuilder};
use crate::grid::mixed_grid::{MixedGrid, MixedGridBuilder};
use crate::grid::single_element_grid::{SingleElementGrid, SingleElementGridBuilder};
use crate::traits::grid::Builder;
use ndelement::types::ReferenceCellType;
use num::Float;
use rlst::{
    dense::array::{views::ArrayViewMut, Array},
    BaseArray, MatrixInverse, RlstScalar, VectorContainer,
};
/// Create a square grid with triangle cells
///
/// Create a grid of the square \[0,1\]^2 with triangle cells. The input ncells is the number of cells
/// along each side of the square.
pub fn screen_triangles<T: Float + RlstScalar<Real = T>>(ncells: usize) -> FlatTriangleGrid<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    if ncells == 0 {
        panic!("Cannot create a grid with 0 cells");
    }
    let mut b = FlatTriangleGridBuilder::new_with_capacity(
        (ncells + 1) * (ncells + 1),
        2 * ncells * ncells,
        (),
    );

    let zero = T::from(0.0).unwrap();
    let n = T::from(ncells + 1).unwrap();
    for y in 0..ncells + 1 {
        for x in 0..ncells + 1 {
            b.add_point(
                y * (ncells + 1) + x,
                [T::from(x).unwrap() / n, T::from(y).unwrap() / n, zero],
            );
        }
    }
    for y in 0..ncells {
        for x in 0..ncells {
            b.add_cell(
                2 * y * ncells + 2 * x,
                [
                    y * (ncells + 1) + x,
                    y * (ncells + 1) + x + 1,
                    y * (ncells + 1) + x + ncells + 2,
                ],
            );
            b.add_cell(
                2 * y * ncells + 2 * x + 1,
                [
                    y * (ncells + 1) + x,
                    y * (ncells + 1) + x + ncells + 2,
                    y * (ncells + 1) + x + ncells + 1,
                ],
            );
        }
    }

    b.create_grid()
}

/// Create a square grid with quadrilateral cells
///
/// Create a grid of the square \[0,1\]^2 with quadrilateral cells. The input ncells is the number of
/// cells along each side of the square.
pub fn screen_quadrilaterals<T: Float + RlstScalar<Real = T>>(ncells: usize) -> SingleElementGrid<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    if ncells == 0 {
        panic!("Cannot create a grid with 0 cells");
    }
    let mut b = SingleElementGridBuilder::new_with_capacity(
        (ncells + 1) * (ncells + 1),
        ncells * ncells,
        (ReferenceCellType::Quadrilateral, 1),
    );

    let zero = T::from(0.0).unwrap();
    let n = T::from(ncells + 1).unwrap();
    for y in 0..ncells + 1 {
        for x in 0..ncells + 1 {
            b.add_point(
                y * (ncells + 1) + x,
                [T::from(x).unwrap() / n, T::from(y).unwrap() / n, zero],
            );
        }
    }
    for y in 0..ncells {
        for x in 0..ncells {
            b.add_cell(
                y * ncells + x,
                vec![
                    y * (ncells + 1) + x,
                    y * (ncells + 1) + x + 1,
                    y * (ncells + 1) + x + ncells + 1,
                    y * (ncells + 1) + x + ncells + 2,
                ],
            );
        }
    }

    b.create_grid()
}

/// Create a rectangular grid with quadrilateral cells
///
/// Create a grid of the square \[0,2\]x\[0,1\] with triangle cells on the left half and quadrilateral
/// cells on the right half. The input ncells is the number of cells along each side of the unit
/// square.
pub fn screen_mixed<T: Float + RlstScalar<Real = T>>(ncells: usize) -> MixedGrid<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    if ncells == 0 {
        panic!("Cannot create a grid with 0 cells");
    }
    let mut b = MixedGridBuilder::new_with_capacity(
        2 * (ncells + 1) * (ncells + 1),
        3 * ncells * ncells,
        (),
    );

    let zero = T::from(0.0).unwrap();
    let n = T::from(ncells + 1).unwrap();
    for y in 0..ncells + 1 {
        for x in 0..2 * (ncells + 1) {
            b.add_point(
                2 * y * (ncells + 1) + x,
                [T::from(x).unwrap() / n, T::from(y).unwrap() / n, zero],
            );
        }
    }
    for y in 0..ncells {
        for x in 0..ncells {
            b.add_cell(
                2 * y * ncells + 2 * x,
                (
                    vec![
                        2 * y * (ncells + 1) + x,
                        2 * y * (ncells + 1) + x + 1,
                        2 * y * (ncells + 1) + x + 2 * ncells + 3,
                    ],
                    ReferenceCellType::Triangle,
                    1,
                ),
            );
            b.add_cell(
                2 * y * ncells + 2 * x + 1,
                (
                    vec![
                        2 * y * (ncells + 1) + x,
                        2 * y * (ncells + 1) + x + 2 * ncells + 3,
                        2 * y * (ncells + 1) + x + 2 * ncells + 2,
                    ],
                    ReferenceCellType::Triangle,
                    1,
                ),
            );
            b.add_cell(
                2 * ncells * ncells + y * ncells + x,
                (
                    vec![
                        (ncells + 1) + 2 * y * (ncells + 1) + x,
                        (ncells + 1) + 2 * y * (ncells + 1) + x + 1,
                        (ncells + 1) + 2 * y * (ncells + 1) + x + 2 * ncells + 3,
                        (ncells + 1) + 2 * y * (ncells + 1) + x + 2 * ncells + 2,
                    ],
                    ReferenceCellType::Quadrilateral,
                    1,
                ),
            );
        }
    }
    b.create_grid()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::traits::grid::{GridType, ReferenceMapType};

    #[test]
    fn test_screen_triangles() {
        let _g1 = screen_triangles::<f64>(1);
        let _g2 = screen_triangles::<f64>(2);
        let _g3 = screen_triangles::<f64>(3);
    }

    #[test]
    fn test_screen_triangles_normal() {
        for i in 1..5 {
            let g = screen_triangles::<f64>(i);
            let points = vec![1.0 / 3.0, 1.0 / 3.0];
            let map = g.reference_to_physical_map(&points);
            let mut mapped_pt = vec![0.0; 3];
            let mut normal = vec![0.0; 3];
            for i in 0..g.number_of_cells() {
                map.reference_to_physical(i, &mut mapped_pt);
                map.normal(i, &mut normal);
                assert!(normal[2] > 0.0);
            }
        }
    }

    #[test]
    fn test_screen_quadrilaterals() {
        let _g1 = screen_quadrilaterals::<f64>(1);
        let _g2 = screen_quadrilaterals::<f64>(2);
        let _g3 = screen_quadrilaterals::<f64>(3);
    }

    #[test]
    fn test_screen_quadrilaterals_normal() {
        for i in 1..5 {
            let g = screen_quadrilaterals::<f64>(i);
            let points = vec![1.0 / 3.0, 1.0 / 3.0];
            let map = g.reference_to_physical_map(&points);
            let mut mapped_pt = vec![0.0; 3];
            let mut normal = vec![0.0; 3];
            for i in 0..g.number_of_cells() {
                map.reference_to_physical(i, &mut mapped_pt);
                map.normal(i, &mut normal);
                assert!(normal[2] > 0.0);
            }
        }
    }

    #[test]
    fn test_screen_mixed() {
        let _g1 = screen_mixed::<f64>(1);
        let _g2 = screen_mixed::<f64>(2);
        let _g3 = screen_mixed::<f64>(3);
    }

    #[test]
    fn test_screen_mixed_normal() {
        for i in 1..5 {
            let g = screen_mixed::<f64>(i);
            let points = vec![1.0 / 3.0, 1.0 / 3.0];
            let map = g.reference_to_physical_map(&points);
            let mut mapped_pt = vec![0.0; 3];
            let mut normal = vec![0.0; 3];
            for i in 0..g.number_of_cells() {
                map.reference_to_physical(i, &mut mapped_pt);
                map.normal(i, &mut normal);
                assert!(normal[2] > 0.0);
            }
        }
    }
}
