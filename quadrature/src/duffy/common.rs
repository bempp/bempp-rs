//! Utility functions for creating Duffy rules
use itertools::Itertools;

/// Apply a callable to each tuple chunk (single point) of an array.
///
/// Each 2-tuple in `points` represents a 2d point. The callable is applied to
/// each point and transforms it to a new point.
pub(crate) fn transform_coords(points: &mut [f64], fun: &impl Fn((f64, f64)) -> (f64, f64)) {
    for (first, second) in points.iter_mut().tuples() {
        (*first, *second) = fun((*first, *second));
    }
}

/// Create a triangle mapper
///
/// The vertices in our reference element are
/// 0: (0, 0), 1: (1, 0), 2: (0, 1)
///
/// The Duffy rule has vertices with a reference element
/// 0: (0, 0), 1: (1, 0), 2: (1, 1)
/// We need to map (0, 0) -> v0, (1, 0) -> v1, and (1, 1) -> v2,
/// where v2 is implicitly defined as the index 3 - v0 - v1 (
/// since all triangle indices together sum up to 3)
pub(crate) fn create_triangle_mapper(v0: usize, v1: usize) -> impl Fn((f64, f64)) -> (f64, f64) {
    let get_reference_vertex = |index| -> Result<(f64, f64), ()> {
        match index {
            0 => Ok((0.0, 0.0)),
            1 => Ok((1.0, 0.0)),
            2 => Ok((0.0, 1.0)),
            _ => Err(()),
        }
    };

    let p0 = get_reference_vertex(v0).unwrap();
    let p1 = get_reference_vertex(v1).unwrap();
    let p2 = get_reference_vertex(3 - v0 - v1).unwrap();

    //  The tranformation is offset + A * point,
    //  The offset is just identical to p0.

    // The matrix A has two columns. The first column is p1 - p0.
    // The second column is p2 - p1

    let col0 = (p1.0 - p0.0, p1.1 - p0.1);
    let col1 = (p2.0 - p1.0, p2.1 - p1.1);

    // We return a closure that performs the actual transformation.
    // We need to capture its values via move as they stop existing
    // once the closure is moved out of the function.

    move |point: (f64, f64)| -> (f64, f64) {
        (
            p0.0 + col0.0 * point.0 + col1.0 * point.1,
            p0.1 + col0.1 * point.0 + col1.1 * point.1,
        )
    }
}

/// Create a quadrilateral mapper
///
/// The vertices in the reference element are.
/// 0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)
///
/// This function creates a map so that (0, 0) -> v0 and (1, 0) -> v1.
pub(crate) fn create_quadrilateral_mapper(
    v0: usize,
    v1: usize,
) -> impl Fn((f64, f64)) -> (f64, f64) {
    // Choose the third vertex.

    let v2 = match (v0, v1) {
        // We map the 0th vertex to v0 and the first vertex to v1.
        // This match chooses the correct vertex for v2. There are
        // two vertices adjacent to v0. v1 is one of them and v2
        // will be the other one.
        (0, 1) => 2,
        (1, 0) => 3,
        (1, 3) => 0,
        (3, 1) => 2,
        (3, 2) => 1,
        (2, 3) => 0,
        (0, 2) => 1,
        (2, 0) => 3,
        _ => panic!("({v0}, {v1}) is not an edge of the unit quadrilateral."),
    };

    let get_reference_vertex = |index| -> Result<(f64, f64), ()> {
        match index {
            0 => Ok((0.0, 0.0)),
            1 => Ok((1.0, 0.0)),
            2 => Ok((0.0, 1.0)),
            3 => Ok((1.0, 1.0)),
            _ => Err(()),
        }
    };

    let p0 = get_reference_vertex(v0).unwrap();
    let p1 = get_reference_vertex(v1).unwrap();
    let p2 = get_reference_vertex(v2).unwrap();

    //  The tranformation is offset + A * point,
    //  The offset is just identical to p0.

    // The matrix A has two columns. The first column is p1 - p0.
    // The second column is p2 - p0

    let col0 = (p1.0 - p0.0, p1.1 - p0.1);
    let col1 = (p2.0 - p0.0, p2.1 - p0.1);

    // We return a closure that performs the actual transformation.
    // We need to capture its values via move as they stop existing
    // once the closure is moved out of the function.

    move |point: (f64, f64)| -> (f64, f64) {
        (
            p0.0 + col0.0 * point.0 + col1.0 * point.1,
            p0.1 + col0.1 * point.0 + col1.1 * point.1,
        )
    }
}

/// Get the next vertex in an anticlockwise direction
pub(crate) fn next_triangle_vertex(index: usize) -> usize {
    (index + 1) % 3
}

/// Get the next vertex in an anticlockwise direction
pub(crate) fn next_quadrilateral_vertex(index: usize) -> usize {
    match index {
        0 => 1,
        1 => 3,
        2 => 0,
        3 => 2,
        _ => panic!("Unknown vertex index."),
    }
}
