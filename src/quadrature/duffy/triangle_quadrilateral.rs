//! Duffy rules for triangles adjacent to quadrilaterals.
use crate::quadrature::{
    duffy::common::{
        create_quadrilateral_mapper, create_triangle_mapper, next_quadrilateral_vertex,
        next_triangle_vertex, transform_coords,
    },
    simplex_rules::simplex_rule,
    types::{
        CellToCellConnectivity, NumericalQuadratureDefinition, QuadratureError,
        TestTrialNumericalQuadratureDefinition,
    },
};
use ndelement::types::ReferenceCellType;

fn tri_quad_edge_points(
    interval_rule: &NumericalQuadratureDefinition,
    triangle_edge_vertices: (usize, usize),
    quadrilateral_edge_vertices: (usize, usize),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let NumericalQuadratureDefinition {
        dim: _,
        order: _,
        npoints,
        weights,
        points,
    } = interval_rule;

    let n_output_points = 6 * npoints * npoints * npoints * npoints;

    let mut quadrilateral_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut triangle_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut output_weights = Vec::<f64>::with_capacity(n_output_points);

    for index1 in 0..*npoints {
        for index2 in 0..*npoints {
            for index3 in 0..*npoints {
                for index4 in 0..*npoints {
                    let eta1 = points[index1];
                    let eta2 = points[index2];
                    let eta3 = points[index3];
                    let xi = points[index4];

                    // First part

                    let weight0 = weights[index1]
                        * weights[index2]
                        * weights[index3]
                        * weights[index4]
                        * xi
                        * xi
                        * (1.0 - xi);
                    let weight1 = weights[index1]
                        * weights[index2]
                        * weights[index3]
                        * weights[index4]
                        * xi
                        * xi
                        * eta1
                        * (1.0 - xi * eta1);

                    quadrilateral_output_points.push(xi * (1.0 - eta3) + eta3);
                    quadrilateral_output_points.push(xi * eta2);
                    triangle_output_points.push(xi * (1.0 - eta1 - eta3 + eta3));
                    triangle_output_points.push(xi * (1.0 - eta1));
                    output_weights.push(weight0);

                    // Second part

                    quadrilateral_output_points.push((1.0 - xi) * eta3);
                    quadrilateral_output_points.push(xi * eta3);
                    triangle_output_points.push(xi * (1.0 - eta3) + eta3);
                    triangle_output_points.push(xi * eta1);
                    output_weights.push(weight0);

                    // Third part

                    quadrilateral_output_points.push(xi * (1.0 - eta1 - eta3) + eta3);
                    quadrilateral_output_points.push(xi * eta2);
                    triangle_output_points.push(xi * (1.0 - eta3) + eta3);
                    triangle_output_points.push(xi);
                    output_weights.push(weight0);

                    // Fourth part

                    quadrilateral_output_points.push(xi * eta1 * (1.0 - eta3) + eta3);
                    quadrilateral_output_points.push(xi);
                    triangle_output_points.push(xi * eta1 * (1.0 - eta2 - eta3) + eta3);
                    triangle_output_points.push(xi * eta1 * (1.0 - eta2));
                    output_weights.push(weight1);

                    // Fifth part

                    quadrilateral_output_points.push((1.0 - xi * eta1) * eta3);
                    quadrilateral_output_points.push(xi);
                    triangle_output_points.push(xi * eta1 + (1.0 - xi * eta1) * eta3);
                    triangle_output_points.push(xi * eta1 * eta2);
                    output_weights.push(weight1);

                    // Sixth part

                    quadrilateral_output_points.push(xi * eta1 * (1.0 - eta2 - eta3) + eta1);
                    quadrilateral_output_points.push(xi);
                    triangle_output_points.push(xi * eta1 * (1.0 - eta3) + eta3);
                    triangle_output_points.push(xi * eta1);
                    output_weights.push(weight1);
                }
            }
        }
    }

    transform_coords(
        &mut triangle_output_points,
        &create_triangle_mapper(triangle_edge_vertices.0, triangle_edge_vertices.1),
    );
    transform_coords(
        &mut quadrilateral_output_points,
        &create_quadrilateral_mapper(quadrilateral_edge_vertices.0, quadrilateral_edge_vertices.1),
    );

    (
        output_weights,
        triangle_output_points,
        quadrilateral_output_points,
    )
}
fn edge_adjacent_triangle_quadrilateral(
    interval_rule: &NumericalQuadratureDefinition,
    test_singular_edge_vertices: (usize, usize),
    trial_singular_edge_vertices: (usize, usize),
) -> TestTrialNumericalQuadratureDefinition {
    let NumericalQuadratureDefinition {
        dim,
        order,
        npoints: _,
        weights: _,
        points: _,
    } = interval_rule;
    let (output_weights, test_output_points, trial_output_points) = tri_quad_edge_points(
        interval_rule,
        test_singular_edge_vertices,
        trial_singular_edge_vertices,
    );
    TestTrialNumericalQuadratureDefinition {
        dim: *dim,
        order: *order,
        npoints: output_weights.len(),
        weights: output_weights,
        test_points: test_output_points,
        trial_points: trial_output_points,
    }
}
fn edge_adjacent_quadrilateral_triangle(
    interval_rule: &NumericalQuadratureDefinition,
    test_singular_edge_vertices: (usize, usize),
    trial_singular_edge_vertices: (usize, usize),
) -> TestTrialNumericalQuadratureDefinition {
    let NumericalQuadratureDefinition {
        dim,
        order,
        npoints: _,
        weights: _,
        points: _,
    } = interval_rule;
    let (output_weights, trial_output_points, test_output_points) = tri_quad_edge_points(
        interval_rule,
        trial_singular_edge_vertices,
        test_singular_edge_vertices,
    );
    TestTrialNumericalQuadratureDefinition {
        dim: *dim,
        order: *order,
        npoints: output_weights.len(),
        weights: output_weights,
        test_points: test_output_points,
        trial_points: trial_output_points,
    }
}

fn tri_quad_vertex_points(
    interval_rule: &NumericalQuadratureDefinition,
    triangle_vertex: usize,
    quadrilateral_vertex: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let NumericalQuadratureDefinition {
        dim: _,
        order: _,
        npoints,
        weights,
        points,
    } = interval_rule;

    let n_output_points = 3 * npoints * npoints * npoints * npoints;

    let mut quadrilateral_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut triangle_output_points = Vec::<f64>::with_capacity(2 * n_output_points);
    let mut output_weights = Vec::<f64>::with_capacity(n_output_points);

    for index1 in 0..*npoints {
        for index2 in 0..*npoints {
            for index3 in 0..*npoints {
                for index4 in 0..*npoints {
                    let eta1 = points[index1];
                    let eta2 = points[index2];
                    let eta3 = points[index3];
                    let xi = points[index4];

                    // First part

                    let weight = weights[index1]
                        * weights[index2]
                        * weights[index3]
                        * weights[index4]
                        * xi
                        * xi
                        * xi;

                    quadrilateral_output_points.push(xi);
                    quadrilateral_output_points.push(xi * eta1);
                    triangle_output_points.push(xi * eta2);
                    triangle_output_points.push(xi * eta2 * eta3);
                    output_weights.push(weight * eta2);

                    // Second part

                    quadrilateral_output_points.push(xi * eta1);
                    quadrilateral_output_points.push(xi);
                    triangle_output_points.push(xi * eta2);
                    triangle_output_points.push(xi * eta2 * eta3);
                    output_weights.push(weight * eta2);

                    // Third part

                    quadrilateral_output_points.push(xi * eta1);
                    quadrilateral_output_points.push(xi * eta2);
                    triangle_output_points.push(xi);
                    triangle_output_points.push(xi * eta3);
                    output_weights.push(weight);
                }
            }
        }
    }

    transform_coords(
        &mut triangle_output_points,
        &create_triangle_mapper(triangle_vertex, next_triangle_vertex(triangle_vertex)),
    );
    transform_coords(
        &mut quadrilateral_output_points,
        &create_quadrilateral_mapper(
            quadrilateral_vertex,
            next_quadrilateral_vertex(quadrilateral_vertex),
        ),
    );

    (
        output_weights,
        triangle_output_points,
        quadrilateral_output_points,
    )
}

fn vertex_adjacent_triangle_quadrilateral(
    interval_rule: &NumericalQuadratureDefinition,
    test_singular_vertex: usize,
    trial_singular_vertex: usize,
) -> TestTrialNumericalQuadratureDefinition {
    let NumericalQuadratureDefinition {
        dim,
        order,
        npoints: _,
        weights: _,
        points: _,
    } = interval_rule;
    let (output_weights, test_output_points, trial_output_points) =
        tri_quad_vertex_points(interval_rule, test_singular_vertex, trial_singular_vertex);
    TestTrialNumericalQuadratureDefinition {
        dim: *dim,
        order: *order,
        npoints: output_weights.len(),
        weights: output_weights,
        test_points: test_output_points,
        trial_points: trial_output_points,
    }
}

fn vertex_adjacent_quadrilateral_triangle(
    interval_rule: &NumericalQuadratureDefinition,
    test_singular_vertex: usize,
    trial_singular_vertex: usize,
) -> TestTrialNumericalQuadratureDefinition {
    let NumericalQuadratureDefinition {
        dim,
        order,
        npoints: _,
        weights: _,
        points: _,
    } = interval_rule;
    let (output_weights, trial_output_points, test_output_points) =
        tri_quad_vertex_points(interval_rule, trial_singular_vertex, test_singular_vertex);
    TestTrialNumericalQuadratureDefinition {
        dim: *dim,
        order: *order,
        npoints: output_weights.len(),
        weights: output_weights,
        test_points: test_output_points,
        trial_points: trial_output_points,
    }
}

/// Create a Duffy rule on a triangle
pub fn triangle_quadrilateral_duffy(
    connectivity: &CellToCellConnectivity,
    npoints: usize,
) -> Result<TestTrialNumericalQuadratureDefinition, QuadratureError> {
    let rule = simplex_rule(ReferenceCellType::Interval, npoints)?;

    match connectivity.connectivity_dimension {
        0 => {
            // Cells have adjacent vertex
            if connectivity.local_indices.len() != 1 {
                Err(QuadratureError::ConnectivityError)
            } else {
                let (test_singular_vertex, trial_singular_vertex) = connectivity.local_indices[0];
                Ok(vertex_adjacent_triangle_quadrilateral(
                    &rule,
                    test_singular_vertex,
                    trial_singular_vertex,
                ))
            }
        }
        1 => {
            // Cells have adjacent edge
            if connectivity.local_indices.len() != 2 {
                Err(QuadratureError::ConnectivityError)
            } else {
                let first_pair = connectivity.local_indices[0];
                let second_pair = connectivity.local_indices[1];

                let test_singular_edge = (first_pair.0, second_pair.0);
                let trial_singular_edge = (first_pair.1, second_pair.1);
                Ok(edge_adjacent_triangle_quadrilateral(
                    &rule,
                    test_singular_edge,
                    trial_singular_edge,
                ))
            }
        }
        _ => Err(QuadratureError::ConnectivityError),
    }
}

/// Create a Duffy rule on a triangle
pub fn quadrilateral_triangle_duffy(
    connectivity: &CellToCellConnectivity,
    npoints: usize,
) -> Result<TestTrialNumericalQuadratureDefinition, QuadratureError> {
    let rule = simplex_rule(ReferenceCellType::Interval, npoints)?;

    match connectivity.connectivity_dimension {
        0 => {
            // Cells have adjacent vertex
            if connectivity.local_indices.len() != 1 {
                Err(QuadratureError::ConnectivityError)
            } else {
                let (test_singular_vertex, trial_singular_vertex) = connectivity.local_indices[0];
                Ok(vertex_adjacent_quadrilateral_triangle(
                    &rule,
                    test_singular_vertex,
                    trial_singular_vertex,
                ))
            }
        }
        1 => {
            // Cells have adjacent edge
            if connectivity.local_indices.len() != 2 {
                Err(QuadratureError::ConnectivityError)
            } else {
                let first_pair = connectivity.local_indices[0];
                let second_pair = connectivity.local_indices[1];

                let test_singular_edge = (first_pair.0, second_pair.0);
                let trial_singular_edge = (first_pair.1, second_pair.1);
                Ok(edge_adjacent_quadrilateral_triangle(
                    &rule,
                    test_singular_edge,
                    trial_singular_edge,
                ))
            }
        }
        _ => Err(QuadratureError::ConnectivityError),
    }
}
