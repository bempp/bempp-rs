//! Type definitions.
use std::fmt;

/// Invalid quadrature rule
#[derive(Debug)]
pub struct InvalidQuadrature;

impl fmt::Display for InvalidQuadrature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid quadrature")
    }
}

impl std::error::Error for InvalidQuadrature {}

/// Quadrature error
#[derive(Debug)]
pub enum QuadratureError {
    /// Rule not found
    RuleNotFound,
    /// Connectivity error
    ConnectivityError,
}

/// Definition of a numerical quadrature rule.
pub struct NumericalQuadratureDefinition {
    /// The dimension d of a single point.
    pub dim: usize,

    /// The order of the quadrature rule.
    pub order: usize,

    /// The number of points of the quadrature rule.
    pub npoints: usize,

    /// The weights of the quadrature rule.
    pub weights: Vec<f64>,
    /// The point coordinates of the quadrature rule.
    ///
    /// A single point has the coordinates p_1, p_2, ..., p_d,
    /// with d being the dimension of the point (typically, 1, 2, or 3).
    /// The vector points stores all points in consecutive order.
    /// Hence, the first point starts at position zero, the second point at
    /// position d, and the third point at position 2d.
    pub points: Vec<f64>,
}

/// Definition of a quadrature rule for double test/trial integrals.
///
/// This is necessary in cases where such integrals cannot be evaluated via
/// tensor application of rules for one simplex, such as for integration of
/// weak singularities (e.g. Duffy transformation rules).
pub struct TestTrialNumericalQuadratureDefinition {
    /// The dimension d of a single point.
    pub dim: usize,

    /// The order of the quadrature rule.
    pub order: usize,

    /// The number of points of the quadrature rule.
    pub npoints: usize,

    /// The weights of the quadrature rule.
    pub weights: Vec<f64>,
    /// The test point coordinates of the quadrature rule.
    ///
    /// A single point has the coordinates p_1, p_2, ..., p_d,
    /// with d being the dimension of the point (typically, 1, 2, or 3).
    /// The vector points stores all points in consecutive order.
    /// Hence, the first point starts at position zero, the second point at
    /// position d, and the third point at position 2d.
    pub test_points: Vec<f64>,

    /// The trial point coordinates of the quadrature rule.
    ///
    /// A single point has the coordinates p_1, p_2, ..., p_d,
    /// with d being the dimension of the point (typically, 1, 2, or 3).
    /// The vector points stores all points in consecutive order.
    /// Hence, the first point starts at position zero, the second point at
    /// position d, and the third point at position 2d.
    pub trial_points: Vec<f64>,
}

/// Storage for connectivity information.
///
/// Connectivity is important for many singular
/// quadrature rules. We need to know how cells are
/// connected to each other.
pub struct CellToCellConnectivity {
    /// Describe the dimension of the entity that
    /// connects the two cells (0 for point, 1 for edge, etc.)
    pub connectivity_dimension: usize,

    /// The local indices for the shared entity,
    /// for example the first edge of one triangle
    /// could be shared with the second edge of the neighboring
    /// triangle. The local indices are vectors for each entity
    /// to describe that the connection is possible through multiple
    /// entities, e.g. if cell to cell connectivity is described via
    /// matched up vertices.
    pub local_indices: Vec<(usize, usize)>,
}

/// A trait for obtaining a numerical quadrature rule. Depending
/// on a specified number of points a `NumericalQuadratureDefinition` is
/// returned with the weights and points of the rule.
pub trait NumericalQuadratureGenerator {
    /// Return the quadrature rule for a given order.
    fn get_rule(&self, npoints: usize) -> Result<NumericalQuadratureDefinition, InvalidQuadrature>;
}

/// A trait for singular quadrature rules. These are rules that
/// depend on the connectivity information between two cells.
/// So we need to supply the `Connectivity` structure. The result
/// is two separate quadrature containers for the two cells that
/// are integrated against each other.
pub trait SingularQuadratureGenerator {
    /// Return the quadrature rule for two cells.
    ///
    /// The method takes an `order` parameter and `connectivity` information
    /// that specifies how the two cells are linked to each other.
    fn get_singular_rule(
        &self,
        order: usize,
        connectivity: CellToCellConnectivity,
    ) -> Result<(NumericalQuadratureDefinition, NumericalQuadratureDefinition), InvalidQuadrature>;
}
