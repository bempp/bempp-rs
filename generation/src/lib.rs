extern crate proc_macro;
use bempp_element::element::create_element;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_quadrature::types::{
    CellToCellConnectivity, NumericalQuadratureDefinition, TestTrialNumericalQuadratureDefinition,
};
use bempp_tools::arrays::{Array2D, Array4D};
use bempp_traits::arrays::{Array2DAccess, Array4DAccess};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::ElementFamily;
use bempp_traits::element::FiniteElement;
use num::traits::real::Real;
use num::Num;
use proc_macro::TokenStream;
use quote::quote;
use std::collections::HashMap;
use std::fmt::Debug;

use syn::{parse::Parse, parse_macro_input, Expr, Token};

struct GenerationInput {
    kernel_name: Expr,
    _comma0: Token![,],
    test_element_family: Expr,
    _comma1: Token![,],
    test_element_cell: Expr,
    _comma2: Token![,],
    test_element_degree: Expr,
    _comma3: Token![,],
    test_element_discontinuous: Expr,
    _comma4: Token![,],
    trial_element_family: Expr,
    _comma5: Token![,],
    trial_element_cell: Expr,
    _comma6: Token![,],
    trial_element_degree: Expr,
    _comma7: Token![,],
    trial_element_discontinuous: Expr,
    _comma8: Token![,],
    test_geometry_element_family: Expr,
    _comma9: Token![,],
    test_geometry_element_cell: Expr,
    _comma10: Token![,],
    test_geometry_element_degree: Expr,
    _comma11: Token![,],
    test_geometry_element_discontinuous: Expr,
    _comma12: Token![,],
    trial_geometry_element_family: Expr,
    _comma13: Token![,],
    trial_geometry_element_cell: Expr,
    _comma14: Token![,],
    trial_geometry_element_degree: Expr,
    _comma15: Token![,],
    trial_geometry_element_discontinuous: Expr,
}

impl Parse for GenerationInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            kernel_name: input.parse()?,
            _comma0: input.parse()?,
            test_element_family: input.parse()?,
            _comma1: input.parse()?,
            test_element_cell: input.parse()?,
            _comma2: input.parse()?,
            test_element_degree: input.parse()?,
            _comma3: input.parse()?,
            test_element_discontinuous: input.parse()?,
            _comma4: input.parse()?,
            trial_element_family: input.parse()?,
            _comma5: input.parse()?,
            trial_element_cell: input.parse()?,
            _comma6: input.parse()?,
            trial_element_degree: input.parse()?,
            _comma7: input.parse()?,
            trial_element_discontinuous: input.parse()?,
            _comma8: input.parse()?,
            test_geometry_element_family: input.parse()?,
            _comma9: input.parse()?,
            test_geometry_element_cell: input.parse()?,
            _comma10: input.parse()?,
            test_geometry_element_degree: input.parse()?,
            _comma11: input.parse()?,
            test_geometry_element_discontinuous: input.parse()?,
            _comma12: input.parse()?,
            trial_geometry_element_family: input.parse()?,
            _comma13: input.parse()?,
            trial_geometry_element_cell: input.parse()?,
            _comma14: input.parse()?,
            trial_geometry_element_degree: input.parse()?,
            _comma15: input.parse()?,
            trial_geometry_element_discontinuous: input.parse()?,
        })
    }
}

fn parse_family(family: &Expr) -> ElementFamily {
    let family_str = format!("{}", quote! { #family });
    if family_str == "\"Lagrange\"" {
        ElementFamily::Lagrange
    } else if family_str == "\"Raviart-Thomas\"" {
        ElementFamily::RaviartThomas
    } else {
        panic!("Unsupported element family: {}", family_str);
    }
}

fn parse_cell(cell: &Expr) -> ReferenceCellType {
    let cell_str = format!("{}", quote! { #cell });
    if cell_str == "\"Triangle\"" {
        ReferenceCellType::Triangle
    } else if cell_str == "\"Quadrilateral\"" {
        ReferenceCellType::Quadrilateral
    } else {
        panic!("Unsupported cell type: {}", cell_str);
    }
}

fn parse_int(int: &Expr) -> usize {
    format!("{}", quote! { #int }).parse().unwrap()
}

fn parse_bool(bool: &Expr) -> bool {
    format!("{}", quote! { #bool }).parse().unwrap()
}
fn parse_string(string: &Expr) -> String {
    format!("{}", quote! { #string })
}

fn format_table<T: Num + Debug>(name: String, table: &[T]) -> String {
    let mut t = String::new();
    t += &format!(
        "const {name}: [{}; {}] = [",
        std::any::type_name::<T>(),
        table.len()
    );
    for w in table {
        t += &format!("{w:?}, ");
    }
    t += "];\n";
    t
}

fn format_eval_table<T: Num + Debug>(
    name: String,
    tables: &HashMap<usize, Array4D<T>>,
    deriv: usize,
    component: usize,
    level: usize,
) -> String {
    let mut t = String::new();
    if tables.len() == 1 {
        if let Some((_qid, table)) = tables.iter().next() {
            t += &indent(level);
            t += &format!(
                "const {name}: [{}; {}] = [",
                std::any::type_name::<T>(),
                table.shape().1 * table.shape().2
            );
            for i in 0..table.shape().1 {
                for j in 0..table.shape().2 {
                    if i > 0 || j > 0 {
                        t += ", ";
                    }
                    t += &format!("{:?}", table.get(deriv, i, j, component).unwrap());
                }
            }
            t += "];\n";
        }
    } else {
        t += &indent(level);
        t += &format!("let {name} = match qid {{\n");
        for (qid, table) in tables.iter() {
            t += &indent(level + 1);
            t += &format!("{qid} => [");
            for i in 0..table.shape().1 {
                for j in 0..table.shape().2 {
                    if i > 0 || j > 0 {
                        t += ", ";
                    }
                    t += &format!("{:?}", table.get(deriv, i, j, component).unwrap());
                }
            }
            t += "],\n";
        }
        t += &indent(level + 1);
        t += "_ => { panic!(\"Unrecognised quadrature id.\"); },\n";
        t += &indent(level);
        t += "};\n";
    }
    t
}

fn linear_derivative<T: Num + Debug + Real>(
    name: String,
    vertices: &String,
    gdim: usize,
    table: &Array4D<T>,
    deriv: usize,
) -> String {
    let mut code = String::new();
    code += &format!("let {name} = (");
    for d in 0..gdim {
        if d > 0 {
            code += ", ";
        }
        let mut started = false;
        for f in 0..table.shape().2 {
            let value = table.get(deriv, 0, f, 0).unwrap();
            if value.abs() > (T::one() + T::one()).powi(-15) {
                if (*value - T::one()).abs() > (T::one() + T::one()).powi(-15) {
                    // value is 1
                    if started {
                        code += " + ";
                    }
                    code += &format!("{vertices}[{}]", gdim * f + d);
                } else if (*value + T::one()).abs() > (T::one() + T::one()).powi(-15) {
                    // value is -1
                    code += " - ";
                    code += &format!("{vertices}[{}]", gdim * f + d);
                } else {
                    if started {
                        code += " + ";
                    }
                    code += &format!("{:?} * {vertices}[{}]", value, gdim * f + d);
                }
                started = true;
            }
        }
    }
    code += ");\n";
    code
}

fn linear_jacobian<T: Num + Debug + Real>(
    name: String,
    vertices: String,
    tdim: usize,
    gdim: usize,
    tables: &HashMap<usize, Array4D<T>>,
    level: usize,
) -> String {
    let mut code = String::new();
    if let Some((_qid, table)) = tables.iter().next() {
        if tdim == 2 && gdim == 3 {
            code += &indent(level);
            code += &linear_derivative("dx".to_string(), &vertices, gdim, table, 1);
            code += &indent(level);
            code += &linear_derivative("dy".to_string(), &vertices, gdim, table, 2);
            code += &indent(level);
            code += "let j = (dx.1 * dy.2 - dx.2 * dy.1, dx.2 * dy.0 - dx.0 * dy.2, dx.0 * dy.1 - dx.1 * dy.0);\n";
            code += &indent(level);
            code += &format!("let {name} = (j.0.powi(2) + j.1.powi(2) + j.2.powi(2)).sqrt();\n");
        } else {
            panic!(
                "Jacobian computation not implemented for tdim {} and gdim {}.",
                tdim, gdim
            );
        }
    }
    code
}

fn derivative(
    name: String,
    vertices: &String,
    gdim: usize,
    table: &String,
    point: &String,
    basis_count: usize,
) -> String {
    let mut code = String::new();
    code += &format!("let {name} = (");
    for d in 0..gdim {
        if d > 0 {
            code += ", ";
        }
        let mut started = false;
        for f in 0..basis_count {
            if started {
                code += " + ";
            }
            code += &format!(
                "{table}[{point} * {basis_count} + {f}] * {vertices}[{}]",
                gdim * f + d
            );
            started = true;
        }
    }
    code += ");\n";
    code
}

#[allow(clippy::too_many_arguments)]
fn jacobian(
    name: String,
    vertices: String,
    tdim: usize,
    gdim: usize,
    dx: String,
    dy: String,
    point: String,
    basis_count: usize,
    level: usize,
) -> String {
    let mut code = String::new();
    if tdim == 2 && gdim == 3 {
        code += &indent(level);
        code += &derivative("dx".to_string(), &vertices, gdim, &dx, &point, basis_count);
        code += &indent(level);
        code += &derivative("dy".to_string(), &vertices, gdim, &dy, &point, basis_count);
        code += &indent(level);
        code += "let j = (dx.1 * dy.2 - dx.2 * dy.1, dx.2 * dy.0 - dx.0 * dy.2, dx.0 * dy.1 - dx.1 * dy.0);\n";
        code += &indent(level);
        code += &format!("let {name} = (j.0.powi(2) + j.1.powi(2) + j.2.powi(2)).sqrt();\n");
    } else {
        panic!(
            "Jacobian computation not implemented for tdim {} and gdim {}.",
            tdim, gdim
        );
    }
    code
}

#[allow(clippy::too_many_arguments)]
fn singular_kernel(
    name: String,
    quadrules: HashMap<usize, TestTrialNumericalQuadratureDefinition>,
    test_element: &impl FiniteElement,
    trial_element: &impl FiniteElement,
    test_geometry_element: &impl FiniteElement,
    trial_geometry_element: &impl FiniteElement,
    tdim: usize,
    gdim: usize,
) -> String {
    // TODO: make this into an input
    type T = f64;

    let typename = std::any::type_name::<T>().to_string();
    let mut npts = 0;
    if let Some((_qid, qrule)) = quadrules.iter().next() {
        npts = qrule.npoints;
    }
    let mut test_points = Array2D::<T>::new((npts, tdim));
    let mut trial_points = Array2D::<T>::new((npts, tdim));

    let mut test_evals = HashMap::new();
    let mut trial_evals = HashMap::new();
    let mut test_geometry_evals = HashMap::new();
    let mut trial_geometry_evals = HashMap::new();

    for (qid, qrule) in quadrules.iter() {
        for p in 0..npts {
            for d in 0..tdim {
                *test_points.get_mut(p, d).unwrap() = qrule.test_points[p * tdim + d];
                *trial_points.get_mut(p, d).unwrap() = qrule.trial_points[p * tdim + d];
            }
        }
        let mut test_eval_table = Array4D::<T>::new(test_element.tabulate_array_shape(1, npts));
        test_element.tabulate(&test_points, 1, &mut test_eval_table);
        test_evals.insert(*qid, test_eval_table);
        let mut trial_eval_table = Array4D::<T>::new(trial_element.tabulate_array_shape(1, npts));
        trial_element.tabulate(&trial_points, 1, &mut trial_eval_table);
        trial_evals.insert(*qid, trial_eval_table);
        let mut test_geometry_eval_table =
            Array4D::<T>::new(test_geometry_element.tabulate_array_shape(1, npts));
        test_geometry_element.tabulate(&test_points, 1, &mut test_geometry_eval_table);
        test_geometry_evals.insert(*qid, test_geometry_eval_table);
        let mut trial_geometry_eval_table =
            Array4D::<T>::new(trial_geometry_element.tabulate_array_shape(1, npts));
        trial_geometry_element.tabulate(&trial_points, 1, &mut trial_geometry_eval_table);
        trial_geometry_evals.insert(*qid, trial_geometry_eval_table);
    }

    let mut code = String::new();
    // Write const tables
    code += &format!("fn {name}(&self, result: &mut [{typename}], test_vertices: &[{typename}], trial_vertices: &[{typename}]");
    if quadrules.len() > 1 {
        code += ", qid: usize";
    }
    code += ") {\n";
    code += &indent(1);
    if let Some((_qid, qrule)) = quadrules.iter().next() {
        code += &format_table("WTS".to_string(), &qrule.weights);
    }
    code += &format_eval_table(
        "TEST_GEOMETRY_EVALS".to_string(),
        &test_geometry_evals,
        0,
        0,
        1,
    );
    if test_geometry_element.degree() > 1 {
        code += &format_eval_table(
            "TEST_GEOMETRY_EVALS_DX".to_string(),
            &test_geometry_evals,
            1,
            0,
            1,
        );
        code += &format_eval_table(
            "TEST_GEOMETRY_EVALS_DY".to_string(),
            &test_geometry_evals,
            2,
            0,
            1,
        );
    }
    code += &format_eval_table(
        "TRIAL_GEOMETRY_EVALS".to_string(),
        &trial_geometry_evals,
        0,
        0,
        1,
    );
    if trial_geometry_element.degree() > 1 {
        code += &format_eval_table(
            "TRIAL_GEOMETRY_EVALS_DX".to_string(),
            &trial_geometry_evals,
            1,
            0,
            1,
        );
        code += &format_eval_table(
            "TRIAL_GEOMETRY_EVALS_DY".to_string(),
            &trial_geometry_evals,
            2,
            0,
            1,
        );
    }
    if test_element.degree() > 0 {
        code += &format_eval_table("TEST_EVALS".to_string(), &test_evals, 0, 0, 1);
    }
    if trial_element.degree() > 0 {
        code += &format_eval_table("TRIAL_EVALS".to_string(), &trial_evals, 0, 0, 1);
    }
    code += &indent(1);
    code += "const ONE_OVER_4PI: f64 = 0.25 * std::f64::consts::FRAC_1_PI;\n";
    code += "\n";

    if test_geometry_element.degree() == 1 {
        code += &linear_jacobian(
            "test_jdet".to_string(),
            "test_vertices".to_string(),
            tdim,
            gdim,
            &test_geometry_evals,
            1,
        );
    }
    if trial_geometry_element.degree() == 1 {
        code += &linear_jacobian(
            "trial_jdet".to_string(),
            "trial_vertices".to_string(),
            tdim,
            gdim,
            &trial_geometry_evals,
            1,
        );
    }

    code += "\n";
    code += &indent(1);
    code += &format!("for i in 0..{} {{\n", test_element.dim());
    code += &indent(2);
    code += &format!("for j in 0..{} {{\n", trial_element.dim());
    code += &indent(3);
    code += &format!("result[i * {} + j] = 0.0;\n", trial_element.dim());
    code += &indent(2);
    code += "}\n";
    code += &indent(1);
    code += "}\n";

    code += &indent(1);
    code += &format!("for q in 0..{npts} {{\n");
    code += &indent(2);
    code += &format!("let mut sum_squares: {typename} = 0.0;\n");
    code += &indent(2);
    code += &format!("for d in 0..{gdim} {{\n");
    code += &indent(3);
    code += "let x = ";
    for b in 0..test_geometry_element.dim() {
        if b == 0 {
            code += &format!(
                "test_vertices[d] * TEST_GEOMETRY_EVALS[q * {}]",
                test_geometry_element.dim()
            );
        } else {
            code += &format!(
                " + test_vertices[{} + d] * TEST_GEOMETRY_EVALS[q * {} + {b}]",
                b * gdim,
                test_geometry_element.dim()
            );
        }
    }
    code += ";\n";
    code += &indent(3);
    code += "let y = ";
    for b in 0..trial_geometry_element.dim() {
        if b == 0 {
            code += &format!(
                "trial_vertices[d] * TRIAL_GEOMETRY_EVALS[q * {}]",
                trial_geometry_element.dim()
            );
        } else {
            code += &format!(
                " + trial_vertices[{} + d] * TRIAL_GEOMETRY_EVALS[q * {} + {b}]",
                b * gdim,
                trial_geometry_element.dim()
            );
        }
    }
    code += ";\n";
    code += &indent(3);
    code += "sum_squares += (x - y).powi(2);\n";
    code += &indent(2);
    code += "}\n";
    code += &indent(2);
    code += "let distance = sum_squares.sqrt();\n";
    if test_geometry_element.degree() > 1 {
        code += &jacobian(
            "test_jdet".to_string(),
            "test_vertices".to_string(),
            tdim,
            gdim,
            "TEST_GEOMETRY_EVALS_DX".to_string(),
            "TEST_GEOMETRY_EVALS_DY".to_string(),
            "q".to_string(),
            test_geometry_element.dim(),
            2,
        );
    }
    if trial_geometry_element.degree() > 1 {
        code += &jacobian(
            "trial_jdet".to_string(),
            "trial_vertices".to_string(),
            tdim,
            gdim,
            "TRIAL_GEOMETRY_EVALS_DX".to_string(),
            "TRIAL_GEOMETRY_EVALS_DY".to_string(),
            "q".to_string(),
            trial_geometry_element.dim(),
            2,
        );
    }
    code += &indent(2);
    code += &format!("for i in 0..{} {{\n", test_element.dim());
    code += &indent(3);
    code += &format!("for j in 0..{} {{\n", trial_element.dim());
    code += &indent(4);
    code += &format!("result[i * {} + j] += WTS[q]", trial_element.dim());
    if test_element.degree() > 0 {
        code += &format!(" * TEST_EVALS[q * {} + i]", test_element.dim());
    }
    if trial_element.degree() > 0 {
        code += &format!(" * TRIAL_EVALS[q * {} + j]", trial_element.dim());
    }
    code += " * test_jdet * trial_jdet * ONE_OVER_4PI / distance;\n";
    code += &indent(3);
    code += "}\n";
    code += &indent(2);
    code += "}\n";
    code += &indent(1);
    code += "}\n";

    code += "}\n\n";
    code
}

#[allow(clippy::too_many_arguments)]
fn nonsingular_kernel(
    name: String,
    test_rule: NumericalQuadratureDefinition,
    trial_rule: NumericalQuadratureDefinition,
    test_element: &impl FiniteElement,
    trial_element: &impl FiniteElement,
    test_geometry_element: &impl FiniteElement,
    trial_geometry_element: &impl FiniteElement,
    tdim: usize,
    gdim: usize,
) -> String {
    // TODO: make this into an input
    type T = f64;

    let typename = std::any::type_name::<T>().to_string();
    let test_npts = test_rule.npoints;
    let trial_npts = trial_rule.npoints;
    let test_points = Array2D::<T>::from_data(test_rule.points, (test_npts, tdim));
    let trial_points = Array2D::<T>::from_data(trial_rule.points, (trial_npts, tdim));

    let mut test_table = Array4D::<T>::new(test_element.tabulate_array_shape(1, test_npts));
    test_element.tabulate(&test_points, 1, &mut test_table);
    let mut trial_table = Array4D::<T>::new(trial_element.tabulate_array_shape(1, trial_npts));
    trial_element.tabulate(&trial_points, 1, &mut trial_table);

    let mut test_geometry_table =
        Array4D::<T>::new(test_geometry_element.tabulate_array_shape(1, test_npts));
    test_geometry_element.tabulate(&test_points, 1, &mut test_geometry_table);
    let mut trial_geometry_table =
        Array4D::<T>::new(trial_geometry_element.tabulate_array_shape(1, trial_npts));
    trial_geometry_element.tabulate(&trial_points, 1, &mut trial_geometry_table);

    let mut test_evals = HashMap::new();
    test_evals.insert(0, test_table);
    let mut test_geometry_evals = HashMap::new();
    test_geometry_evals.insert(0, test_geometry_table);
    let mut trial_evals = HashMap::new();
    trial_evals.insert(0, trial_table);
    let mut trial_geometry_evals = HashMap::new();
    trial_geometry_evals.insert(0, trial_geometry_table);

    let mut code = String::new();
    // Write const tables
    code += &format!("fn {name}(&self, result: &mut [{typename}], test_vertices: &[{typename}], trial_vertices: &[{typename}]) {{\n");

    code += &indent(1);
    code += &format_table("TEST_WTS".to_string(), &test_rule.weights);
    code += &indent(1);
    code += &format_table("TRIAL_WTS".to_string(), &trial_rule.weights);
    code += &format_eval_table(
        "TEST_GEOMETRY_EVALS".to_string(),
        &test_geometry_evals,
        0,
        0,
        1,
    );
    if test_geometry_element.degree() > 1 {
        code += &format_eval_table(
            "TEST_GEOMETRY_EVALS_DX".to_string(),
            &test_geometry_evals,
            1,
            0,
            1,
        );
        code += &format_eval_table(
            "TEST_GEOMETRY_EVALS_DY".to_string(),
            &test_geometry_evals,
            2,
            0,
            1,
        );
    }
    code += &format_eval_table(
        "TRIAL_GEOMETRY_EVALS".to_string(),
        &trial_geometry_evals,
        0,
        0,
        1,
    );
    if trial_geometry_element.degree() > 1 {
        code += &format_eval_table(
            "TRIAL_GEOMETRY_EVALS_DX".to_string(),
            &trial_geometry_evals,
            1,
            0,
            1,
        );
        code += &format_eval_table(
            "TRIAL_GEOMETRY_EVALS_DY".to_string(),
            &trial_geometry_evals,
            2,
            0,
            1,
        );
    }
    if test_element.degree() > 0 {
        code += &format_eval_table("TEST_EVALS".to_string(), &test_evals, 0, 0, 1);
    }
    if trial_element.degree() > 0 {
        code += &format_eval_table("TRIAL_EVALS".to_string(), &trial_evals, 0, 0, 1);
    }
    code += &indent(1);
    code += "const ONE_OVER_4PI: f64 = 0.25 * std::f64::consts::FRAC_1_PI;\n";
    code += "\n";

    if test_geometry_element.degree() == 1 {
        code += &linear_jacobian(
            "test_jdet".to_string(),
            "test_vertices".to_string(),
            tdim,
            gdim,
            &test_geometry_evals,
            1,
        );
    }
    if trial_geometry_element.degree() == 1 {
        code += &linear_jacobian(
            "trial_jdet".to_string(),
            "trial_vertices".to_string(),
            tdim,
            gdim,
            &trial_geometry_evals,
            1,
        );
    }

    code += "\n";
    code += &indent(1);
    code += &format!("for i in 0..{} {{\n", test_element.dim());
    code += &indent(2);
    code += &format!("for j in 0..{} {{\n", trial_element.dim());
    code += &indent(3);
    code += &format!("result[i * {} + j] = 0.0;\n", trial_element.dim());
    code += &indent(2);
    code += "}\n";
    code += &indent(1);
    code += "}\n";

    code += &indent(1);
    code += &format!("for test_q in 0..{test_npts} {{\n");
    if test_geometry_element.degree() > 1 {
        code += &jacobian(
            "test_jdet".to_string(),
            "test_vertices".to_string(),
            tdim,
            gdim,
            "TEST_GEOMETRY_EVALS_DX".to_string(),
            "TEST_GEOMETRY_EVALS_DY".to_string(),
            "test_q".to_string(),
            test_geometry_element.dim(),
            2,
        );
    }
    code += &indent(2);
    code += &format!("for trial_q in 0..{trial_npts} {{\n");
    code += &indent(3);
    code += &format!("let mut sum_squares: {typename} = 0.0;\n");
    code += &indent(3);
    code += &format!("for d in 0..{gdim} {{\n");
    code += &indent(4);
    code += "let x = ";
    for b in 0..test_geometry_element.dim() {
        if b == 0 {
            code += &format!(
                "test_vertices[d] * TEST_GEOMETRY_EVALS[test_q * {}]",
                test_geometry_element.dim()
            );
        } else {
            code += &format!(
                " + test_vertices[{} + d] * TEST_GEOMETRY_EVALS[test_q * {} + {b}]",
                b * gdim,
                test_geometry_element.dim()
            );
        }
    }
    code += ";\n";
    code += &indent(4);
    code += "let y = ";
    for b in 0..trial_geometry_element.dim() {
        if b == 0 {
            code += &format!(
                "trial_vertices[d] * TRIAL_GEOMETRY_EVALS[trial_q * {}]",
                trial_geometry_element.dim()
            );
        } else {
            code += &format!(
                " + trial_vertices[{} + d] * TRIAL_GEOMETRY_EVALS[trial_q * {} + {b}]",
                b * gdim,
                trial_geometry_element.dim()
            );
        }
    }
    code += ";\n";
    code += &indent(4);
    code += "sum_squares += (x - y).powi(2);\n";
    code += &indent(3);
    code += "}\n";
    code += &indent(3);
    code += "let distance = sum_squares.sqrt();\n";
    if trial_geometry_element.degree() > 1 {
        code += &jacobian(
            "trial_jdet".to_string(),
            "trial_vertices".to_string(),
            tdim,
            gdim,
            "TRIAL_GEOMETRY_EVALS_DX".to_string(),
            "TRIAL_GEOMETRY_EVALS_DY".to_string(),
            "trial_q".to_string(),
            trial_geometry_element.dim(),
            3,
        );
    }
    code += &indent(3);
    code += &format!("for i in 0..{} {{\n", test_element.dim());
    code += &indent(4);
    code += &format!("for j in 0..{} {{\n", trial_element.dim());
    code += &indent(5);
    code += &format!(
        "result[i * {} + j] += TEST_WTS[test_q] * TRIAL_WTS[trial_q]",
        trial_element.dim()
    );
    if test_element.degree() > 0 {
        code += &format!(" * TEST_EVALS[test_q * {} + i]", test_element.dim());
    }
    if trial_element.degree() > 0 {
        code += &format!(" * TRIAL_EVALS[trial_q * {} + j]", trial_element.dim());
    }
    code += " * test_jdet * trial_jdet * ONE_OVER_4PI / distance;\n";
    code += &indent(4);
    code += "}\n";
    code += &indent(3);
    code += "}\n";
    code += &indent(2);
    code += "}\n";
    code += &indent(1);
    code += "}\n";
    code += "}\n\n";
    code
}

fn indent(level: usize) -> String {
    let mut t = String::new();
    for _ in 0..level * 4 {
        t += " ";
    }
    t
}

#[proc_macro]
pub fn generate_kernels(input: TokenStream) -> TokenStream {
    // TODO: make these into inputs
    type T = f64;
    let tdim = 2;
    let gdim = 3;
    let singular_quad_degree = 3;
    let nonsingular_quad_degree = 7;

    let typename = std::any::type_name::<T>().to_string();
    let es = parse_macro_input!(input as GenerationInput);

    let kernel_name = parse_string(&es.kernel_name);

    let test_element = create_element(
        parse_family(&es.test_element_family),
        parse_cell(&es.test_element_cell),
        parse_int(&es.test_element_degree),
        parse_bool(&es.test_element_discontinuous),
    );
    let trial_element = create_element(
        parse_family(&es.trial_element_family),
        parse_cell(&es.trial_element_cell),
        parse_int(&es.trial_element_degree),
        parse_bool(&es.trial_element_discontinuous),
    );
    let test_geometry_element = create_element(
        parse_family(&es.test_geometry_element_family),
        parse_cell(&es.test_geometry_element_cell),
        parse_int(&es.test_geometry_element_degree),
        parse_bool(&es.test_geometry_element_discontinuous),
    );
    let trial_geometry_element = create_element(
        parse_family(&es.trial_geometry_element_family),
        parse_cell(&es.trial_geometry_element_cell),
        parse_int(&es.trial_geometry_element_degree),
        parse_bool(&es.trial_geometry_element_discontinuous),
    );

    let mut code = String::new();
    code += &format!("struct _BemppKernel_{kernel_name} {{\n");
    code += "    test_element: bempp_element::element::CiarletElement,\n";
    code += "    trial_element: bempp_element::element::CiarletElement,\n";
    code += "    test_geometry_element: bempp_element::element::CiarletElement,\n";
    code += "    trial_geometry_element: bempp_element::element::CiarletElement,\n";
    code += "};\n\n";
    code +=
        &format!("impl bempp_traits::bem::Kernel<{typename}> for _BemppKernel_{kernel_name} {{");
    code += "\n\n";
    code += "fn test_element_dim(&self) -> usize { self.test_element.dim() }\n";
    code += "fn trial_element_dim(&self) -> usize { self.trial_element.dim() }\n";
    code += "fn test_geometry_element_dim(&self) -> usize { self.test_geometry_element.dim() }\n";
    code += "fn trial_geometry_element_dim(&self) -> usize { self.trial_geometry_element.dim() }\n";

    // TODO: quads
    let mut quadrules = HashMap::new();
    quadrules.insert(
        0,
        triangle_duffy(
            &CellToCellConnectivity {
                connectivity_dimension: 2,
                local_indices: vec![(0, 0), (1, 1), (2, 2)],
            },
            singular_quad_degree,
        )
        .unwrap(),
    );
    code += &singular_kernel(
        "same_cell_kernel".to_string(),
        quadrules,
        &test_element,
        &trial_element,
        &test_geometry_element,
        &trial_geometry_element,
        tdim,
        gdim,
    );

    let mut quadrules = HashMap::new();
    for i in 0..3 {
        for j in i + 1..3 {
            for k in 0..3 {
                for l in 0..3 {
                    if k != l {
                        quadrules.insert(
                            i * 64 + k * 16 + j * 4 + l,
                            triangle_duffy(
                                &CellToCellConnectivity {
                                    connectivity_dimension: 1,
                                    local_indices: vec![(i, k), (j, l)],
                                },
                                singular_quad_degree,
                            )
                            .unwrap(),
                        );
                    }
                }
            }
        }
    }
    code += &singular_kernel(
        "shared_edge_kernel".to_string(),
        quadrules,
        &test_element,
        &trial_element,
        &test_geometry_element,
        &trial_geometry_element,
        tdim,
        gdim,
    );

    let mut quadrules = HashMap::new();
    for i in 0..3 {
        for j in 0..3 {
            quadrules.insert(
                4 * j + i,
                triangle_duffy(
                    &CellToCellConnectivity {
                        connectivity_dimension: 0,
                        local_indices: vec![(i, j)],
                    },
                    singular_quad_degree,
                )
                .unwrap(),
            );
        }
    }
    code += &singular_kernel(
        "shared_vertex_kernel".to_string(),
        quadrules,
        &test_element,
        &trial_element,
        &test_geometry_element,
        &trial_geometry_element,
        tdim,
        gdim,
    );

    let test_rule = simplex_rule(test_element.cell_type(), nonsingular_quad_degree).unwrap();
    let trial_rule = simplex_rule(trial_element.cell_type(), nonsingular_quad_degree).unwrap();

    code += &nonsingular_kernel(
        "nonneighbour_kernel".to_string(),
        test_rule,
        trial_rule,
        &test_element,
        &trial_element,
        &test_geometry_element,
        &trial_geometry_element,
        tdim,
        gdim,
    );

    code += "}\n\n";
    code += &format!("let {kernel_name} = _BemppKernel_{kernel_name} {{\n");
    code += &format!("    test_element: bempp_element::element::create_element(bempp_traits::element::ElementFamily::{:?}, bempp_traits::cell::ReferenceCellType::{:?}, {}, {}),\n",
                     test_element.family(), test_element.cell_type(), test_element.degree(), test_element.discontinuous());
    code += &format!("    trial_element: bempp_element::element::create_element(bempp_traits::element::ElementFamily::{:?}, bempp_traits::cell::ReferenceCellType::{:?}, {}, {}),\n",
                     trial_element.family(), trial_element.cell_type(), trial_element.degree(), trial_element.discontinuous());
    code += &format!("    test_geometry_element: bempp_element::element::create_element(bempp_traits::element::ElementFamily::{:?}, bempp_traits::cell::ReferenceCellType::{:?}, {}, {}),\n",
                     test_geometry_element.family(), test_geometry_element.cell_type(), test_geometry_element.degree(), test_geometry_element.discontinuous());
    code += &format!("    trial_geometry_element: bempp_element::element::create_element(bempp_traits::element::ElementFamily::{:?}, bempp_traits::cell::ReferenceCellType::{:?}, {}, {}),\n",
                     trial_geometry_element.family(), trial_geometry_element.cell_type(), trial_geometry_element.degree(), trial_geometry_element.discontinuous());
    code += "};";
    code.parse().unwrap()
}
