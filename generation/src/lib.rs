//! Generation of boundary element kernels
#![cfg_attr(feature = "strict", deny(warnings))]

// TODO: make a struct that stores all information about the kernel, then look up (eg) tables names from it

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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
enum FunctionType {
    Test = 0,
    Trial = 1,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
enum ValueType {
    Value = 0,
    DX = 1,
    DY = 2,
}

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

fn assert_sizes(
    test_element: &impl FiniteElement,
    trial_element: &impl FiniteElement,
    test_geometry_element: &impl FiniteElement,
    trial_geometry_element: &impl FiniteElement,
    gdim: usize,
    level: usize,
) -> String {
    let mut code = String::new();
    code += &indent(level);
    code += &format!(
        "assert_eq!(result.len(), {});\n",
        trial_element.dim() * test_element.dim()
    );
    code += &indent(level);
    code += &format!(
        "assert_eq!(test_vertices.len(), {});\n",
        test_geometry_element.dim() * gdim
    );
    code += &indent(level);
    code += &format!(
        "assert_eq!(trial_vertices.len(), {});\n",
        trial_geometry_element.dim() * gdim
    );
    code
}

fn table<T: Num + Debug>(table: &[T]) -> String {
    let mut t = String::new();
    t += "[";
    for (i, w) in table.iter().enumerate() {
        if i > 0 {
            t += ", ";
        }
        t += &format!("{w:?}");
    }
    t += "]";
    t
}

fn eval_table_transpose<T: Num + Debug>(
    table: &Array4D<T>,
    deriv: usize,
    component: usize,
) -> String {
    let mut t = String::new();
    t += "[";
    for j in 0..table.shape().2 {
        for i in 0..table.shape().1 {
            if i > 0 || j > 0 {
                t += ", ";
            }
            t += &format!("{:?}", table.get(deriv, i, j, component).unwrap());
        }
    }
    t += "]";
    t
}

fn tables(
    test_element: &impl FiniteElement,
    trial_element: &impl FiniteElement,
    test_geometry_element: &impl FiniteElement,
    trial_geometry_element: &impl FiniteElement,
    test_points: &HashMap<usize, Array2D<f64>>,
    trial_points: &HashMap<usize, Array2D<f64>>,
    level: usize,
) -> String {
    type T = f64;
    let typename = std::any::type_name::<T>().to_string();

    let test_npts = if let Some((_qid, pts)) = test_points.iter().next() {
        pts.shape().0
    } else {
        panic!("No points.");
    };
    let trial_npts = if let Some((_qid, pts)) = trial_points.iter().next() {
        pts.shape().0
    } else {
        panic!("No points.");
    };

    let test_derivs = if test_geometry_element.degree() == 1 {
        0
    } else {
        1
    };
    let trial_derivs = if trial_geometry_element.degree() == 1 {
        0
    } else {
        1
    };

    let mut test_eval_table = Array4D::<T>::new(test_element.tabulate_array_shape(0, test_npts));
    let mut test_geometry_eval_table =
        Array4D::<T>::new(test_geometry_element.tabulate_array_shape(test_derivs, test_npts));
    let mut trial_eval_table = Array4D::<T>::new(trial_element.tabulate_array_shape(0, trial_npts));
    let mut trial_geometry_eval_table =
        Array4D::<T>::new(trial_geometry_element.tabulate_array_shape(trial_derivs, trial_npts));
    let mut out = String::new();
    if test_points.len() == 1 {
        if let Some((_qid, pts)) = test_points.iter().next() {
            test_element.tabulate(pts, 0, &mut test_eval_table);
            test_geometry_element.tabulate(pts, test_derivs, &mut test_geometry_eval_table);
        }
        if let Some((_qid, pts)) = trial_points.iter().next() {
            trial_element.tabulate(pts, 0, &mut trial_eval_table);
            trial_geometry_element.tabulate(pts, trial_derivs, &mut trial_geometry_eval_table);
        }
        out += &indent(level);
        out += &format!(
            "let TEST_EVALS: [{typename}; {}] = ",
            test_npts * test_element.dim()
        );
        out += &eval_table_transpose(&test_eval_table, 0, 0);
        out += ";\n";
        out += &indent(level);
        out += &format!(
            "let TEST_GEOMETRY_EVALS: [{typename}; {}] = ",
            test_npts * test_geometry_element.dim()
        );
        out += &eval_table_transpose(&test_geometry_eval_table, 0, 0);
        out += ";\n";
        if test_derivs > 0 {
            out += &indent(level);
            out += &format!(
                "let TEST_GEOMETRY_EVALS_DX: [{typename}; {}] = ",
                test_npts * test_geometry_element.dim()
            );
            out += &eval_table_transpose(&test_geometry_eval_table, 1, 0);
            out += ";\n";
            out += &indent(level);
            out += &format!(
                "let TEST_GEOMETRY_EVALS_DY: [{typename}; {}] = ",
                test_npts * test_geometry_element.dim()
            );
            out += &eval_table_transpose(&test_geometry_eval_table, 2, 0);
            out += ";\n";
        }
        out += &indent(level);
        out += &format!(
            "let TRIAL_EVALS: [{typename}; {}] = ",
            trial_npts * trial_element.dim()
        );
        out += &eval_table_transpose(&trial_eval_table, 0, 0);
        out += ";\n";
        out += &indent(level);
        out += &format!(
            "let TRIAL_GEOMETRY_EVALS: [{typename}; {}] = ",
            trial_npts * trial_geometry_element.dim()
        );
        out += &eval_table_transpose(&trial_geometry_eval_table, 0, 0);
        out += ";\n";
        if trial_derivs > 0 {
            out += &indent(level);
            out += &format!(
                "let TRIAL_GEOMETRY_EVALS_DX: [{typename}; {}] = ",
                trial_npts * trial_geometry_element.dim()
            );
            out += &eval_table_transpose(&trial_geometry_eval_table, 1, 0);
            out += ";\n";
            out += &indent(level);
            out += &format!(
                "let TRIAL_GEOMETRY_EVALS_DY: [{typename}; {}] = ",
                trial_npts * trial_geometry_element.dim()
            );
            out += &eval_table_transpose(&trial_geometry_eval_table, 2, 0);
            out += ";\n";
        }
    } else {
        out += &indent(level);
        out += &format!(
            "let TEST_EVALS: [{typename}; {}];\n",
            test_npts * test_element.dim()
        );
        out += &indent(level);
        out += &format!(
            "let TEST_GEOMETRY_EVALS: [{typename}; {}];\n",
            test_npts * test_geometry_element.dim()
        );
        if test_derivs > 0 {
            out += &indent(level);
            out += &format!(
                "let TEST_GEOMETRY_EVALS_DX: [{typename}; {}];\n",
                test_npts * test_geometry_element.dim()
            );
            out += &indent(level);
            out += &format!(
                "let TEST_GEOMETRY_EVALS_DY: [{typename}; {}];\n",
                test_npts * test_geometry_element.dim()
            );
        }
        out += &indent(level);
        out += &format!(
            "let TRIAL_EVALS: [{typename}; {}];\n",
            trial_npts * trial_element.dim()
        );
        out += &indent(level);
        out += &format!(
            "let TRIAL_GEOMETRY_EVALS: [{typename}; {}];\n",
            trial_npts * trial_geometry_element.dim()
        );
        if trial_derivs > 0 {
            out += &indent(level);
            out += &format!(
                "let TRIAL_GEOMETRY_EVALS_DX: [{typename}; {}];\n",
                trial_npts * trial_geometry_element.dim()
            );
            out += &indent(level);
            out += &format!(
                "let TRIAL_GEOMETRY_EVALS_DY: [{typename}; {}];\n",
                trial_npts * trial_geometry_element.dim()
            );
        }
        out += &indent(level);
        out += "match qid {\n";
        for (qid, test_pts) in test_points.iter() {
            let trial_pts = trial_points.get(qid).unwrap();
            test_element.tabulate(test_pts, test_derivs, &mut test_eval_table);
            test_geometry_element.tabulate(test_pts, test_derivs, &mut test_geometry_eval_table);
            trial_element.tabulate(trial_pts, trial_derivs, &mut trial_eval_table);
            trial_geometry_element.tabulate(
                trial_pts,
                trial_derivs,
                &mut trial_geometry_eval_table,
            );
            out += &indent(level + 1);
            out += &format!("{qid} => {{\n");
            out += &indent(level + 2);
            out += "TEST_EVALS = ";
            out += &eval_table_transpose(&test_eval_table, 0, 0);
            out += ";\n";
            out += &indent(level + 2);
            out += "TEST_GEOMETRY_EVALS = ";
            out += &eval_table_transpose(&test_geometry_eval_table, 0, 0);
            out += ";\n";
            if test_derivs > 0 {
                out += &indent(level + 2);
                out += "TEST_GEOMETRY_EVALS_DX = ";
                out += &eval_table_transpose(&test_geometry_eval_table, 1, 0);
                out += ";\n";
                out += &indent(level + 2);
                out += "TEST_GEOMETRY_EVALS_DY = ";
                out += &eval_table_transpose(&test_geometry_eval_table, 2, 0);
                out += ";\n";
            }
            out += &indent(level + 2);
            out += "TRIAL_EVALS = ";
            out += &eval_table_transpose(&trial_eval_table, 0, 0);
            out += ";\n";
            out += &indent(level + 2);
            out += "TRIAL_GEOMETRY_EVALS = ";
            out += &eval_table_transpose(&trial_geometry_eval_table, 0, 0);
            out += ";\n";
            if trial_derivs > 0 {
                out += &indent(level + 2);
                out += "TRIAL_GEOMETRY_EVALS_DX = ";
                out += &eval_table_transpose(&trial_geometry_eval_table, 1, 0);
                out += ";\n";
                out += &indent(level + 2);
                out += "TRIAL_GEOMETRY_EVALS_DY = ";
                out += &eval_table_transpose(&trial_geometry_eval_table, 2, 0);
                out += ";\n";
            }
            out += &indent(level + 1);
            out += "},\n";
        }
        out += &indent(level + 1);
        out += "_ => { panic!(\"Unsupported quadrature rule.\"); },\n";
        out += &indent(level);
        out += "}\n";
    }

    out
}

fn set_zero(len: usize, level: usize) -> String {
    let mut out = String::new();

    if len == 1 {
        out += &indent(level);
        out += "result[0] = 0.0;\n";
    } else {
        out += &indent(level);
        out += &format!("for i in 0..{len} {{\n");
        out += &indent(level + 1);
        out += "result[i] = 0.0;\n";
        out += &indent(level);
        out += "}\n";
    }
    out
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "slice-static-tables")]
fn take_slices(
    test_element: &impl FiniteElement,
    trial_element: &impl FiniteElement,
    test_geometry_element: &impl FiniteElement,
    trial_geometry_element: &impl FiniteElement,
    test_npts: usize,
    trial_npts: usize,
    gdim: usize,
    level: usize,
) -> String {
    let mut out = String::new();
    for b in 0..test_geometry_element.dim() {
        out += &indent(level);
        out += &format!(
            "let tsv{b} = &test_vertices[{}..{}];\n",
            b * gdim,
            (b + 1) * gdim
        );
    }
    for b in 0..test_geometry_element.dim() {
        out += &indent(level);
        out += &format!(
            "let tsg{b} = &TEST_GEOMETRY_EVALS[{}..{}];\n",
            b * test_npts,
            (b + 1) * test_npts,
        );
    }
    if test_geometry_element.degree() > 1 {
        for b in 0..test_geometry_element.dim() {
            out += &indent(level);
            out += &format!(
                "let tsgx{b} = &TEST_GEOMETRY_EVALS_DX[{}..{}];\n",
                b * test_npts,
                (b + 1) * test_npts,
            );
            out += &indent(level);
            out += &format!(
                "let tsgy{b} = &TEST_GEOMETRY_EVALS_DY[{}..{}];\n",
                b * test_npts,
                (b + 1) * test_npts,
            );
        }
    }
    for b in 0..trial_geometry_element.dim() {
        out += &indent(level);
        out += &format!(
            "let trv{b} = &trial_vertices[{}..{}];\n",
            b * gdim,
            (b + 1) * gdim
        );
    }
    for b in 0..trial_geometry_element.dim() {
        out += &indent(level);
        out += &format!(
            "let trg{b} = &TRIAL_GEOMETRY_EVALS[{}..{}];\n",
            b * trial_npts,
            (b + 1) * trial_npts,
        );
    }
    if trial_geometry_element.degree() > 1 {
        for b in 0..trial_geometry_element.dim() {
            out += &indent(level);
            out += &format!(
                "let trgx{b} = &TRIAL_GEOMETRY_EVALS_DX[{}..{}];\n",
                b * trial_npts,
                (b + 1) * trial_npts,
            );
            out += &indent(level);
            out += &format!(
                "let trgy{b} = &TRIAL_GEOMETRY_EVALS_DY[{}..{}];\n",
                b * trial_npts,
                (b + 1) * trial_npts,
            );
        }
    }
    if test_element.degree() > 0 {
        for b in 0..test_element.dim() {
            out += &indent(level);
            out += &format!(
                "let ts{b} = &TEST_EVALS[{}..{}];\n",
                b * test_npts,
                (b + 1) * test_npts
            );
        }
    }
    if trial_element.degree() > 0 {
        for b in 0..trial_element.dim() {
            out += &indent(level);
            out += &format!(
                "let tr{b} = &TRIAL_EVALS[{}..{}];\n",
                b * trial_npts,
                (b + 1) * trial_npts
            );
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "slice-static-tables"))]
fn take_slices(
    _test_element: &impl FiniteElement,
    _trial_element: &impl FiniteElement,
    test_geometry_element: &impl FiniteElement,
    trial_geometry_element: &impl FiniteElement,
    _test_npts: usize,
    _trial_npts: usize,
    gdim: usize,
    level: usize,
) -> String {
    let mut out = String::new();
    for b in 0..test_geometry_element.dim() {
        out += &indent(level);
        out += &format!(
            "let tsv{b} = &test_vertices[{}..{}];\n",
            b * gdim,
            (b + 1) * gdim
        );
    }
    for b in 0..trial_geometry_element.dim() {
        out += &indent(level);
        out += &format!(
            "let trv{b} = &trial_vertices[{}..{}];\n",
            b * gdim,
            (b + 1) * gdim
        );
    }
    out
}

trait Index {
    fn is_int(&self) -> bool;
    fn str(&self) -> String;
    fn int(&self) -> usize;
}
struct IntIndex {
    i: usize,
}
impl Index for IntIndex {
    fn is_int(&self) -> bool {
        true
    }
    fn str(&self) -> String {
        self.i.to_string()
    }
    fn int(&self) -> usize {
        self.i
    }
}
struct StrIndex {
    i: String,
}
impl Index for StrIndex {
    fn is_int(&self) -> bool {
        false
    }
    fn str(&self) -> String {
        self.i.to_string()
    }
    fn int(&self) -> usize {
        panic!("Not an integer.");
    }
}
fn vertex(t: FunctionType, v: &impl Index, d: &impl Index, gdim: usize) -> String {
    if v.is_int() {
        format!(
            "{}v{}[{}]",
            match t {
                FunctionType::Test => "ts",
                FunctionType::Trial => "tr",
            },
            v.str(),
            d.str()
        )
    } else {
        format!(
            "{}_vertices[{} * {gdim} + {}]",
            match t {
                FunctionType::Test => "test",
                FunctionType::Trial => "trial",
            },
            v.str(),
            d.str()
        )
    }
}

#[cfg(features = "slice-static-tables")]
fn function_value(
    t: FunctionType,
    v: ValueType,
    b: &impl Index,
    p: &impl Index,
    edim: usize,
) -> String {
    if v.is_int() {
        format!(
            "{}{}{}[{}]",
            match t {
                FunctionType::Test => "ts",
                FunctionType::Trial => "tr",
            },
            match v {
                ValueType::Value => "",
                ValueType::DX => "x",
                ValueType::DY => "y",
            },
            b.str(),
            p.str()
        )
    } else {
        if b.is_int() {
            format!(
                "{}_EVALS{}[{} + {}]",
                match t {
                    FunctionType::Test => "TEST",
                    FunctionType::Trial => "TRIAL",
                },
                match v {
                    ValueType::Value => "",
                    ValueType::DX => "_DX",
                    ValueType::DY => "_DY",
                },
                b.int() * edim,
                p.str()
            )
        } else {
            format!(
                "{}_EVALS{}[{} * {edim} + {}]",
                match t {
                    FunctionType::Test => "TEST",
                    FunctionType::Trial => "TRIAL",
                },
                match v {
                    ValueType::Value => "",
                    ValueType::DX => "_DX",
                    ValueType::DY => "_DY",
                },
                b.str(),
                p.str()
            )
        }
    }
}
#[cfg(not(features = "slice-static-tables"))]
fn function_value(
    t: FunctionType,
    v: ValueType,
    b: &impl Index,
    p: &impl Index,
    edim: usize,
) -> String {
    if b.is_int() {
        format!(
            "{}_EVALS{}[{} + {}]",
            match t {
                FunctionType::Test => "TEST",
                FunctionType::Trial => "TRIAL",
            },
            match v {
                ValueType::Value => "",
                ValueType::DX => "_DX",
                ValueType::DY => "_DY",
            },
            b.int() * edim,
            p.str()
        )
    } else {
        format!(
            "{}_EVALS{}[{} * {edim} + {}]",
            match t {
                FunctionType::Test => "TEST",
                FunctionType::Trial => "TRIAL",
            },
            match v {
                ValueType::Value => "",
                ValueType::DX => "_DX",
                ValueType::DY => "_DY",
            },
            b.str(),
            p.str()
        )
    }
}

#[cfg(features = "slice-static-tables")]
fn geometry(t: FunctionType, v: ValueType, b: &impl Index, p: &impl Index, npts: usize) -> String {
    if v.is_int() {
        format!(
            "{}g{}{}[{}]",
            match t {
                FunctionType::Test => "ts",
                FunctionType::Trial => "tr",
            },
            match v {
                ValueType::Value => "",
                ValueType::DX => "x",
                ValueType::DY => "y",
            },
            b.str(),
            p.str()
        )
    } else {
        if b.is_int() {
            format!(
                "{}_GEOMETRY_EVALS{}[{} + {}]",
                match t {
                    FunctionType::Test => "TEST",
                    FunctionType::Trial => "TRIAL",
                },
                match v {
                    ValueType::Value => "",
                    ValueType::DX => "_DX",
                    ValueType::DY => "_DY",
                },
                b.int() * npts,
                p.str()
            )
        } else {
            format!(
                "{}_GEOMETRY_EVALS{}[{} * {npts} + {}]",
                match t {
                    FunctionType::Test => "TEST",
                    FunctionType::Trial => "TRIAL",
                },
                match v {
                    ValueType::Value => "",
                    ValueType::DX => "_DX",
                    ValueType::DY => "_DY",
                },
                b.str(),
                p.str()
            )
        }
    }
}
#[cfg(not(features = "slice-static-tables"))]
fn geometry(t: FunctionType, v: ValueType, b: &impl Index, p: &impl Index, npts: usize) -> String {
    if b.is_int() {
        format!(
            "{}_GEOMETRY_EVALS{}[{} + {}]",
            match t {
                FunctionType::Test => "TEST",
                FunctionType::Trial => "TRIAL",
            },
            match v {
                ValueType::Value => "",
                ValueType::DX => "_DX",
                ValueType::DY => "_DY",
            },
            b.int() * npts,
            p.str()
        )
    } else {
        format!(
            "{}_GEOMETRY_EVALS{}[{} * {npts} + {}]",
            match t {
                FunctionType::Test => "TEST",
                FunctionType::Trial => "TRIAL",
            },
            match v {
                ValueType::Value => "",
                ValueType::DX => "_DX",
                ValueType::DY => "_DY",
            },
            b.str(),
            p.str()
        )
    }
}

fn linear_derivative<T: Num + Debug + Real>(
    name: String,
    vertices: &String,
    gdim: usize,
    table: &Array4D<T>,
    deriv: usize,
) -> String {
    let mut code = String::new();
    code += &format!("let {name} = [");
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
                } else if (*value + T::one()).abs() > (T::one() + T::one()).powi(-15) {
                    // value is -1
                    code += " - ";
                } else {
                    if started {
                        code += " + ";
                    }
                    code += &format!("{:?} * ", value);
                }
                code += &format!("{vertices}{f}[{d}]");
                started = true;
            }
        }
    }
    code += "];\n";
    code
}

fn linear_jacobian<T: Num + Debug + Real>(
    name: String,
    vertices: String,
    tdim: usize,
    gdim: usize,
    table: &Array4D<T>,
    level: usize,
) -> String {
    let mut code = String::new();
    if tdim == 2 && gdim == 3 {
        code += &indent(level);
        code += &linear_derivative("dx".to_string(), &vertices, gdim, table, 1);
        code += &indent(level);
        code += &linear_derivative("dy".to_string(), &vertices, gdim, table, 2);
        code += &indent(level);
        code += "let j = [dx[1] * dy[2] - dx[2] * dy[1], dx[2] * dy[0] - dx[0] * dy[2], dx[0] * dy[1] - dx[1] * dy[0]];\n";
        code += &indent(level);
        code += &format!("let {name} = (j[0].powi(2) + j[1].powi(2) + j[2].powi(2)).sqrt();\n");
    } else {
        panic!(
            "Jacobian computation not implemented for tdim {} and gdim {}.",
            tdim, gdim
        );
    }
    code
}

fn geometry_physical(
    t: FunctionType,
    v: ValueType,
    point: &String,
    gdim: usize,
    basis_count: usize,
) -> String {
    let mut code = String::new();
    code += "[";
    for d in 0..gdim {
        if d > 0 {
            code += ", ";
        }
        for f in 0..basis_count {
            if f > 0 {
                code += " + ";
            }
            code += &geometry(
                t,
                v,
                &IntIndex { i: f },
                &StrIndex {
                    i: point.to_string(),
                },
                basis_count,
            );
            code += " * ";
            code += &vertex(t, &IntIndex { i: f }, &IntIndex { i: d }, gdim);
        }
    }
    code += "]";
    code
}

fn jacobian(
    t: FunctionType,
    geometry_element: &impl FiniteElement,
    point: String,
    gdim: usize,
    tdim: usize,
    level: usize,
) -> String {
    let mut out = String::new();
    if geometry_element.degree() > 1 {
        if tdim == 2 && gdim == 3 {
            out += &indent(level);
            out += "let dx = ";
            out += &geometry_physical(t, ValueType::DX, &point, gdim, geometry_element.dim());
            out += ";\n";
            out += &indent(level);
            out += "let dy = ";
            out += &geometry_physical(t, ValueType::DY, &point, gdim, geometry_element.dim());
            out += ";\n";
            out += &indent(level);
            out += "let j = [dx[1] * dy[2] - dx[2] * dy[1], dx[2] * dy[0] - dx[0] * dy[2], dx[0] * dy[1] - dx[1] * dy[0]];\n";
            out += &indent(level);
            out += &format!(
                "let {}_jdet = (j[0].powi(2) + j[1].powi(2) + j[2].powi(2)).sqrt();\n",
                match t {
                    FunctionType::Test => "test",
                    FunctionType::Trial => "trial",
                }
            );
        } else {
            panic!(
                "Jacobian computation not implemented for tdim {} and gdim {}.",
                tdim, gdim
            );
        }
    }
    out
}

#[cfg(not(nightly))]
fn fadd(a: String, b: String) -> String {
    format!("{a} += {b};")
}

#[cfg(nightly)]
fn fadd(a: String, b: String) -> String {
    format!("{a} = unsafe {{ std::intrinsics::fadd_fast({a}, {b}) }};")
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
    let npts = if let Some((_qid, qrule)) = quadrules.iter().next() {
        qrule.npoints
    } else {
        panic!("No quadrature rule given.");
    };

    let mut code = String::new();
    // Function name and inputs
    code += &format!("fn {name}(&self, result: &mut [{typename}], test_vertices: &[{typename}], trial_vertices: &[{typename}]");
    if quadrules.len() > 1 {
        code += ", qid: u8";
    }
    code += ") {\n";

    // Assert that inputs have correct size
    code += &assert_sizes(
        test_element,
        trial_element,
        test_geometry_element,
        trial_geometry_element,
        gdim,
        1,
    );

    // Write quadrature weights
    code += &indent(1);
    code += &format!("let WTS: [{typename}; {npts}] = ");
    if let Some((_qid, qrule)) = quadrules.iter().next() {
        code += &table(&qrule.weights);
    } else {
        panic!("No quadrature rule given.");
    }
    code += ";\n";

    // Write tabulated elements
    let mut test_points = HashMap::new();
    let mut trial_points = HashMap::new();
    for (qid, qrule) in quadrules.iter() {
        let mut ts = Array2D::<T>::new((npts, tdim));
        let mut tr = Array2D::<T>::new((npts, tdim));
        for p in 0..npts {
            for d in 0..tdim {
                *ts.get_mut(p, d).unwrap() = qrule.test_points[p * tdim + d];
                *tr.get_mut(p, d).unwrap() = qrule.trial_points[p * tdim + d];
            }
        }
        test_points.insert(*qid, ts);
        trial_points.insert(*qid, tr);
    }
    code += &tables(
        test_element,
        trial_element,
        test_geometry_element,
        trial_geometry_element,
        &test_points,
        &trial_points,
        1,
    );
    code += "\n";

    // Set result to zero
    code += &set_zero(test_element.dim() * trial_element.dim(), 1);

    // Write slices
    code += &take_slices(
        test_element,
        trial_element,
        test_geometry_element,
        trial_geometry_element,
        npts,
        npts,
        gdim,
        1,
    );

    // Quadrature loop
    code += &indent(1);
    code += &format!("for q in 0..{npts} {{\n");

    // Compute the distance between points
    code += &indent(2);
    code += &format!("let mut sum_squares: {typename} = 0.0;\n");
    code += &indent(2);
    code += &format!("for d in 0..{gdim} {{\n");
    code += &indent(3);
    code += "let x = ";
    for b in 0..test_geometry_element.dim() {
        if b > 0 {
            code += " + ";
        }
        code += &vertex(
            FunctionType::Test,
            &IntIndex { i: b },
            &StrIndex { i: "d".to_string() },
            gdim,
        );
        code += " * ";
        code += &geometry(
            FunctionType::Test,
            ValueType::Value,
            &IntIndex { i: b },
            &StrIndex { i: "q".to_string() },
            npts,
        );
    }
    code += ";\n";
    code += &indent(3);
    code += "let y = ";
    for b in 0..trial_geometry_element.dim() {
        if b > 0 {
            code += " + ";
        }
        code += &vertex(
            FunctionType::Trial,
            &IntIndex { i: b },
            &StrIndex { i: "d".to_string() },
            gdim,
        );
        code += " * ";
        code += &geometry(
            FunctionType::Trial,
            ValueType::Value,
            &IntIndex { i: b },
            &StrIndex { i: "q".to_string() },
            npts,
        );
    }
    code += ";\n";
    code += &indent(3);
    code += &fadd("sum_squares".to_string(), "(x - y).powi(2)".to_string());
    code += "\n";
    code += &indent(2);
    code += "}\n";
    code += &indent(2);
    code += "let distance = sum_squares.sqrt();\n";

    // Compute jacobians
    code += &jacobian(
        FunctionType::Test,
        test_geometry_element,
        "q".to_string(),
        gdim,
        tdim,
        2,
    );
    code += &jacobian(
        FunctionType::Trial,
        trial_geometry_element,
        "q".to_string(),
        gdim,
        tdim,
        2,
    );

    code += &indent(2);
    code += "let c = WTS[q]";
    if test_geometry_element.degree() > 1 {
        code += " * test_jdet";
    }
    if trial_geometry_element.degree() > 1 {
        code += " * trial_jdet";
    }
    code += " / distance;\n";
    for i in 0..test_element.dim() {
        for j in 0..trial_element.dim() {
            let mut term = String::new();
            term += "c";
            if test_element.degree() > 0 {
                term += &format!(" * ");
                term += &function_value(
                    FunctionType::Test,
                    ValueType::Value,
                    &IntIndex { i },
                    &StrIndex { i: "q".to_string() },
                    test_element.dim(),
                );
            }
            if trial_element.degree() > 0 {
                term += &format!(" * ");
                term += &function_value(
                    FunctionType::Trial,
                    ValueType::Value,
                    &IntIndex { i },
                    &StrIndex { i: "q".to_string() },
                    trial_element.dim(),
                );
            }
            code += &indent(2);
            code += &fadd(format!("result[{}]", i * trial_element.dim() + j), term);
            code += "\n";
        }
    }

    // End quadrature loop
    code += &indent(1);
    code += "}\n";

    if test_geometry_element.degree() == 1 {
        let pts = Array2D::from_data(vec![0.0; tdim], (1, tdim));
        let mut test_geometry_eval_table =
            Array4D::<T>::new(test_geometry_element.tabulate_array_shape(1, 1));
        test_geometry_element.tabulate(&pts, 1, &mut test_geometry_eval_table);
        code += &linear_jacobian(
            "test_jdet".to_string(),
            "tsv".to_string(),
            tdim,
            gdim,
            &test_geometry_eval_table,
            1,
        );
    }
    if trial_geometry_element.degree() == 1 {
        let pts = Array2D::from_data(vec![0.0; tdim], (1, tdim));
        let mut trial_geometry_eval_table =
            Array4D::<T>::new(trial_geometry_element.tabulate_array_shape(1, 1));
        trial_geometry_element.tabulate(&pts, 1, &mut trial_geometry_eval_table);
        code += &linear_jacobian(
            "trial_jdet".to_string(),
            "trv".to_string(),
            tdim,
            gdim,
            &trial_geometry_eval_table,
            1,
        );
    }
    code += &indent(1);
    code += "let multiplier: f64 = 0.25 * std::f64::consts::FRAC_1_PI";
    if test_geometry_element.degree() == 1 {
        code += " * test_jdet";
    }
    if trial_geometry_element.degree() == 1 {
        code += " * trial_jdet";
    }
    code += ";\n";

    code += &indent(1);
    code += &format!(
        "for i in 0..{} {{\n",
        test_element.dim() * trial_element.dim()
    );
    code += &indent(2);
    code += "result[i] *= multiplier;\n";
    code += &indent(1);
    code += "}\n";

    code += "}\n\n";
    // If you want to see the code that this macro generates, uncomment this line
    // println!("{code}");
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

    let mut code = String::new();
    // Function name and inputs
    code += &format!("fn {name}(&self, result: &mut [{typename}], test_vertices: &[{typename}], trial_vertices: &[{typename}]) {{\n");

    // Assert that inputs have correct size
    code += &assert_sizes(
        test_element,
        trial_element,
        test_geometry_element,
        trial_geometry_element,
        gdim,
        1,
    );

    // Write quadrature weights
    code += &indent(1);
    code += &format!("let TEST_WTS: [{typename}; {test_npts}] = ");
    code += &table(&test_rule.weights);
    code += ";\n";
    code += &indent(1);
    code += &format!("let TRIAL_WTS: [{typename}; {trial_npts}] = ");
    code += &table(&trial_rule.weights);
    code += ";\n";

    // Write tabulated elements
    let mut test_points = HashMap::new();
    let mut trial_points = HashMap::new();
    test_points.insert(
        0,
        Array2D::<T>::from_data(test_rule.points, (test_npts, tdim)),
    );
    trial_points.insert(
        0,
        Array2D::<T>::from_data(trial_rule.points, (trial_npts, tdim)),
    );

    code += &tables(
        test_element,
        trial_element,
        test_geometry_element,
        trial_geometry_element,
        &test_points,
        &trial_points,
        1,
    );

    code += "\n";

    // Set result to zero
    code += &set_zero(test_element.dim() * trial_element.dim(), 1);

    // Write slices
    code += &take_slices(
        test_element,
        trial_element,
        test_geometry_element,
        trial_geometry_element,
        test_npts,
        trial_npts,
        gdim,
        1,
    );

    code += &indent(1);
    code += &format!("let mut x = [0.0; {gdim}];\n");
    code += &indent(1);
    code += &format!("for test_q in 0..{test_npts} {{\n");

    if test_geometry_element.degree() > 1 {
        code += &jacobian(
            FunctionType::Test,
            test_geometry_element,
            "test_q".to_string(),
            gdim,
            tdim,
            2,
        );
    }
    code += &indent(2);
    code += &format!("for d in 0..{gdim} {{\n");
    code += &indent(3);
    code += "x[d] = ";
    for b in 0..test_geometry_element.dim() {
        if b > 0 {
            code += " + ";
        }
        code += &vertex(
            FunctionType::Test,
            &IntIndex { i: b },
            &StrIndex { i: "d".to_string() },
            gdim,
        );
        code += " * ";
        code += &geometry(
            FunctionType::Test,
            ValueType::Value,
            &IntIndex { i: b },
            &StrIndex {
                i: "test_q".to_string(),
            },
            test_npts,
        );
    }
    code += ";\n";
    code += &indent(2);
    code += "}\n";
    code += &indent(2);
    code += &format!("for trial_q in 0..{trial_npts} {{\n");
    code += &indent(3);
    code += "let mut sum_squares = 0.0;\n";
    code += &indent(3);
    code += &format!("for d in 0..{gdim} {{\n");
    code += &indent(4);
    code += "let y = ";
    for b in 0..trial_geometry_element.dim() {
        if b > 0 {
            code += " + ";
        }
        code += &vertex(
            FunctionType::Trial,
            &IntIndex { i: b },
            &StrIndex { i: "d".to_string() },
            gdim,
        );
        code += " * ";
        code += &geometry(
            FunctionType::Trial,
            ValueType::Value,
            &IntIndex { i: b },
            &StrIndex {
                i: "trial_q".to_string(),
            },
            trial_npts,
        );
    }
    code += ";\n";
    code += &indent(4);
    code += &fadd("sum_squares".to_string(), "(x[d] - y).powi(2)".to_string());
    code += "\n";
    code += &indent(3);
    code += "}\n";
    code += &indent(3);
    code += "let distance = sum_squares.sqrt();\n";
    if trial_geometry_element.degree() > 1 {
        code += &jacobian(
            FunctionType::Trial,
            trial_geometry_element,
            "trial_q".to_string(),
            gdim,
            tdim,
            2,
        );
    }
    code += &indent(3);
    code += "let c = TEST_WTS[test_q] * TRIAL_WTS[trial_q]";
    if test_geometry_element.degree() > 1 {
        code += " * test_jdet";
    }
    if trial_geometry_element.degree() > 1 {
        code += " * trial_jdet";
    }
    code += " / distance;\n";
    for i in 0..test_element.dim() {
        for j in 0..trial_element.dim() {
            let mut term = String::new();
            term += "c";
            if test_element.degree() > 0 {
                term += &format!(" * ");
                term += &function_value(
                    FunctionType::Test,
                    ValueType::Value,
                    &IntIndex { i },
                    &StrIndex {
                        i: "test_q".to_string(),
                    },
                    test_element.dim(),
                );
            }
            if trial_element.degree() > 0 {
                term += &format!(" * ");
                term += &function_value(
                    FunctionType::Trial,
                    ValueType::Value,
                    &IntIndex { i },
                    &StrIndex {
                        i: "trial_q".to_string(),
                    },
                    trial_element.dim(),
                );
            }
            code += &indent(3);
            code += &fadd(format!("result[{}]", i * trial_element.dim() + j), term);
            code += "\n";
        }
    }
    code += &indent(2);
    code += "}\n";
    code += &indent(1);
    code += "}\n";

    if test_geometry_element.degree() == 1 {
        let pts = Array2D::from_data(vec![0.0; tdim], (1, tdim));
        let mut test_geometry_eval_table =
            Array4D::<T>::new(test_geometry_element.tabulate_array_shape(1, 1));
        test_geometry_element.tabulate(&pts, 1, &mut test_geometry_eval_table);
        code += &linear_jacobian(
            "test_jdet".to_string(),
            "tsv".to_string(),
            tdim,
            gdim,
            &test_geometry_eval_table,
            1,
        );
    }
    if trial_geometry_element.degree() == 1 {
        let pts = Array2D::from_data(vec![0.0; tdim], (1, tdim));
        let mut trial_geometry_eval_table =
            Array4D::<T>::new(trial_geometry_element.tabulate_array_shape(1, 1));
        trial_geometry_element.tabulate(&pts, 1, &mut trial_geometry_eval_table);
        code += &linear_jacobian(
            "trial_jdet".to_string(),
            "trv".to_string(),
            tdim,
            gdim,
            &trial_geometry_eval_table,
            1,
        );
    }

    code += &indent(1);
    code += "let multiplier: f64 = 0.25 * std::f64::consts::FRAC_1_PI";
    if test_geometry_element.degree() == 1 {
        code += " * test_jdet";
    }
    if trial_geometry_element.degree() == 1 {
        code += " * trial_jdet";
    }
    code += ";\n";

    code += &indent(1);
    code += &format!(
        "for i in 0..{} {{\n",
        test_element.dim() * trial_element.dim()
    );
    code += &indent(2);
    code += "result[i] *= multiplier;\n";
    code += &indent(1);
    code += "}\n";

    code += "}\n\n";
    // If you want to see the code that this macro generates, uncomment this line
    // println!("{code}");
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
    // If you want to print the generated code, uncomment this line
    // println!("{code}");
    code.parse().unwrap()
}
