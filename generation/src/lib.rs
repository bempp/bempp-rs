extern crate proc_macro;
use bempp_element::element::create_element;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_tools::arrays::{Array2D, Array4D};
use bempp_traits::arrays::Array4DAccess;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::ElementFamily;
use bempp_traits::element::FiniteElement;
use proc_macro::TokenStream;
use quote::quote;

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
    } else if cell_str == "\"Raviart-Thomas\"" {
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

fn format_table(name: String, table: &[f64]) -> String {
    let mut t = String::new();
    t += &format!("const {name}: [f64; {}] = [", table.len());
    for w in table {
        t += &format!("{w:?}, ");
    }
    t += "];\n";
    t
}

fn format_eval_table(name: String, table: &Array4D<f64>, deriv: usize, component: usize) -> String {
    let mut t = String::new();
    t += &format!(
        "const {name}: [f64; {}] = [",
        table.shape().1 * table.shape().2
    );
    for i in 0..table.shape().1 {
        for j in 0..table.shape().2 {
            t += &format!("{:?}, ", table.get(deriv, i, j, component).unwrap());
        }
    }
    t += "];\n";
    t
}

#[proc_macro]
pub fn make_answer(_item: TokenStream) -> TokenStream {
    let a = 42;
    let kernel_id = "untitled";
    let mut code = String::new();
    code += &format!("fn __bempp_kernel_{kernel_id}() -> u32 {{ {a} }}");
    code.parse().unwrap()
}

#[proc_macro]
pub fn generate_kernels(input: TokenStream) -> TokenStream {
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

    // TODO: degree
    let quadrule = triangle_duffy(
        &CellToCellConnectivity {
            connectivity_dimension: 2,
            local_indices: vec![(0, 0), (1, 1), (2, 2)],
        },
        1,
    )
    .unwrap();
    let npts = quadrule.npoints;
    let test_points = Array2D::<f64>::from_data(quadrule.test_points, (npts, 3));
    let trial_points = Array2D::<f64>::from_data(quadrule.trial_points, (npts, 3));

    let mut test_evals = Array4D::<f64>::new(test_element.tabulate_array_shape(1, npts));
    test_element.tabulate(&test_points, 1, &mut test_evals);
    let mut trial_evals = Array4D::<f64>::new(trial_element.tabulate_array_shape(1, npts));
    trial_element.tabulate(&trial_points, 1, &mut trial_evals);

    let mut test_geometry_evals =
        Array4D::<f64>::new(test_geometry_element.tabulate_array_shape(1, npts));
    test_geometry_element.tabulate(&test_points, 1, &mut test_geometry_evals);
    let mut trial_geometry_evals =
        Array4D::<f64>::new(trial_geometry_element.tabulate_array_shape(1, npts));
    trial_geometry_element.tabulate(&trial_points, 1, &mut trial_geometry_evals);

    let mut code = String::new();
    code += &format!("struct _BemppKernel_{kernel_name} {{\n");
    code += "    test_element: bempp_element::element::CiarletElement,\n";
    code += "    trial_element: bempp_element::element::CiarletElement,\n";
    code += "};\n\n";
    code += &format!("impl _BemppKernel_{kernel_name} {{");
    code += "\n\n";

    // TODO: split this kernel generation into its own function

    // Write const tables
    code += "fn same_triangle_kernel(result: &mut [f64], test_vertices: &[f64], trial_vertices: &[f64]) {\n";
    code += &format_table("WTS".to_string(), &quadrule.weights);
    code += &format_eval_table(
        "TEST_GEOMETRY_EVALS".to_string(),
        &test_geometry_evals,
        0,
        0,
    );
    if test_geometry_element.degree() > 1 {
        code += &format_eval_table(
            "TEST_GEOMETRY_EVALS_DX".to_string(),
            &test_geometry_evals,
            1,
            0,
        );
        code += &format_eval_table(
            "TEST_GEOMETRY_EVALS_DY".to_string(),
            &test_geometry_evals,
            2,
            0,
        );
    }
    code += &format_eval_table(
        "TRIAL_GEOMETRY_EVALS".to_string(),
        &trial_geometry_evals,
        0,
        0,
    );
    if trial_geometry_element.degree() > 1 {
        code += &format_eval_table(
            "TRIAL_GEOMETRY_EVALS_DX".to_string(),
            &trial_geometry_evals,
            1,
            0,
        );
        code += &format_eval_table(
            "TRIAL_GEOMETRY_EVALS_DY".to_string(),
            &trial_geometry_evals,
            2,
            0,
        );
    }
    if test_element.degree() > 0 {
        code += &format_eval_table("TEST_EVALS".to_string(), &test_evals, 0, 0);
    }
    if trial_element.degree() > 0 {
        code += &format_eval_table("TRIAL_EVALS".to_string(), &trial_evals, 0, 0);
    }
    code += "\n";

    code += &format!("for q in 0..{npts} {{\n");
    code += "let mut sum_squares = 0.0;\n";
    // TODO
    code += "let distance = f64::sqrt(sum_squares);\n";
    code += "}\n";

    code += "}\n\n";
    // END KERNEL

    code += "}\n\n";
    code += &format!("let {kernel_name} = _BemppKernel_{kernel_name} {{\n");
    code += &format!("    test_element: bempp_element::element::create_element(bempp_traits::element::ElementFamily::{:?}, bempp_traits::cell::ReferenceCellType::{:?}, {}, {}),\n",
                     test_element.family(), test_element.cell_type(), test_element.degree(), test_element.discontinuous());
    code += &format!("    trial_element: bempp_element::element::create_element(bempp_traits::element::ElementFamily::{:?}, bempp_traits::cell::ReferenceCellType::{:?}, {}, {}),\n",
                     trial_element.family(), trial_element.cell_type(), trial_element.degree(), trial_element.discontinuous());
    code += "};";
    println!("{}", code);
    code.parse().unwrap()
}
