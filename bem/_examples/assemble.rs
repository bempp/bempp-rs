use bempp_bem::assembly::{assemble, AssemblyType, BoundaryOperator, PDEType};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::{create_element, ElementFamily};
use bempp_grid::shapes::regular_sphere;
use bempp_traits::bem::DofMap;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::Continuity;
use rlst_dense::rlst_dynamic_array2;

fn main() {
    println!("Creating grid");
    let grid = regular_sphere(2);
    println!("Creating spaces");
    let element0 = create_element(
        ElementFamily::Lagrange,
        ReferenceCellType::Triangle,
        0,
        Continuity::Discontinuous,
    );
    let element1 = create_element(
        ElementFamily::Lagrange,
        ReferenceCellType::Triangle,
        1,
        Continuity::Discontinuous,
    );

    let space0 = SerialFunctionSpace::new(&grid, &element0);
    let space1 = SerialFunctionSpace::new(&grid, &element1);

    println!(
        "Assigning memory for {} by {} matrix",
        space1.dofmap().global_size(),
        space0.dofmap().global_size()
    );
    let mut matrix = rlst_dynamic_array2!(
        f64,
        [space1.dofmap().global_size(), space0.dofmap().global_size()]
    );

    println!("Assembling dense matrix (complex)");
    assemble(
        &mut matrix,
        AssemblyType::Dense,
        BoundaryOperator::SingleLayer,
        PDEType::Laplace,
        &space0,
        &space1,
    );
}
