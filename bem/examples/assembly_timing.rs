use bempp_bem::assembly::{assemble_batched, BoundaryOperator, PDEType};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::create_element;
use bempp_grid::shapes::regular_sphere;
use bempp_tools::arrays::Array2D;
use bempp_traits::bem::DofMap;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{Continuity, ElementFamily};
use num::complex::Complex;
use std::time::Instant;

fn main() {
    for i in 0..5 {
        let now = Instant::now();
        let grid = regular_sphere(i);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );

        let space = SerialFunctionSpace::new(&grid, &element);
        let mut matrix = Array2D::<Complex<f64>>::new((
            space.dofmap().global_size(),
            space.dofmap().global_size(),
        ));

        assemble_batched(
            &mut matrix,
            BoundaryOperator::SingleLayer,
            PDEType::Helmholtz(5.0),
            &space,
            &space,
        );

        println!(
            "{} {}",
            space.dofmap().global_size(),
            now.elapsed().as_millis()
        )
    }
}
