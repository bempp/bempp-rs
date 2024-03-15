use approx::*;
use bempp_bem::assembly::{batched, batched::BatchedAssembler};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::lagrange;
use bempp_grid::shapes::regular_sphere;
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::Continuity;
use bempp_traits::types::ReferenceCellType;
use cauchy::c64;
use rlst_dense::{rlst_dynamic_array2, traits::RandomAccessByRef};

#[test]
fn test_laplace_single_layer_dp0_dp0() {
    let grid = regular_sphere(0);
    let element = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();

    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);

    let a = batched::LaplaceSingleLayerAssembler::default();
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.08755414595678074, 0.05963897421514473, 0.04670742127454548, 0.05963897421514472], vec![0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.05963897421514472, 0.08755414595678074, 0.05963897421514473, 0.04670742127454548], vec![0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.04670742127454548, 0.05963897421514472, 0.08755414595678074, 0.05963897421514473], vec![0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.05963897421514473, 0.04670742127454548, 0.05963897421514472, 0.08755414595678074], vec![0.08755414595678074, 0.05963897421514472, 0.046707421274545476, 0.05963897421514473, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074], vec![0.05963897421514473, 0.08755414595678074, 0.05963897421514472, 0.046707421274545476, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472], vec![0.046707421274545476, 0.05963897421514473, 0.08755414595678074, 0.05963897421514472, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074], vec![0.05963897421514472, 0.046707421274545476, 0.05963897421514473, 0.08755414595678074, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487]];

    for (i, row) in from_cl.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            assert_relative_eq!(*matrix.get([i, j]).unwrap(), entry, epsilon = 1e-3);
        }
    }
}

#[test]
fn test_laplace_double_layer_dp0_dp0() {
    let grid = regular_sphere(0);
    let element = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();

    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceDoubleLayerAssembler::default();
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![-1.9658941517361406e-33, -0.08477786720045567, -0.048343860959178774, -0.08477786720045567, -0.08477786720045566, -0.048343860959178774, -0.033625570841778946, -0.04834386095917877], vec![-0.08477786720045567, -1.9658941517361406e-33, -0.08477786720045567, -0.048343860959178774, -0.04834386095917877, -0.08477786720045566, -0.048343860959178774, -0.033625570841778946], vec![-0.048343860959178774, -0.08477786720045567, -1.9658941517361406e-33, -0.08477786720045567, -0.033625570841778946, -0.04834386095917877, -0.08477786720045566, -0.048343860959178774], vec![-0.08477786720045567, -0.048343860959178774, -0.08477786720045567, -1.9658941517361406e-33, -0.048343860959178774, -0.033625570841778946, -0.04834386095917877, -0.08477786720045566], vec![-0.08477786720045566, -0.04834386095917877, -0.033625570841778946, -0.04834386095917877, 4.910045345075783e-33, -0.08477786720045566, -0.048343860959178774, -0.08477786720045566], vec![-0.04834386095917877, -0.08477786720045566, -0.04834386095917877, -0.033625570841778946, -0.08477786720045566, 4.910045345075783e-33, -0.08477786720045566, -0.048343860959178774], vec![-0.033625570841778946, -0.04834386095917877, -0.08477786720045566, -0.04834386095917877, -0.048343860959178774, -0.08477786720045566, 4.910045345075783e-33, -0.08477786720045566], vec![-0.04834386095917877, -0.033625570841778946, -0.04834386095917877, -0.08477786720045566, -0.08477786720045566, -0.048343860959178774, -0.08477786720045566, 4.910045345075783e-33]];

    for (i, row) in from_cl.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            assert_relative_eq!(*matrix.get([i, j]).unwrap(), entry, epsilon = 1e-4);
        }
    }
}

#[test]
fn test_laplace_adjoint_double_layer_dp0_dp0() {
    let grid = regular_sphere(0);
    let element = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();

    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceAdjointDoubleLayerAssembler::default();
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![1.9658941517361406e-33, -0.08478435261011981, -0.048343860959178774, -0.0847843526101198, -0.08478435261011981, -0.04834386095917877, -0.033625570841778946, -0.048343860959178774], vec![-0.0847843526101198, 1.9658941517361406e-33, -0.08478435261011981, -0.048343860959178774, -0.048343860959178774, -0.08478435261011981, -0.04834386095917877, -0.033625570841778946], vec![-0.048343860959178774, -0.0847843526101198, 1.9658941517361406e-33, -0.08478435261011981, -0.033625570841778946, -0.048343860959178774, -0.08478435261011981, -0.04834386095917877], vec![-0.08478435261011981, -0.048343860959178774, -0.0847843526101198, 1.9658941517361406e-33, -0.04834386095917877, -0.033625570841778946, -0.048343860959178774, -0.08478435261011981], vec![-0.0847843526101198, -0.04834386095917877, -0.033625570841778946, -0.04834386095917877, -4.910045345075783e-33, -0.0847843526101198, -0.048343860959178774, -0.08478435261011981], vec![-0.04834386095917877, -0.0847843526101198, -0.04834386095917877, -0.033625570841778946, -0.08478435261011981, -4.910045345075783e-33, -0.0847843526101198, -0.048343860959178774], vec![-0.033625570841778946, -0.04834386095917877, -0.0847843526101198, -0.04834386095917877, -0.048343860959178774, -0.08478435261011981, -4.910045345075783e-33, -0.0847843526101198], vec![-0.04834386095917877, -0.033625570841778946, -0.04834386095917877, -0.0847843526101198, -0.0847843526101198, -0.048343860959178774, -0.08478435261011981, -4.910045345075783e-33]];

    for (i, row) in from_cl.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            assert_relative_eq!(*matrix.get([i, j]).unwrap(), entry, epsilon = 1e-4);
        }
    }
}

#[test]
fn test_laplace_hypersingular_dp0_dp0() {
    let grid = regular_sphere(0);
    let element = lagrange::create(
        ReferenceCellType::Triangle,
        0,
        Continuity::Discontinuous,
    );
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();

    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceHypersingularAssembler::default();
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    for i in 0..ndofs {
        for j in 0..ndofs {
            assert_relative_eq!(*matrix.get([i, j]).unwrap(), 0.0, epsilon = 1e-4);
        }
    }
}

#[test]
fn test_laplace_hypersingular_p1_p1() {
    let grid = regular_sphere(0);
    let element = lagrange::create(
        ReferenceCellType::Triangle,
        1,
        Continuity::Continuous,
    );
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();

    let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
    let a = batched::LaplaceHypersingularAssembler::default();
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![0.33550642155494004, -0.10892459915262698, -0.05664545560057827, -0.05664545560057828, -0.0566454556005783, -0.05664545560057828], vec![-0.10892459915262698, 0.33550642155494004, -0.05664545560057828, -0.05664545560057827, -0.05664545560057828, -0.05664545560057829], vec![-0.05664545560057828, -0.05664545560057827, 0.33550642155494004, -0.10892459915262698, -0.056645455600578286, -0.05664545560057829], vec![-0.05664545560057827, -0.05664545560057828, -0.10892459915262698, 0.33550642155494004, -0.05664545560057828, -0.056645455600578286], vec![-0.05664545560057829, -0.0566454556005783, -0.05664545560057829, -0.05664545560057829, 0.33550642155494004, -0.10892459915262698], vec![-0.05664545560057829, -0.05664545560057831, -0.05664545560057829, -0.05664545560057829, -0.10892459915262698, 0.33550642155494004]];

    let perm = [0, 5, 2, 4, 3, 1];

    for (i, pi) in perm.iter().enumerate() {
        for (j, pj) in perm.iter().enumerate() {
            assert_relative_eq!(
                *matrix.get([i, j]).unwrap(),
                from_cl[*pi][*pj],
                epsilon = 1e-4
            );
        }
    }
}

#[test]
fn test_helmholtz_single_layer_dp0_dp0() {
    let grid = regular_sphere(0);
    let element = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();
    let mut matrix = rlst_dynamic_array2!(c64, [ndofs, ndofs]);

    let a = batched::HelmholtzSingleLayerAssembler::new(3.0);
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![c64::new(0.08742460357596939, 0.11004203436820102), c64::new(-0.02332791148192136, 0.04919102584271124), c64::new(-0.04211947809894265, 0.003720159902487029), c64::new(-0.02332791148192136, 0.04919102584271125), c64::new(-0.023327911481921364, 0.04919102584271124), c64::new(-0.042119478098942634, 0.003720159902487025), c64::new(-0.03447046598405515, -0.02816544680626108), c64::new(-0.04211947809894265, 0.0037201599024870254)], vec![c64::new(-0.023327911481921364, 0.04919102584271125), c64::new(0.08742460357596939, 0.11004203436820104), c64::new(-0.02332791148192136, 0.04919102584271124), c64::new(-0.04211947809894265, 0.0037201599024870263), c64::new(-0.04211947809894265, 0.0037201599024870254), c64::new(-0.02332791148192136, 0.04919102584271125), c64::new(-0.042119478098942634, 0.003720159902487025), c64::new(-0.03447046598405515, -0.028165446806261072)], vec![c64::new(-0.04211947809894265, 0.003720159902487029), c64::new(-0.02332791148192136, 0.04919102584271125), c64::new(0.08742460357596939, 0.11004203436820102), c64::new(-0.02332791148192136, 0.04919102584271124), c64::new(-0.03447046598405515, -0.02816544680626108), c64::new(-0.04211947809894265, 0.0037201599024870254), c64::new(-0.023327911481921364, 0.04919102584271124), c64::new(-0.042119478098942634, 0.003720159902487025)], vec![c64::new(-0.02332791148192136, 0.04919102584271124), c64::new(-0.04211947809894265, 0.0037201599024870263), c64::new(-0.023327911481921364, 0.04919102584271125), c64::new(0.08742460357596939, 0.11004203436820104), c64::new(-0.042119478098942634, 0.003720159902487025), c64::new(-0.03447046598405515, -0.028165446806261072), c64::new(-0.04211947809894265, 0.0037201599024870254), c64::new(-0.02332791148192136, 0.04919102584271125)], vec![c64::new(-0.023327911481921364, 0.04919102584271125), c64::new(-0.04211947809894265, 0.0037201599024870263), c64::new(-0.03447046598405515, -0.02816544680626108), c64::new(-0.042119478098942634, 0.003720159902487025), c64::new(0.08742460357596939, 0.11004203436820104), c64::new(-0.02332791148192136, 0.04919102584271124), c64::new(-0.04211947809894265, 0.0037201599024870267), c64::new(-0.023327911481921364, 0.04919102584271125)], vec![c64::new(-0.042119478098942634, 0.003720159902487025), c64::new(-0.02332791148192136, 0.04919102584271125), c64::new(-0.04211947809894265, 0.0037201599024870263), c64::new(-0.034470465984055156, -0.028165446806261075), c64::new(-0.02332791148192136, 0.04919102584271124), c64::new(0.08742460357596939, 0.11004203436820104), c64::new(-0.023327911481921364, 0.04919102584271125), c64::new(-0.04211947809894265, 0.0037201599024870237)], vec![c64::new(-0.03447046598405515, -0.02816544680626108), c64::new(-0.042119478098942634, 0.003720159902487025), c64::new(-0.023327911481921364, 0.04919102584271125), c64::new(-0.04211947809894265, 0.0037201599024870263), c64::new(-0.04211947809894265, 0.0037201599024870267), c64::new(-0.023327911481921364, 0.04919102584271125), c64::new(0.08742460357596939, 0.11004203436820104), c64::new(-0.02332791148192136, 0.04919102584271124)], vec![c64::new(-0.04211947809894265, 0.0037201599024870263), c64::new(-0.034470465984055156, -0.028165446806261075), c64::new(-0.042119478098942634, 0.003720159902487025), c64::new(-0.02332791148192136, 0.04919102584271125), c64::new(-0.023327911481921364, 0.04919102584271125), c64::new(-0.04211947809894265, 0.0037201599024870237), c64::new(-0.02332791148192136, 0.04919102584271124), c64::new(0.08742460357596939, 0.11004203436820104)]];

    for (i, row) in from_cl.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            assert_relative_eq!(*matrix.get([i, j]).unwrap(), entry, epsilon = 1e-4);
        }
    }
}

#[test]
fn test_helmholtz_double_layer_dp0_dp0() {
    let grid = regular_sphere(0);
    let element = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();
    let mut matrix = rlst_dynamic_array2!(c64, [ndofs, ndofs]);

    let a = batched::HelmholtzDoubleLayerAssembler::new(3.0);
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![c64::new(-1.025266688854119e-33, -7.550086433767158e-36), c64::new(-0.07902626473768169, -0.08184681047051735), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(-0.07902626473768172, -0.08184681047051737), c64::new(-0.07902626473768169, -0.08184681047051737), c64::new(0.01906923918000323, -0.10276858786959302), c64::new(0.10089706509966115, -0.07681163409722505), c64::new(0.019069239180003215, -0.10276858786959299)], vec![c64::new(-0.07902626473768172, -0.08184681047051737), c64::new(-1.025266688854119e-33, 1.0291684702482414e-35), c64::new(-0.0790262647376817, -0.08184681047051737), c64::new(0.019069239180003212, -0.10276858786959299), c64::new(0.019069239180003212, -0.10276858786959298), c64::new(-0.07902626473768168, -0.08184681047051737), c64::new(0.01906923918000323, -0.10276858786959299), c64::new(0.10089706509966115, -0.07681163409722506)], vec![c64::new(0.01906923918000321, -0.10276858786959298), c64::new(-0.07902626473768172, -0.08184681047051737), c64::new(-1.025266688854119e-33, -7.550086433767158e-36), c64::new(-0.07902626473768169, -0.08184681047051735), c64::new(0.10089706509966115, -0.07681163409722505), c64::new(0.019069239180003215, -0.10276858786959299), c64::new(-0.07902626473768169, -0.08184681047051737), c64::new(0.01906923918000323, -0.10276858786959302)], vec![c64::new(-0.0790262647376817, -0.08184681047051737), c64::new(0.019069239180003212, -0.10276858786959299), c64::new(-0.07902626473768172, -0.08184681047051737), c64::new(-1.025266688854119e-33, 1.0291684702482414e-35), c64::new(0.01906923918000323, -0.10276858786959299), c64::new(0.10089706509966115, -0.07681163409722506), c64::new(0.019069239180003212, -0.10276858786959298), c64::new(-0.07902626473768168, -0.08184681047051737)], vec![c64::new(-0.07902626473768172, -0.08184681047051737), c64::new(0.019069239180003215, -0.10276858786959298), c64::new(0.10089706509966115, -0.07681163409722505), c64::new(0.01906923918000323, -0.10276858786959299), c64::new(5.00373588753262e-33, -1.8116810507789718e-36), c64::new(-0.07902626473768169, -0.08184681047051735), c64::new(0.019069239180003212, -0.10276858786959299), c64::new(-0.07902626473768169, -0.08184681047051737)], vec![c64::new(0.019069239180003222, -0.10276858786959299), c64::new(-0.07902626473768173, -0.08184681047051737), c64::new(0.01906923918000322, -0.10276858786959299), c64::new(0.10089706509966115, -0.07681163409722506), c64::new(-0.07902626473768169, -0.08184681047051735), c64::new(7.314851820797302e-33, -1.088140415641433e-35), c64::new(-0.07902626473768169, -0.08184681047051737), c64::new(0.01906923918000322, -0.10276858786959299)], vec![c64::new(0.10089706509966115, -0.07681163409722505), c64::new(0.01906923918000323, -0.10276858786959299), c64::new(-0.07902626473768172, -0.08184681047051737), c64::new(0.019069239180003215, -0.10276858786959298), c64::new(0.019069239180003212, -0.10276858786959299), c64::new(-0.07902626473768169, -0.08184681047051737), c64::new(5.00373588753262e-33, -1.8116810507789718e-36), c64::new(-0.07902626473768169, -0.08184681047051735)], vec![c64::new(0.01906923918000322, -0.10276858786959299), c64::new(0.10089706509966115, -0.07681163409722506), c64::new(0.019069239180003222, -0.10276858786959299), c64::new(-0.07902626473768173, -0.08184681047051737), c64::new(-0.07902626473768169, -0.08184681047051737), c64::new(0.01906923918000322, -0.10276858786959299), c64::new(-0.07902626473768169, -0.08184681047051735), c64::new(7.314851820797302e-33, -1.088140415641433e-35)]];

    for (i, row) in from_cl.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            assert_relative_eq!(matrix.get([i, j]).unwrap(), entry, epsilon = 1e-4);
        }
    }
}
#[test]
fn test_helmholtz_adjoint_double_layer_dp0_dp0() {
    let grid = regular_sphere(0);
    let element = lagrange::create(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();
    let mut matrix = rlst_dynamic_array2!(c64, [ndofs, ndofs]);

    let a = batched::HelmholtzAdjointDoubleLayerAssembler::new(3.0);
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![c64::new(1.025266688854119e-33, 7.550086433767158e-36), c64::new(-0.079034545070751, -0.08184700030244885), c64::new(0.019069239180003205, -0.10276858786959298), c64::new(-0.07903454507075097, -0.08184700030244886), c64::new(-0.07903454507075099, -0.08184700030244887), c64::new(0.01906923918000323, -0.10276858786959299), c64::new(0.10089706509966115, -0.07681163409722505), c64::new(0.019069239180003212, -0.10276858786959298)], vec![c64::new(-0.07903454507075097, -0.08184700030244885), c64::new(1.025266688854119e-33, -1.0291684702482414e-35), c64::new(-0.079034545070751, -0.08184700030244887), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(-0.07903454507075099, -0.08184700030244887), c64::new(0.019069239180003233, -0.10276858786959299), c64::new(0.10089706509966115, -0.07681163409722506)], vec![c64::new(0.019069239180003205, -0.10276858786959298), c64::new(-0.07903454507075097, -0.08184700030244886), c64::new(1.025266688854119e-33, 7.550086433767158e-36), c64::new(-0.079034545070751, -0.08184700030244885), c64::new(0.10089706509966115, -0.07681163409722505), c64::new(0.019069239180003212, -0.10276858786959298), c64::new(-0.07903454507075099, -0.08184700030244887), c64::new(0.01906923918000323, -0.10276858786959299)], vec![c64::new(-0.079034545070751, -0.08184700030244887), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(-0.07903454507075097, -0.08184700030244885), c64::new(1.025266688854119e-33, -1.0291684702482414e-35), c64::new(0.019069239180003233, -0.10276858786959299), c64::new(0.10089706509966115, -0.07681163409722506), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(-0.07903454507075099, -0.08184700030244887)], vec![c64::new(-0.07903454507075099, -0.08184700030244887), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(0.10089706509966115, -0.07681163409722505), c64::new(0.01906923918000323, -0.10276858786959302), c64::new(-5.00373588753262e-33, 1.8116810507789718e-36), c64::new(-0.07903454507075099, -0.08184700030244885), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(-0.07903454507075099, -0.08184700030244886)], vec![c64::new(0.019069239180003233, -0.10276858786959302), c64::new(-0.07903454507075099, -0.08184700030244886), c64::new(0.019069239180003212, -0.10276858786959298), c64::new(0.10089706509966115, -0.07681163409722506), c64::new(-0.07903454507075099, -0.08184700030244885), c64::new(-7.314851820797302e-33, 1.088140415641433e-35), c64::new(-0.07903454507075099, -0.08184700030244886), c64::new(0.019069239180003215, -0.10276858786959298)], vec![c64::new(0.10089706509966115, -0.07681163409722505), c64::new(0.01906923918000323, -0.10276858786959302), c64::new(-0.07903454507075099, -0.08184700030244887), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(0.01906923918000321, -0.10276858786959298), c64::new(-0.07903454507075099, -0.08184700030244886), c64::new(-5.00373588753262e-33, 1.8116810507789718e-36), c64::new(-0.07903454507075099, -0.08184700030244885)], vec![c64::new(0.019069239180003212, -0.10276858786959298), c64::new(0.10089706509966115, -0.07681163409722506), c64::new(0.019069239180003233, -0.10276858786959302), c64::new(-0.07903454507075099, -0.08184700030244886), c64::new(-0.07903454507075099, -0.08184700030244886), c64::new(0.019069239180003215, -0.10276858786959298), c64::new(-0.07903454507075099, -0.08184700030244885), c64::new(-7.314851820797302e-33, 1.088140415641433e-35)]];

    for (i, row) in from_cl.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            assert_relative_eq!(matrix.get([i, j]).unwrap(), entry, epsilon = 1e-4);
        }
    }
}
/*
#[test]
fn test_helmholtz_hypersingular_p1_p1() {
    let grid = regular_sphere(0);
    let element = lagrange::create(
        ReferenceCellType::Triangle,
        1,
        Continuity::Continuous,
    );
    let space = SerialFunctionSpace::new(&grid, &element);

    let ndofs = space.global_size();
    let mut matrix = rlst_dynamic_array2!(c64, [ndofs, ndofs]);

    let a = batched::HelmholtzHypersingularAssembler::new(3.0);
    a.assemble_into_dense::<128, _, _>(&mut matrix, &space, &space);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![c64::new(-0.24054975187128322, -0.37234907871793793), c64::new(-0.2018803657726846, -0.3708486980714607), c64::new(-0.31151549914430937, -0.36517694339435425), c64::new(-0.31146604913280734, -0.3652407688678574), c64::new(-0.3114620814217625, -0.36524076431695807), c64::new(-0.311434147468966, -0.36530056813389983)], vec![c64::new(-0.2018803657726846, -0.3708486980714607), c64::new(-0.24054975187128322, -0.3723490787179379), c64::new(-0.31146604913280734, -0.3652407688678574), c64::new(-0.31151549914430937, -0.36517694339435425), c64::new(-0.3114620814217625, -0.36524076431695807), c64::new(-0.311434147468966, -0.36530056813389983)], vec![c64::new(-0.31146604913280734, -0.3652407688678574), c64::new(-0.31151549914430937, -0.36517694339435425), c64::new(-0.24054975187128322, -0.3723490787179379), c64::new(-0.2018803657726846, -0.3708486980714607), c64::new(-0.31146208142176246, -0.36524076431695807), c64::new(-0.31143414746896597, -0.36530056813389983)], vec![c64::new(-0.31151549914430937, -0.36517694339435425), c64::new(-0.31146604913280734, -0.3652407688678574), c64::new(-0.2018803657726846, -0.3708486980714607), c64::new(-0.24054975187128322, -0.3723490787179379), c64::new(-0.3114620814217625, -0.36524076431695807), c64::new(-0.311434147468966, -0.36530056813389983)], vec![c64::new(-0.31146208142176257, -0.36524076431695807), c64::new(-0.3114620814217625, -0.3652407643169581), c64::new(-0.3114620814217625, -0.3652407643169581), c64::new(-0.3114620814217625, -0.3652407643169581), c64::new(-0.24056452443903534, -0.37231826606213236), c64::new(-0.20188036577268464, -0.37084869807146076)], vec![c64::new(-0.3114335658086867, -0.36530052927274986), c64::new(-0.31143356580868675, -0.36530052927274986), c64::new(-0.3114335658086867, -0.36530052927274986), c64::new(-0.3114335658086867, -0.36530052927274986), c64::new(-0.2018803657726846, -0.37084869807146076), c64::new(-0.2402983805938184, -0.37203286968364935)]];

    let perm = [0, 5, 2, 4, 3, 1];

    for (i, pi) in perm.iter().enumerate() {
        for (j, pj) in perm.iter().enumerate() {
            assert_relative_eq!(
                *matrix.get([i, j]).unwrap(),
                from_cl[*pi][*pj],
                epsilon = 1e-3
            );
        }
    }
}
*/
