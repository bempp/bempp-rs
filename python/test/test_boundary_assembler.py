import pytest
import numpy as np
from ndelement.reference_cell import ReferenceCellType
from bempp.assembly.boundary import OperatorType, create_laplace_assembler
from bempp.function_space import function_space
from ndgrid.shapes import regular_sphere
from ndelement.ciarlet import create_family, Family, Continuity


@pytest.mark.parametrize(
    "otype",
    [
        OperatorType.SingleLayer,
        OperatorType.DoubleLayer,
        OperatorType.AdjointDoubleLayer,
        OperatorType.Hypersingular,
    ],
)
def test_create_assembler(otype):
    a = create_laplace_assembler(otype)

    assert a.quadrature_degree(ReferenceCellType.Triangle) != 3
    a.set_quadrature_degree(ReferenceCellType.Triangle, 3)
    assert a.quadrature_degree(ReferenceCellType.Triangle) == 3
    with pytest.raises(ValueError):
        a.set_quadrature_degree(ReferenceCellType.Interval, 3)
    with pytest.raises(ValueError):
        a.quadrature_degree(ReferenceCellType.Interval)

    assert a.singular_quadrature_degree(ReferenceCellType.Triangle, ReferenceCellType.Triangle) != 3
    a.set_singular_quadrature_degree(ReferenceCellType.Triangle, ReferenceCellType.Triangle, 3)
    assert a.singular_quadrature_degree(ReferenceCellType.Triangle, ReferenceCellType.Triangle) == 3
    with pytest.raises(ValueError):
        a.set_singular_quadrature_degree(ReferenceCellType.Interval, ReferenceCellType.Interval, 3)

    assert a.batch_size != 4
    a.set_batch_size(4)
    assert a.batch_size == 4

    assert a.dtype == np.float64


@pytest.mark.parametrize(
    "operator",
    [
        OperatorType.SingleLayer,
        OperatorType.DoubleLayer,
        OperatorType.AdjointDoubleLayer,
        OperatorType.Hypersingular,
    ],
)
@pytest.mark.parametrize("test_degree", range(3))
@pytest.mark.parametrize("trial_degree", range(3))
def test_assemble_singular(operator, test_degree, trial_degree):
    grid = regular_sphere(0)
    test_element = create_family(Family.Lagrange, test_degree, Continuity.Discontinuous)
    test_space = function_space(grid, test_element)
    trial_element = create_family(Family.Lagrange, trial_degree, Continuity.Discontinuous)
    trial_space = function_space(grid, trial_element)

    a = create_laplace_assembler(operator)
    mat = a.assemble_singular(trial_space, test_space)

    dense = a.assemble_into_dense(trial_space, test_space)
    for i, j, value in zip(mat.row, mat.col, mat.data):
        assert np.isclose(dense[i, j], value)


@pytest.mark.parametrize(
    "operator",
    [
        OperatorType.SingleLayer,
        OperatorType.DoubleLayer,
        OperatorType.AdjointDoubleLayer,
        OperatorType.Hypersingular,
    ],
)
@pytest.mark.parametrize(
    "test_degree, test_continuity",
    [
        (0, Continuity.Discontinuous),
        (1, Continuity.Discontinuous),
        (1, Continuity.Standard),
        (2, Continuity.Standard),
    ],
)
@pytest.mark.parametrize(
    "trial_degree, trial_continuity",
    [
        (0, Continuity.Discontinuous),
        (1, Continuity.Discontinuous),
        (1, Continuity.Standard),
        (2, Continuity.Standard),
    ],
)
def test_assemble_singular_sparse_vs_dense(
    operator, test_degree, test_continuity, trial_degree, trial_continuity
):
    grid = regular_sphere(0)
    test_element = create_family(Family.Lagrange, test_degree, test_continuity)
    test_space = function_space(grid, test_element)
    trial_element = create_family(Family.Lagrange, trial_degree, trial_continuity)
    trial_space = function_space(grid, trial_element)

    a = create_laplace_assembler(operator)
    mat = a.assemble_singular(trial_space, test_space)

    dense = a.assemble_singular_into_dense(trial_space, test_space)
    assert np.allclose(dense, mat.todense())


@pytest.mark.parametrize(
    "operator",
    [
        OperatorType.SingleLayer,
        OperatorType.DoubleLayer,
        OperatorType.AdjointDoubleLayer,
        OperatorType.Hypersingular,
    ],
)
@pytest.mark.parametrize("test_degree", range(3))
@pytest.mark.parametrize("trial_degree", range(3))
def test_assemble_singular_correction(operator, test_degree, trial_degree):
    grid = regular_sphere(0)
    test_element = create_family(Family.Lagrange, test_degree, Continuity.Discontinuous)
    test_space = function_space(grid, test_element)
    trial_element = create_family(Family.Lagrange, trial_degree, Continuity.Discontinuous)
    trial_space = function_space(grid, trial_element)

    a = create_laplace_assembler(operator)
    a.assemble_singular_correction(trial_space, test_space)


@pytest.mark.parametrize(
    "operator",
    [
        OperatorType.SingleLayer,
        OperatorType.DoubleLayer,
        OperatorType.AdjointDoubleLayer,
        OperatorType.Hypersingular,
    ],
)
@pytest.mark.parametrize(
    "test_degree, test_continuity",
    [
        (0, Continuity.Discontinuous),
        (1, Continuity.Discontinuous),
        (1, Continuity.Standard),
        (2, Continuity.Standard),
    ],
)
@pytest.mark.parametrize(
    "trial_degree, trial_continuity",
    [
        (0, Continuity.Discontinuous),
        (1, Continuity.Discontinuous),
        (1, Continuity.Standard),
        (2, Continuity.Standard),
    ],
)
def test_assemble_singular_and_nonsingular(
    operator, test_degree, test_continuity, trial_degree, trial_continuity
):
    grid = regular_sphere(0)
    test_element = create_family(Family.Lagrange, test_degree, test_continuity)
    test_space = function_space(grid, test_element)
    trial_element = create_family(Family.Lagrange, trial_degree, trial_continuity)
    trial_space = function_space(grid, trial_element)

    a = create_laplace_assembler(operator)
    mat = a.assemble_singular_into_dense(trial_space, test_space)
    mat += a.assemble_nonsingular_into_dense(trial_space, test_space)

    mat2 = a.assemble_into_dense(trial_space, test_space)
    assert np.allclose(mat, mat2)


def test_single_layer_sphere0_dp0():
    grid = regular_sphere(0)
    element = create_family(Family.Lagrange, 0, Continuity.Discontinuous)
    space = function_space(grid, element)

    a = create_laplace_assembler(OperatorType.SingleLayer)

    mat = a.assemble_into_dense(space, space)

    from_cl = np.array(
        [
            [
                0.1854538822982487,
                0.08755414595678074,
                0.05963897421514472,
                0.08755414595678074,
                0.08755414595678074,
                0.05963897421514473,
                0.04670742127454548,
                0.05963897421514472,
            ],
            [
                0.08755414595678074,
                0.1854538822982487,
                0.08755414595678074,
                0.05963897421514472,
                0.05963897421514472,
                0.08755414595678074,
                0.05963897421514473,
                0.04670742127454548,
            ],
            [
                0.05963897421514472,
                0.08755414595678074,
                0.1854538822982487,
                0.08755414595678074,
                0.04670742127454548,
                0.05963897421514472,
                0.08755414595678074,
                0.05963897421514473,
            ],
            [
                0.08755414595678074,
                0.05963897421514472,
                0.08755414595678074,
                0.1854538822982487,
                0.05963897421514473,
                0.04670742127454548,
                0.05963897421514472,
                0.08755414595678074,
            ],
            [
                0.08755414595678074,
                0.05963897421514472,
                0.046707421274545476,
                0.05963897421514473,
                0.1854538822982487,
                0.08755414595678074,
                0.05963897421514472,
                0.08755414595678074,
            ],
            [
                0.05963897421514473,
                0.08755414595678074,
                0.05963897421514472,
                0.046707421274545476,
                0.08755414595678074,
                0.1854538822982487,
                0.08755414595678074,
                0.05963897421514472,
            ],
            [
                0.046707421274545476,
                0.05963897421514473,
                0.08755414595678074,
                0.05963897421514472,
                0.05963897421514472,
                0.08755414595678074,
                0.1854538822982487,
                0.08755414595678074,
            ],
            [
                0.05963897421514472,
                0.046707421274545476,
                0.05963897421514473,
                0.08755414595678074,
                0.08755414595678074,
                0.05963897421514472,
                0.08755414595678074,
                0.1854538822982487,
            ],
        ]
    )

    assert np.allclose(mat, from_cl, rtol=1e-4)
