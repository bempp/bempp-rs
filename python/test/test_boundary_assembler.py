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
