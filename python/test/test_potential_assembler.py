import pytest
import numpy as np
from ndelement.reference_cell import ReferenceCellType
from bempp.assembly.potential import OperatorType, create_laplace_assembler
from bempp.function_space import function_space
from ndgrid.shapes import regular_sphere
from ndelement.ciarlet import create_family, Family, Continuity


@pytest.mark.parametrize(
    "otype",
    [
        OperatorType.SingleLayer,
        OperatorType.DoubleLayer,
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

    assert a.batch_size != 4
    a.set_batch_size(4)
    assert a.batch_size == 4

    assert a.dtype == np.float64


def test_single_layer_sphere0_dp0():
    grid = regular_sphere(0)
    element = create_family(Family.Lagrange, 0, Continuity.Discontinuous)
    space = function_space(grid, element)

    a = create_laplace_assembler(OperatorType.SingleLayer)

    points = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

    mat = a.assemble_into_dense(space, points)

    from_cl = np.array(
        [
            [
                0.04038047926587569,
                0.02879904511649957,
                0.02879904511649957,
                0.0403804792658757,
                0.04038047926587569,
                0.028799045116499562,
                0.02879904511649957,
                0.04038047926587571,
            ],
            [
                0.0403804792658757,
                0.04038047926587569,
                0.028799045116499573,
                0.02879904511649957,
                0.04038047926587571,
                0.04038047926587569,
                0.028799045116499573,
                0.028799045116499573,
            ],
            [
                0.04038047926587571,
                0.04038047926587571,
                0.04038047926587571,
                0.04038047926587571,
                0.028799045116499573,
                0.028799045116499573,
                0.028799045116499573,
                0.028799045116499573,
            ],
        ]
    )

    assert np.allclose(mat, from_cl, rtol=1e-4)
