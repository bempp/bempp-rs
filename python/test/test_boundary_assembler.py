import pytest
import numpy as np
from ndelement.reference_cell import ReferenceCellType
from bempp.assembly.boundary import OperatorType, create_laplace_assembler


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
