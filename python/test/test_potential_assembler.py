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
