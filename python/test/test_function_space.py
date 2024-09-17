import pytest
from bempp.function_space import function_space
from ndgrid.shapes import regular_sphere
from ndelement.ciarlet import create_family, Family, Continuity
from ndelement.reference_cell import ReferenceCellType


@pytest.mark.parametrize("level", range(4))
def test_create_space_dp0(level):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, 0, Continuity.Discontinuous)

    space = function_space(grid, element)

    assert space.local_size == grid.entity_count(ReferenceCellType.Triangle)
    assert space.local_size == space.global_size


@pytest.mark.parametrize("level", range(4))
def test_create_space_p1(level):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, 1)
    space = function_space(grid, element)

    assert space.local_size == grid.entity_count(ReferenceCellType.Point)
    assert space.local_size == space.global_size


@pytest.mark.parametrize("level", range(4))
def test_create_space_p2(level):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, 2)
    space = function_space(grid, element)

    assert space.local_size == grid.entity_count(ReferenceCellType.Point) + grid.entity_count(
        ReferenceCellType.Interval
    )
    assert space.local_size == space.global_size


@pytest.mark.parametrize("level", range(4))
def test_grid(level):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, 1)
    space = function_space(grid, element)

    assert space.grid.topology_dim == 2
    assert space.grid.geometry_dim == 3
    for e in [
        ReferenceCellType.Point,
        ReferenceCellType.Interval,
        ReferenceCellType.Triangle,
        ReferenceCellType.Quadrilateral,
    ]:
        assert space.grid.entity_count(e) == grid.entity_count(e)


@pytest.mark.parametrize("level", range(4))
def test_element(level):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, 1)
    space = function_space(grid, element)

    e = ReferenceCellType.Triangle
    assert space.element(e).dim == element.element(e).dim
    assert space.element(e).entity_dofs(0, 0) == element.element(e).entity_dofs(0, 0)
