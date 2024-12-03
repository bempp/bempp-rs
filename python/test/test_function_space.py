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
    assert space.is_serial


@pytest.mark.parametrize("level", range(4))
def test_create_space_p1(level):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, 1)
    space = function_space(grid, element)

    assert space.local_size == grid.entity_count(ReferenceCellType.Point)
    assert space.local_size == space.global_size
    assert space.is_serial


@pytest.mark.parametrize("level", range(4))
def test_create_space_p2(level):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, 2)
    space = function_space(grid, element)

    assert space.local_size == grid.entity_count(ReferenceCellType.Point) + grid.entity_count(
        ReferenceCellType.Interval
    )
    assert space.local_size == space.global_size
    assert space.is_serial


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


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("degree", range(1, 5))
def test_get_local_dof_numbers(level, degree):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, degree)
    space = function_space(grid, element)

    for i in range(grid.entity_count(ReferenceCellType.Point)):
        assert len(space.get_local_dof_numbers(0, i)) == 1
    for i in range(grid.entity_count(ReferenceCellType.Interval)):
        assert len(space.get_local_dof_numbers(1, i)) == degree - 1
    for i in range(grid.entity_count(ReferenceCellType.Triangle)):
        assert len(space.get_local_dof_numbers(2, i)) == (degree - 1) * (degree - 2) // 2


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("degree", range(1, 5))
def test_cell_dofs(level, degree):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, degree)
    space = function_space(grid, element)

    for i in range(grid.entity_count(ReferenceCellType.Triangle)):
        assert len(space.cell_dofs(i)) == (degree + 1) * (degree + 2) // 2


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("degree", range(1, 5))
def test_global_dofs(level, degree):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, degree)
    space = function_space(grid, element)

    assert space.local_size == space.global_size
    for i in range(space.local_size):
        assert space.global_dof_index(i) == i


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("degree", range(1, 5))
def test_ownership(level, degree):
    grid = regular_sphere(level)
    element = create_family(Family.Lagrange, degree)
    space = function_space(grid, element)

    assert space.local_size == space.global_size
    for i in range(space.local_size):
        assert space.ownership(i).is_owned
