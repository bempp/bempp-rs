//! C Interface

use crate::cell::*;
use libc::size_t;
pub use solvers_tools::RustyDataContainer;

pub struct ReferenceCellContainer(Box<dyn ReferenceCell>);

impl ReferenceCellContainer {
    pub fn to_box(self) -> Box<ReferenceCellContainer> {
        Box::new(self)
    }
}

// get_cell_property_impl!(dim, size_t);

fn get_reference(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> &'static dyn ReferenceCell {
    Box::leak(cell_container.unwrap()).0.as_ref()
}

/// New cell container from interval.
#[no_mangle]
pub extern "C" fn reference_cell_container_new_from_interval() -> Box<ReferenceCellContainer> {
    ReferenceCellContainer(Box::new(Interval {})).to_box()
}

/// New cell container from triangle.
#[no_mangle]
pub extern "C" fn reference_cell_container_new_from_triangle() -> Box<ReferenceCellContainer> {
    ReferenceCellContainer(Box::new(Triangle {})).to_box()
}
/// New cell container from quadrilateral.
#[no_mangle]
pub extern "C" fn reference_cell_container_new_from_quadrilateral() -> Box<ReferenceCellContainer> {
    ReferenceCellContainer(Box::new(Quadrilateral {})).to_box()
}

/// New cell container from tetrahedron.
#[no_mangle]
pub extern "C" fn reference_cell_container_new_from_tetrahedron() -> Box<ReferenceCellContainer> {
    ReferenceCellContainer(Box::new(Tetrahedron {})).to_box()
}
/// New cell container from Hexahedron.
#[no_mangle]
pub extern "C" fn reference_cell_container_new_from_hexahedron() -> Box<ReferenceCellContainer> {
    ReferenceCellContainer(Box::new(Hexahedron {})).to_box()
}
/// New cell container from Prism.
#[no_mangle]
pub extern "C" fn reference_cell_container_new_from_prism() -> Box<ReferenceCellContainer> {
    ReferenceCellContainer(Box::new(Prism {})).to_box()
}
/// New cell container from Pyramid.
#[no_mangle]
pub extern "C" fn reference_cell_container_new_from_pyramid() -> Box<ReferenceCellContainer> {
    ReferenceCellContainer(Box::new(Pyramid {})).to_box()
}

/// Destroy a cell container.
#[no_mangle]
pub extern "C" fn reference_cell_container_destroy(_: Option<Box<ReferenceCellContainer>>) {}

/// Get dimension.
#[no_mangle]
pub extern "C" fn reference_cell_container_get_dim(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> size_t {
    get_reference(cell_container).dim() as size_t
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_vertices(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> Box<RustyDataContainer> {
    let cell = get_reference(cell_container);
    RustyDataContainer::from_slice(cell.vertices()).to_box()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_edges(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> Box<RustyDataContainer> {
    let cell = get_reference(cell_container);
    RustyDataContainer::from_slice(cell.edges()).to_box()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_faces(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> Box<RustyDataContainer> {
    let cell = get_reference(cell_container);
    RustyDataContainer::from_slice(cell.faces()).to_box()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_faces_nvertices(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> Box<RustyDataContainer> {
    let cell = get_reference(cell_container);
    RustyDataContainer::from_slice(cell.faces_nvertices()).to_box()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_entity_count(
    dim: usize,
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> usize {
    assert!(dim < 4);
    get_reference(cell_container).entity_count(dim).unwrap()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_vertex_count(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> usize {
    get_reference(cell_container).vertex_count()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_edge_count(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> usize {
    get_reference(cell_container).edge_count()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_face_count(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> usize {
    get_reference(cell_container).face_count()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_volume_count(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> usize {
    get_reference(cell_container).volume_count()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_connectivity(
    entity_dim: usize,
    entity_number: usize,
    connected_dim: usize,
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> Box<RustyDataContainer> {
    let cell = get_reference(cell_container);
    let connectivity = cell.connectivity(entity_dim, entity_number, connected_dim);

    let vec = match connectivity {
        Ok(v) => v,
        Err(()) => Vec::<usize>::new(),
    };
    RustyDataContainer::from_vec(vec).to_box()
}

#[no_mangle]
pub extern "C" fn reference_cell_container_get_cell_type(
    cell_container: Option<Box<ReferenceCellContainer>>,
) -> ReferenceCellType {
    get_reference(cell_container).cell_type()
}
