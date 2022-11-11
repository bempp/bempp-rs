//! Wrappers for methods on Morton Keys.

use crate::types::{
    domain::Domain,
    morton::{KeyType, MortonKey},
    point::PointType,
};

#[no_mangle]
pub unsafe extern "C" fn morton_key_from_anchor(p_anchor: *const [KeyType; 3]) -> *mut MortonKey {
    let anchor: &[KeyType; 3] = p_anchor.as_ref().unwrap();
    Box::into_raw(Box::new(MortonKey::from_anchor(anchor)))
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_from_morton(morton: KeyType) -> *mut MortonKey {
    Box::into_raw(Box::new(MortonKey::from_morton(morton)))
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_from_point(
    p_point: *const [PointType; 3],
    p_origin: *const [PointType; 3],
    p_diameter: *const [PointType; 3],
) -> *mut MortonKey {
    let point: &[PointType; 3] = p_point.as_ref().unwrap();
    let origin: &[PointType; 3] = p_origin.as_ref().unwrap();
    let diameter: &[PointType; 3] = p_diameter.as_ref().unwrap();

    let domain = Domain {
        origin: origin.to_owned(),
        diameter: diameter.to_owned(),
    };

    Box::into_raw(Box::new(MortonKey::from_point(point, &domain)))
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_parent(p_morton: *mut MortonKey) -> *mut MortonKey {
    let parent = (*p_morton).parent();
    Box::into_raw(Box::new(parent))
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_level(p_morton: *mut MortonKey) -> KeyType {
    (*p_morton).level()
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_first_child(p_morton: *mut MortonKey) -> *mut MortonKey {
    let first_child = (*p_morton).first_child();
    Box::into_raw(Box::new(first_child))
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_children(p_morton: *mut MortonKey, ptr: *mut usize) {
    let mut children_vec = (*p_morton).children();

    let children_boxes = std::slice::from_raw_parts_mut(ptr, 8);
    for index in 0..8 {
        let child = children_vec.pop().unwrap();
        children_boxes[7 - index] = Box::into_raw(Box::new(child)) as usize;
    }
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_to_coordinates(
    p_morton: *mut MortonKey,
    p_origin: *const [PointType; 3],
    p_diameter: *const [PointType; 3],
    p_coord: *mut [PointType; 3],
) {
    let origin: &[PointType; 3] =  p_origin.as_ref().unwrap();
    let diameter: &[PointType; 3] = p_diameter.as_ref().unwrap();

    let domain = Domain {
        origin: origin.to_owned(),
        diameter: diameter.to_owned(),
    };

    let tmp = (*p_morton).to_coordinates(&domain);

    for index in 0..3 {
        (*p_coord)[index] = tmp[index]
    }
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_box_coordinates(
    p_morton: *mut MortonKey,
    p_origin: *const [PointType; 3],
    p_diameter: *const [PointType; 3],
    box_coord: *mut [PointType; 24],
) {
    let origin: &[PointType; 3] =  p_origin.as_ref().unwrap();
    let diameter: &[PointType; 3] = p_diameter.as_ref().unwrap();

    let domain = Domain {
        origin: origin.to_owned(),
        diameter: diameter.to_owned(),
    };

    let coords = (*p_morton).box_coordinates(&domain);

    for index in 0..24 {
        (*box_coord)[index] = coords[index];
    }
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_key_in_direction(
    p_morton: *mut MortonKey,
    p_direction: *const [i64; 3],
) -> *mut MortonKey {
    let direction = p_direction.as_ref().unwrap();

    let shifted_key = (*p_morton).find_key_in_direction(direction);

    match shifted_key {
        Some(key) => Box::into_raw(Box::new(key)),

        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_is_ancestor(
    p_morton: *mut MortonKey,
    p_other: *mut MortonKey,
) -> bool {
    (*p_morton).is_ancestor(&*p_other)
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_is_descendent(
    p_morton: *mut MortonKey,
    p_other: *mut MortonKey,
) -> bool {
    (*p_morton).is_descendent(&*p_other)
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_delete(p_morton_key: *mut MortonKey) {
        drop(Box::from_raw(p_morton_key));
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_next(ptr: *const MortonKey) -> *mut &'static MortonKey {
    let mut slice = std::slice::from_raw_parts(ptr, 2).iter();
    slice.next();
    let next = slice.next().unwrap();
    Box::into_raw(Box::new(next))
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_clone(
    ptr: *const MortonKey,
    data_ptr: *mut usize,
    len: usize,
    start: usize,
    stop: usize,
) {
    let slice = std::slice::from_raw_parts(ptr, len);

    let nslice = stop - start;
    let boxes = std::slice::from_raw_parts_mut(data_ptr, nslice);

    for (jdx, idx) in (start..stop).enumerate() {
        boxes[jdx] = Box::into_raw(Box::new(slice[idx])) as usize;
    }
}

#[no_mangle]
pub unsafe extern "C" fn morton_key_index(
    ptr: *const MortonKey,
    len: usize,
    idx: usize,
) -> *mut &'static MortonKey {
    let slice = std::slice::from_raw_parts(ptr, len);
    Box::into_raw(Box::new(&slice[idx]))
}
