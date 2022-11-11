//! Wrappers for Distributed Tree interface
use mpi::{ffi::MPI_Comm, topology::UserCommunicator, traits::*};
use std::ffi::CStr;
use std::os::raw::c_char;

use crate::{
    distributed::DistributedTree,
    types::{
        morton::MortonKey,
        point::{Point, PointType},
    },
};

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_from_points(
    p_points: *const [PointType; 3],
    npoints: usize,
    balanced: bool,
    world: *mut usize,
) -> *mut DistributedTree {
    let points = std::slice::from_raw_parts(p_points, npoints);
    let world = std::mem::ManuallyDrop::new(
        UserCommunicator::from_raw(*(world as *const MPI_Comm)).unwrap()
    );
    Box::into_raw(Box::new(DistributedTree::new(points, balanced, &world)))
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_nkeys(p_tree: *const DistributedTree) -> usize {
    let tree = &*p_tree;
    tree.keys.len()
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_npoints(p_tree: *const DistributedTree) -> usize {
    let tree = &*p_tree;
    tree.points.len()
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_keys(p_tree: *const DistributedTree) -> *const MortonKey {
    let tree = &*p_tree;
    tree.keys.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_points(p_tree: *const DistributedTree) -> *const Point {
    let tree = &*p_tree;
    tree.points.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_balanced(p_tree: *const DistributedTree) -> bool {
    let tree = &*p_tree;
    tree.balanced
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_points_to_keys_get(p_tree: *const DistributedTree, p_point: *const Point) -> *mut MortonKey {
    let tree = &*p_tree;
    let point = *p_point;
    Box::into_raw(Box::new(*tree.points_to_keys.get(&point).unwrap()))
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_keys_to_npoints_get(p_tree: *const DistributedTree, p_key: *const MortonKey) -> usize {
    let tree = &*p_tree;
    let key = *p_key;
    tree.keys_to_points.get(&key).unwrap().len()
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_keys_to_points_get(p_tree: *const DistributedTree, p_key: *const MortonKey) -> *const Point {
    let tree = &*p_tree;
    let key = *p_key;
    let points = tree.keys_to_points.get(&key).unwrap();
    points.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_write_vtk(
    world: *mut usize,
    p_tree: *const DistributedTree,
    p_filename: *mut c_char,
) {
    let c_filename = CStr::from_ptr(p_filename);
    let filename_slice: &str = c_filename.to_str().unwrap();
    let filename = filename_slice.to_string();

    let tree = &*p_tree;
    let world = std::mem::ManuallyDrop::new(
        UserCommunicator::from_raw(*(world as *const MPI_Comm)).unwrap()
    );

    DistributedTree::write_vtk(&world, filename, tree);
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_write_hdf5(
    world: *mut usize,
    p_tree: *const DistributedTree,
    p_filename: *mut c_char,
) {
    let c_filename = CStr::from_ptr(p_filename);
    let filename_slice: &str = c_filename.to_str().unwrap();
    let filename = filename_slice.to_string();

    let tree = &*p_tree;

    let world = std::mem::ManuallyDrop::new(
        UserCommunicator::from_raw(*(world as *const MPI_Comm)).unwrap()
    );
    DistributedTree::write_hdf5(&world, filename, tree).unwrap();
}

#[no_mangle]
pub unsafe extern "C" fn distributed_tree_read_hdf5(
    world: *mut usize,
    p_filepath: *mut c_char,
) -> *mut DistributedTree {
    let c_filepath = CStr::from_ptr(p_filepath);
    let filepath_slice: &str = c_filepath.to_str().unwrap();
    let filepath = filepath_slice.to_string();

    let world = std::mem::ManuallyDrop::new(
        UserCommunicator::from_raw(*(world as *const MPI_Comm)).unwrap()
    );

    Box::into_raw(Box::new(DistributedTree::read_hdf5(&world, filepath)))
}
