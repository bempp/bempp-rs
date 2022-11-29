use mpi::topology::UserCommunicator;

use crate::{
    types::{
        point::{PointType, Point, Points},
        morton::{MortonKey, MortonKeys},
        domain::Domain
    }
};

pub trait Tree {
    // Create a new tree that is optionally balanced
    fn new(points: &[[PointType; 3]], balanced: bool, comm: Option<UserCommunicator>) -> Self;

    // Get balancing information
    fn get_balanced(&self) -> bool;

    // Get all keys, gets local keys in multi-node setting
    fn get_keys(&self) -> &MortonKeys;

    // Get all points, gets local keys in multi-node setting
    fn get_points(&self) -> &Points;

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Domain;

    // Get tree node key associated with a given point
    fn map_point_to_key(&self, point: &Point) -> Option<&MortonKey>;

    // Get points associated with a tree node key
    fn map_key_to_points(&self, key: &MortonKey) -> Option<&Points>;
}