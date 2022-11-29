use crate::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, PointType, Points},
};

pub trait Tree {
    // Create a new tree that is optionally balanced
    pub fn new(points: &[[PointType; 3]], balanced: bool) -> Self;

    // Get balancing information
    pub fn get_balanced() -> bool;

    // Get all keys, gets local keys in multi-node setting
    pub fn get_keys() -> MortonKeys;

    // Get all points, gets local keys in multi-node setting
    pub fn get_points() -> Points;

    // Get domain, gets global domain in multi-node setting
    pub fn get_domain() -> Domain;

    // Get tree node key associated with a given point
    pub fn map_point_to_key(point: &Point) -> &MortonKey;

    // Get points associated with a tree node key
    pub fn map_key_to_points(key: &MortonKey) -> &Points;
}
