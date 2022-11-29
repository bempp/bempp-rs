//! Data Structures and methods to create octrees on a single node.

use std::{
    collections::{HashMap, HashSet},
};

use mpi::{
    topology::{UserCommunicator}
};

use crate::{
    tree::Tree,
    types::{
        morton::{MortonKey, MortonKeys},
        point::{PointType, Point, Points},
        domain::Domain
    }
};

/// Concrete local (non-distributed) Tree.
#[derive(Debug)]
pub struct SingleNodeTree {
    /// Balancing is optional.
    pub balanced: bool,

    ///  A vector of Cartesian points.
    pub points: Points,

    /// The nodes that span the SingleNodeTree, defined by its leaf nodes.
    pub keys: MortonKeys,

    /// Domain spanned by the points in the SingleNodeTree.
    pub domain: Domain,

    /// Map between the points and the nodes in the SingleNodeTree.
    pub points_to_keys: HashMap<Point, MortonKey>,

    /// Map between the nodes in the SingleNodetree and the points they contain.
    pub keys_to_points: HashMap<MortonKey, Points>,
}


/// Create a mapping between points and octree nodes, assumed to overlap.
pub fn assign_points_to_nodes(points: &Points, nodes: &MortonKeys) -> HashMap<Point, MortonKey> {
    let nodes: HashSet<MortonKey> = nodes.iter().cloned().collect();

    let mut map: HashMap<Point, MortonKey> = HashMap::new();

    for point in points.iter() {
        if nodes.contains(&point.key) {
            map.insert(*point, point.key);
        } else {
            let mut ancestors: MortonKeys = MortonKeys{keys: point.key.ancestors().into_iter().collect()};
            ancestors.sort();
            for ancestor in ancestors.keys {
                if nodes.contains(&ancestor) {
                    map.insert(*point, ancestor);
                    break;
                }
            }
        };
    }
    map
}

/// Create a mapping between octree nodes and the points they contain, assumed to overlap.
pub fn assign_nodes_to_points(
    keys: &MortonKeys,
    points: &Points,
) -> HashMap<MortonKey, Points> {
    let keys: HashSet<MortonKey> = keys.iter().cloned().collect();
    let mut map: HashMap<MortonKey, Points> = HashMap::new();

    for point in points.iter() {
        if keys.contains(&point.key) {
            map.entry(point.key).or_insert(Vec::new()).push(*point);
        } else {
            let mut ancestors: MortonKeys = MortonKeys{keys: point.key.ancestors().into_iter().collect()};
            ancestors.sort();

            for ancestor in ancestors.keys {
                if keys.contains(&ancestor) {
                    map.entry(ancestor).or_insert(Vec::new()).push(*point);
                    break;
                }
            }
        }
    }
    map
}

impl SingleNodeTree {

    pub fn new(points: &[[PointType; 3]], balanced: bool) -> SingleNodeTree {

        let domain = Domain::from_local_points(points);
        
        let points: Points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                global_idx: i,
                key: MortonKey::from_point(p, &domain),
            })
            .collect();
        
            let mut keys = MortonKeys { keys: points.iter().map(|p| p.key).collect() };

            if balanced {
                keys.balance();
            }
                            
            let keys_to_points = assign_nodes_to_points(&keys, &points);
            let points_to_keys = assign_points_to_nodes(&points, &keys);
            SingleNodeTree {
                balanced,
                points,
                keys,
                domain,
                points_to_keys,
                keys_to_points
            }
    }
}

impl Tree for SingleNodeTree {

    fn new(points: &[[PointType; 3]], balanced: bool, _comm: Option<UserCommunicator>) -> SingleNodeTree {
        SingleNodeTree::new(points, balanced)
    }

    // Get balancing information
    fn get_balanced(&self) -> bool {
        self.balanced
    }

    // Get all keys, gets local keys in multi-node setting
    fn get_keys(&self) -> &MortonKeys {
        &self.keys
    }

    // Get all points, gets local keys in multi-node setting
    fn get_points(&self) -> &Points {
        &self.points
    }

    // Get domain, gets global domain in multi-node setting
    fn get_domain(&self) -> &Domain {
        &self.domain
    }

    // Get tree node key associated with a given point
    fn map_point_to_key(&self, point: &Point) -> Option<&MortonKey> {
        self.points_to_keys.get(point)
    }

    // Get points associated with a tree node key
    fn map_key_to_points(&self, key: &MortonKey) -> Option<&Points> {
        self.keys_to_points.get(key)
    }
}


mod tests {

    use super::*;

    #[test]
    pub fn test_assign_points_to_nodes() {
        assert!(true);
    }
    
    #[test]
    pub fn test_assign_nodes_to_points() {
        assert!(true);
    }

    #[test]
    pub fn test_unbalanced_tree() {
        assert!(true);
    }
    
    pub fn test_balanced_tree() {
        assert!(true);
    }
}