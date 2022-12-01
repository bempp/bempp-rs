use std::collections::{HashMap, HashSet};

use solvers_traits::tree::Tree;

use crate::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, PointType, Points},
    single_node::SingleNodeTree,
};

/// Create a mapping between points and octree nodes, assumed to overlap.
pub fn assign_points_to_nodes(points: &Points, nodes: &MortonKeys) -> HashMap<Point, MortonKey> {
    let nodes: HashSet<MortonKey> = nodes.iter().cloned().collect();

    let mut map: HashMap<Point, MortonKey> = HashMap::new();

    for point in points.iter() {
        if nodes.contains(&point.key) {
            map.insert(*point, point.key);
        } else {
            let mut ancestors: MortonKeys = MortonKeys {
                keys: point.key.ancestors().into_iter().collect(),
            };
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
pub fn assign_nodes_to_points(keys: &MortonKeys, points: &Points) -> HashMap<MortonKey, Points> {
    let keys: HashSet<MortonKey> = keys.iter().cloned().collect();
    let mut map: HashMap<MortonKey, Points> = HashMap::new();

    for point in points.iter() {
        if keys.contains(&point.key) {
            map.entry(point.key).or_default().push(*point);
        } else {
            let mut ancestors: MortonKeys = MortonKeys {
                keys: point.key.ancestors().into_iter().collect(),
            };
            ancestors.sort();

            for ancestor in ancestors.keys {
                if keys.contains(&ancestor) {
                    map.entry(ancestor).or_default().push(*point);
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

        let mut keys = MortonKeys {
            keys: points.iter().map(|p| p.key).collect(),
        };

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
            keys_to_points,
        }
    }
}

impl Tree for SingleNodeTree {
    type Domain = Domain;
    type Point = Point;
    type Points = Points;
    type NodeIndex = MortonKey;
    type NodeIndices = MortonKeys;

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

#[cfg(test)]
mod tests {

    use super::*;
    use rand::prelude::*;
    use rand::SeedableRng;

    /// Test fixture for NPOINTS randomly distributed points.
    fn points_fixture(npoints: usize) -> Vec<[f64; 3]> {
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points = Vec::new();

        for _ in 0..npoints {
            points.push([
                between.sample(&mut range),
                between.sample(&mut range),
                between.sample(&mut range),
            ])
        }
        points
    }

    #[test]
    pub fn test_assign_points_to_nodes() {
        let points = points_fixture(1000);
        let unbalanced = SingleNodeTree::new(&points, false);
        let balanced = SingleNodeTree::new(&points, true);

        // Test that all points have been assigned
        let keys = unbalanced.get_keys();
        let mut npoints = 0;
        for key in keys.iter() {
            npoints += unbalanced.map_key_to_points(&key).unwrap().len();
        }
        assert!(npoints == unbalanced.get_points().len());

        let keys = balanced.get_keys();
        let mut npoints = 0;
        for key in keys.iter() {
            npoints += balanced.map_key_to_points(&key).unwrap().len();
        }
        assert!(npoints == balanced.get_points().len());
    }
}
