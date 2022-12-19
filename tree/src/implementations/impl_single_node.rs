use std::collections::{HashMap, HashSet};

use solvers_traits::tree::Tree;

use crate::{
    constants::{DEEPEST_LEVEL, NCRIT},
    implementations::impl_morton::{encode_anchor, point_to_anchor},
    types::{
        domain::Domain,
        morton::{MortonKey, MortonKeys},
        point::{Point, PointType, Points},
        single_node::SingleNodeTree,
    },
};

/// Create a mapping between points and octree nodes, assumed to overlap.
pub fn assign_points_to_nodes(points: &Points, nodes: &MortonKeys) -> HashMap<Point, MortonKey> {
    let nodes: HashSet<MortonKey> = nodes.iter().cloned().collect();

    let mut map: HashMap<Point, MortonKey> = HashMap::new();

    for point in points.iter() {
        if nodes.contains(&point.key) {
            map.insert(*point, point.key);
        } else {
            let ancestors = point.key.ancestors();

            for ancestor in ancestors.iter() {
                if nodes.contains(ancestor) {
                    map.insert(*point, *ancestor);
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
                index: 0,
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
    /// Create a new single-node tree. In non-adaptive (uniform) trees are created, they are specified
    /// by a user defined maximum depth, if an adaptive tree is created it is specified by only by the
    /// user defined maximum leaf maximum occupancy n_crit.
    pub fn new(
        points: &[[PointType; 3]],
        adaptive: bool,
        n_crit: Option<usize>,
        depth: Option<u64>,
    ) -> SingleNodeTree {
        let domain = Domain::from_local_points(points);

        // Encode to max user specified depth, if specified, otherwise encode an adaptive tree using n_crit
        let encoded_points: Points;
        let mut encoded_keys: MortonKeys;

        let n_crit = n_crit.unwrap_or(NCRIT);
        let depth = depth.unwrap_or(DEEPEST_LEVEL);

        if !adaptive {
            encoded_keys = MortonKeys {
                keys: points
                    .iter()
                    .map(|p| {
                        let anchor = point_to_anchor(p, depth, &domain).unwrap();
                        MortonKey {
                            morton: encode_anchor(&anchor, depth),
                            anchor,
                        }
                    })
                    .collect(),
                index: 0,
            };

            encoded_points = encoded_keys
                .iter()
                .zip(points)
                .enumerate()
                .map(|(index, (key, point))| Point {
                    coordinate: *point,
                    global_idx: index,
                    key: *key,
                })
                .collect();

            encoded_keys.linearize();
        } else {
            // If adaptive tree, can continue globbing, must also balance
            let mut level = DEEPEST_LEVEL;
            let mut curr = MortonKeys {
                keys: points
                    .iter()
                    .map(|p| MortonKey::from_point(p, &domain))
                    .collect(),
                index: 0,
            };

            loop {
                // Gather hashmap of globbable parents
                let globbable = curr
                    .iter()
                    .fold(HashMap::<MortonKey, usize>::new(), |mut m, x| {
                        *m.entry(x.parent()).or_default() += 1;
                        m
                    });

                let new = MortonKeys {
                    keys: curr
                        .iter()
                        .map(|&key| {
                            let parent = key.parent();
                            if *globbable.get(&parent).unwrap() < n_crit {
                                parent
                            } else {
                                key
                            }
                        })
                        .collect(),
                    index: 0,
                };
                level -= 1;
                // Break loop if can't glob anymore
                if level == 0 || globbable.values().all(|&v| v >= n_crit) {
                    encoded_keys = curr;
                    break;
                }
                curr = new;
            }

            encoded_points = encoded_keys
                .iter()
                .zip(points)
                .enumerate()
                .map(|(index, (key, point))| Point {
                    coordinate: *point,
                    global_idx: index,
                    key: *key,
                })
                .collect();

            // Balance and linearize adaptive tree
            encoded_keys.linearize();
            encoded_keys.balance();
        }

        let keys_to_points = assign_nodes_to_points(&encoded_keys, &encoded_points);
        let points_to_keys = assign_points_to_nodes(&encoded_points, &encoded_keys);
        SingleNodeTree {
            adaptive,
            points: encoded_points,
            keys: encoded_keys,
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

    // Get adaptivity information
    fn get_adaptive(&self) -> bool {
        self.adaptive
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

    pub fn points_fixture(npoints: i32) -> Vec<[f64; 3]> {
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points: Vec<[PointType; 3]> = Vec::new();

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
    pub fn test_uniform_tree() {
        let points = points_fixture(10000);
        let depth = 4;
        let n_crit = 15;
        let tree = SingleNodeTree::new(&points, false, Some(n_crit), Some(depth));

        // Test that particle constraint is met at this level
        for (_, (_, points)) in tree.keys_to_points.iter().enumerate() {
            assert!(points.len() <= n_crit);
        }

        // Test that the tree really is uniform
        let levels: Vec<u64> = tree.get_keys().iter().map(|key| key.level()).collect();
        let first = levels[0];
        assert!(levels.iter().all(|key| *key == first));

        // Test that max level constraint is satisfied
        assert!(first == depth);
    }

    #[test]
    pub fn test_adaptive_tree() {
        let points = points_fixture(10000);
        let adaptive = true;
        let n_crit = 15;
        let tree = SingleNodeTree::new(&points, adaptive, Some(n_crit), None);

        // Test that particle constraint is met
        for (_, (_, points)) in tree.keys_to_points.iter().enumerate() {
            assert!(points.len() <= n_crit);
        }

        // Test that tree is not uniform
        let levels: Vec<u64> = tree.get_keys().iter().map(|key| key.level()).collect();
        let first = levels[0];
        assert_eq!(false, levels.iter().all(|level| *level == first));
    }

    pub fn test_no_overlaps_helper(tree: &SingleNodeTree) {
        let tree_set: HashSet<MortonKey> = tree.get_keys().clone().collect();

        for node in tree_set.iter() {
            let ancestors = node.ancestors();
            let int: Vec<&MortonKey> = tree_set.intersection(&ancestors).collect();
            assert!(int.len() == 1);
        }
    }
    #[test]
    pub fn test_no_overlaps() {
        let points = points_fixture(10000);
        let uniform = SingleNodeTree::new(&points, false, Some(150), Some(4));
        let adaptive = SingleNodeTree::new(&points, true, Some(150), None);
        test_no_overlaps_helper(&uniform);
        test_no_overlaps_helper(&adaptive);
    }
}
