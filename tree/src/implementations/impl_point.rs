use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use crate::types::point::Point;

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Point {}

impl Ord for Point {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // less_than(&self.morton, &other.morton)
        Some(self.key.cmp(&other.key))
    }
}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand::SeedableRng;

    use crate::types::{
        domain::Domain,
        morton::MortonKey,
        point::{Point, PointType},
    };

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
    pub fn test_ordering() {
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        let mut points: Vec<Point> = points_fixture(1000)
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                key: MortonKey::from_point(p, &domain),
                global_idx: i,
            })
            .collect();

        points.sort();

        for i in 0..(points.len() - 1) {
            let a = points[i];
            let b = points[i + 1];
            assert!(a <= b);
        }
    }
}
