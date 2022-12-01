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
    use rand::Rng;
    use rand::SeedableRng;

    use crate::types::{
        domain::Domain,
        morton::MortonKey,
        point::{Point, PointType, Points},
    };

    #[test]
    fn test_ordering() {
        let npoints = 1000;
        let mut range = rand::thread_rng();
        let mut points: Vec<[PointType; 3]> = Vec::new();

        for _ in 0..npoints {
            points.push([range.gen(), range.gen(), range.gen()]);
        }

        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        let mut points: Points = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                global_idx: i,
                key: MortonKey::from_point(p, &domain),
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
