//! Implementation of traits for handling, and sorting, containers of point data.
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use bempp_traits::types::Scalar;

use crate::types::point::{Point, Points};

impl<T> PartialEq for Point<T>
where
    T: Scalar<Real = T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.encoded_key == other.encoded_key
    }
}

impl<T> Eq for Point<T> where T: Scalar<Real = T> {}

impl<T> Ord for Point<T>
where
    T: Scalar<Real = T>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.encoded_key.cmp(&other.encoded_key)
    }
}

impl<T> PartialOrd for Point<T>
where
    T: Scalar<Real = T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // less_than(&self.morton, &other.morton)
        Some(self.encoded_key.cmp(&other.encoded_key))
    }
}

impl<T> Hash for Point<T>
where
    T: Scalar<Real = T>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.encoded_key.hash(state);
    }
}

impl<T> Points<T>
where
    T: Scalar<Real = T>,
{
    pub fn new() -> Points<T> {
        Points {
            points: Vec::new(),
            index: 0,
        }
    }

    pub fn add(&mut self, item: Point<T>) {
        self.points.push(item);
    }

    pub fn sort(&mut self) {
        self.points.sort();
    }
}

// impl <T>Iterator for Points<T> {
//     type Item = Point<T>;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index >= self.points.len() {
//             return None;
//         }

//         self.index += 1;
//         self.points.get(self.index)
//     }
// }

// impl <T> FromIterator<Point<T>> for Points<T> {
//     fn from_iter<I: IntoIterator<Item = Point<T>>>(iter: I) -> Self {
//         let mut c = Points::new();

//         for i in iter {
//             c.add(i);
//         }
//         c
//     }
// }

#[cfg(test)]
mod test {
    use rand::prelude::*;
    use rand::SeedableRng;

    use crate::constants::DEEPEST_LEVEL;
    use crate::types::{
        domain::Domain,
        morton::MortonKey,
        point::{Point, PointType},
    };

    pub fn points_fixture(npoints: i32) -> Vec<[f64; 3]> {
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points: Vec<[PointType<f64>; 3]> = Vec::new();

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

        let mut points: Vec<Point<f64>> = points_fixture(1000)
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                base_key: MortonKey::from_point(p, &domain, DEEPEST_LEVEL),
                encoded_key: MortonKey::from_point(p, &domain, DEEPEST_LEVEL),
                global_idx: i,
            })
            .collect();

        points.sort();

        for i in 0..(points.len() - 1) {
            let a = &points[i];
            let b = &points[i + 1];
            assert!(a <= b);
        }
    }
}
