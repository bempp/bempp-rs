use bempp_traits::{fmm::{
    Fmm, SourceDataTree, SourceTranslation, TargetDataTree, TargetTranslation,
}, tree::AttachedDataTree};
use bempp_tree::types::morton::MortonKey;
use itertools::Itertools;
use std::{collections::HashMap, hash::Hash};
use bempp_tree::types::single_node::SingleNodeTree;
use bempp_tree::types::point::Point;
use bempp_traits::tree::Tree;
use bempp_tree::types::domain::Domain;

pub struct FmmDataTree {
    multipoles: HashMap<MortonKey, Vec<f64>>,
    locals: HashMap<MortonKey, Vec<f64>>,
    potentials: HashMap<MortonKey, Vec<f64>>,
    points: HashMap<MortonKey, Vec<Point>>

}

pub struct KiFmm {}

impl FmmDataTree {

    fn new<'a>(
        &self, 
        tree: SingleNodeTree
    ) -> Self {

        let mut multipoles = HashMap::new();
        let mut locals = HashMap::new();
        let mut potentials = HashMap::new();
        let mut points = HashMap::new();

        for level in (0..=tree.get_depth()).rev() {

            if let Some(keys) = tree.get_keys(level) {
                for key in keys.iter() {
                    multipoles.insert(key.clone(), Vec::new());
                    locals.insert(key.clone(), Vec::new()); 
                    potentials.insert(key.clone(), Vec::new());
                    if let Some(point_data) = tree.get_points(key) {
                        points.insert(key.clone(), point_data.iter().cloned().collect_vec());
                    }

                }
            }
        }

        Self { multipoles, locals, potentials, points }
    }
}

impl SourceDataTree for FmmDataTree {
    type Tree = SingleNodeTree;
    type Coefficient = f64;
    type Coefficients<'a> = &'a [f64];

    fn get_multipole_expansion<'a>(
            &'a self,
            key: &<Self::Tree as Tree>::NodeIndex,
        ) -> Option<Self::Coefficients<'a>> {
        if let Some(multipole) = self.multipoles.get(key) {
            Some(multipole.as_slice())
        } else {
            None
        }
    }

    fn set_multipole_expansion<'a>(
            &'a mut self,
            key: &<Self::Tree as Tree>::NodeIndex,
            coefficients: &Self::Coefficients<'a>,
        ) {
        
        if let Some(multipole) = self.multipoles.get_mut(key) {
            if !multipole.is_empty() {
                for (curr, &new) in multipole.iter_mut().zip(coefficients.iter()) {
                    *curr += new;
                }
            } else {
                *multipole = coefficients.clone().to_vec();
            }
        }
    }

    fn get_points<'a>(
            &'a self,
            key: &<Self::Tree as Tree>::NodeIndex,
        ) -> Option<<Self::Tree as Tree>::PointSlice<'a>> {
        
        if let Some(points) = self.points.get(key) {
            Some(points.as_slice())
        } else {
            None
        }
    }
}

// impl TargetDataTree for FmmDataTree;



// impl Fmm for KiFMM {};
// impl SourceTranslation for FmmDataTree;
// impl TargetTranslation for FmmDataTree;

