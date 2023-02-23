use std::vec;

use itertools::Itertools;
use solvers_traits::fmm::{FmmLeafNodeData, FmmNodeData};

use crate::types::{
    morton::MortonKey,
    node::{LeafNode, Node, NodeData},
    point::{Point, Points},
};

impl <'a>FmmNodeData<'a> for Node {
    type CoefficientData = &'a [f64];
    type CoefficientView = &'a [f64];

    fn set_order(&mut self, order: usize) {
        let ncoeffs = 6 * (order - 1).pow(2) + 2;
        self.data.field_size = vec![ncoeffs, ncoeffs];
        self.data.displacement = self
            .data
            .field_size
            .iter()
            .scan(0, |state, &x| {
                let tmp = *state;
                *state += x;
                Some(tmp)
            })
            .collect();

        self.data.raw = vec![0f64; ncoeffs * 2];
        self.data.init = true;
    }

    fn get_local_expansion(&'a self) -> Self::CoefficientView {
        &self.data.raw[self.data.displacement[0]..self.data.displacement[1]]
    }

    fn get_multipole_expansion(&'a self) -> Self::CoefficientView {
        &self.data.raw[self.data.displacement[1]..]
    }

    fn set_local_expansion(&mut self, data: Self::CoefficientData, order: usize) {

        if !self.data.init {
            self.set_order(order);
        }
        for (i, elem) in self.data.raw[self.data.displacement[0]..self.data.displacement[1]]
            .iter_mut()
            .enumerate()
        {
            *elem += data[i]
        }

    } 

    fn set_multipole_expansion(&mut self, data: Self::CoefficientData, order: usize) {
        
        if !self.data.init {
            self.set_order(order);
        }
        
        for (i, elem) in self.data.raw[self.data.displacement[1]..]
            .iter_mut()
            .enumerate()
        {
            *elem += data[i]
        }
    }
}

impl <'a>FmmLeafNodeData<'a> for LeafNode {
    type Points =  Vec<Point>;
    type PointData = f64;
    type PointDataView = &'a f64;
    type PointIndices = &'a Vec<usize>;

    fn get_points(&'a self) -> &'a Self::Points {
        &self.points
    }

    fn get_points_mut(&mut self) -> &mut Self::Points {
        &mut self.points
    }

    fn get_charge(&'a self, index: usize) -> Self::PointDataView {
        let res: Vec<&Point> = self.points.iter().filter(|&p| p.global_idx == index).collect();
        &res[0].data[0]
    }

    fn set_charge(&mut self, index: usize, data: Self::PointData) {
        let mut pts: Vec<&mut Point> = self.points.iter_mut().filter(|p| p.global_idx == index).collect();
        let pt: &mut Point = pts[0];
        pt.data[0] +=  data;
    }

    fn get_potential(&'a self, index: usize) -> Self::PointDataView {
        let res: Vec<&Point> = self.points.iter().filter(|&p| p.global_idx == index).collect();
        &res[0].data[1] 
    }

    fn set_potential(&mut self, index: usize, data: Self::PointData) {
        let mut pts: Vec<&mut Point> = self.points.iter_mut().filter(|p| p.global_idx == index).collect();
        let pt: &mut Point = pts[0];
        pt.data[1] +=  data; 
    }
}
