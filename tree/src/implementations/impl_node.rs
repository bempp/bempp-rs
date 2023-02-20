use itertools::Itertools;
use solvers_traits::fmm::{FmmLeafNodeData, FmmNodeData};

use crate::types::{
    morton::MortonKey,
    node::{LeafNode, Node, NodeData},
    point::{Point, Points},
};

impl Default for LeafNode {
    fn default() -> Self {
        LeafNode {
            key: MortonKey::default(),
            points: Points::default(),
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Node {
            key: MortonKey::default(),
            data: NodeData::default(),
        }
    }
}

impl <'a>FmmNodeData<'a> for Node {
    type CoefficientData = Vec<f64>;
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
    }

    fn get_local_expansion(&'a self) -> Self::CoefficientView {
        &self.data.raw[self.data.displacement[0]..self.data.displacement[1]]
    }

    fn get_multipole_expansion(&'a self) -> Self::CoefficientView {
        &self.data.raw[self.data.displacement[1]..]
    }

    fn set_local_expansion(&mut self, data: &Self::CoefficientData) {
        for (i, elem) in self.data.raw[self.data.displacement[0]..self.data.displacement[1]]
            .iter_mut()
            .enumerate()
        {
            *elem = data[i]
        }
    }

    fn set_multipole_expansion(&mut self, data: &Self::CoefficientData) {
        for (i, elem) in self.data.raw[self.data.displacement[1]..]
            .iter_mut()
            .enumerate()
        {
            *elem = data[i]
        }
    }
}

impl <'a>FmmLeafNodeData<'a> for LeafNode {
    type Points = Vec<&'a Point>;
    type PointData = Vec<f64>;
    type PointDataView = &'a Vec<f64>;
    type PointIndices = Vec<&'a usize>;

    fn get_points(&'a self) -> Self::Points {
        let pts = self.points.iter().collect();
        pts
    }

    fn get_point_indices(&'a self) -> Self::PointIndices {
        let pts: Vec<&usize> = self.points.iter().map(|p| &p.global_idx).collect();
        pts
    }

    fn get_point_data(&'a self, index: usize) -> Self::PointDataView {
        let res: Vec<&Point> = self.points.iter().filter(|&p| p.global_idx == index).collect();
        &res[0].data
    }

    fn set_point_data(&mut self, index: usize, data: Self::PointData) {
        let mut pts: Vec<&mut Point> = self.points.iter_mut().filter(|p| p.global_idx == index).collect();
        let pt: &mut Point = pts[0];
        pt.data =  data;
    }
}
