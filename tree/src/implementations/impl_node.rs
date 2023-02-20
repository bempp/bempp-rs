use solvers_traits::fmm::{FmmLeafNodeData, FmmNodeData};

use crate::types::{
    morton::MortonKey,
    node::{LeafNode, Node, NodeData},
    point::Points,
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

impl FmmNodeData for Node {
    type CoefficientData = Vec<f64>;
    type CoefficientView = Vec<f64>;

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

    fn get_local_expansion(&self) -> Self::CoefficientView {
        self.data.raw[self.data.displacement[0]..self.data.displacement[1]].to_vec()
    }

    fn get_multipole_expansion(&self) -> Self::CoefficientView {
        self.data.raw[self.data.displacement[1]..].to_vec()
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

// TODO: Implement for real
impl FmmLeafNodeData for LeafNode {
    type ParticleData = Vec<f64>;
    type ParticleView = Vec<f64>;

    fn get_leaf_data(&self) -> Self::ParticleView {
        vec![1.]
    }

    fn set_leaf_data(&self, data: &Self::ParticleData) {}
}
