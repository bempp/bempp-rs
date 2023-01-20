use solvers_traits::tree::FmmData;

use super::morton::MortonKey;

#[derive(Debug, Clone)]
pub struct NodeData {
    // Number of data fields in vec
    nfields: usize,
    field_size: [usize; nfields],
    data: Vec<f64>
}

impl NodeData {
    pub fn default() -> NodeData {
        NodeData { nfields: 1, data: Vec::<f64>::new(), field_size: [1]}
    }
}


impl FmmData for NodeData {

    type CoefficientData = f64;
    type ParticleData = [f64; 3];
    type NodeIndex = MortonKey;

    fn get_expansion_order(&self) -> usize {
        10
    }

    fn get_particles(&self, node_index: &Self::NodeIndex) -> Option<&Self::ParticleData> {
        
    }

    fn get_local_expansion(&self, node_index: &Self::NodeIndex) -> Option<&Self::CoefficientData> {
        
    }

    fn get_multipole_expansion(&self, node_index: &Self::NodeIndex) -> Option<&Self::CoefficientData> {
        
    }
}



