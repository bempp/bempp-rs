use solvers_traits::fmm::FmmData;

use crate::types::data::NodeData;

impl FmmData for NodeData {
    type CoefficientData = Vec<f64>;
    type CoefficientView = Vec<f64>;
    type CoefficientViewMut = Vec<f64>;

    fn new(order: usize) -> NodeData {
        let ncoeffs = 6 * (order - 1).pow(2) + 2;
        let field_size = vec![ncoeffs, ncoeffs];

        let displacement = field_size
            .iter()
            .scan(0, |state, &x| {
                let tmp = *state;
                *state += x;
                Some(tmp)
            })
            .collect();

        let data = vec![0f64; ncoeffs * 2];

        NodeData {field_size, displacement, data}

    }

    fn get_local_expansion(&self) -> Self::CoefficientView {
        self.data[self.displacement[0]..self.displacement[1]].to_vec()
    }
    fn get_local_expansion_mut(&self) -> Self::CoefficientView {
        self.data[self.displacement[0]..self.displacement[1]].to_vec()
    }

    fn get_multipole_expansion(&self) -> Self::CoefficientView {
        self.data[self.displacement[1]..].to_vec()
    }
    
    fn get_multipole_expansion_mut(&self) -> Self::CoefficientViewMut {
        self.data[self.displacement[1]..].to_vec()
    }

    fn set_local_expansion(&mut self, data: &Self::CoefficientData) {
        for (i, elem) in self.data[self.displacement[0]..self.displacement[1]]
            .iter_mut()
            .enumerate()
        {
            *elem = data[i]
        }
    }

    fn set_multipole_expansion(&mut self, data: &Self::CoefficientData) {
        for (i, elem) in self.data[self.displacement[1]..].iter_mut().enumerate() {
            *elem = data[i]
        }
    }
}

mod test {
    use solvers_traits::fmm::FmmData;

    use super::NodeData;

    #[test]
    fn test_fmm_node_data() {
        let order = 5;
        let mut data = NodeData::new(order);
    }
}
