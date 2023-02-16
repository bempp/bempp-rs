#[derive(Debug, Clone)]
pub struct NodeData {
    pub field_size: Vec<usize>,
    pub data: Vec<f64>,
    pub displacement: Vec<usize>,
}
