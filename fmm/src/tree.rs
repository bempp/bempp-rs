use bempp_traits::{
    fmm::InteractionLists,
    tree::{FmmTree, Tree},
};
use bempp_tree::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    single_node::SingleNodeTreeNew,
};

use crate::traits::FmmScalar;

#[derive(Default)]
pub struct SingleNodeFmmTree<T: FmmScalar + Send + Sync> {
    pub source_tree: SingleNodeTreeNew<T>,
    pub target_tree: SingleNodeTreeNew<T>,
    pub domain: Domain<T>,
}

impl<T> FmmTree for SingleNodeFmmTree<T>
where
    T: FmmScalar,
{
    type Precision = T;
    type NodeIndex = MortonKey;
    type Tree = SingleNodeTreeNew<T>;

    fn get_source_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn get_target_tree(&self) -> &Self::Tree {
        &self.target_tree
    }

    fn get_domain(&self) -> &<Self::Tree as Tree>::Domain {
        &self.domain
    }

    fn get_near_field(&self, leaf: &Self::NodeIndex) -> Option<Vec<Self::NodeIndex>> {
        let mut u_list = Vec::new();
        let neighbours = leaf.neighbors();

        // Child level
        let mut neighbors_children_adj = neighbours
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| {
                self.get_source_tree()
                    .get_all_keys_set()
                    .unwrap()
                    .contains(nc)
                    && leaf.is_adjacent(nc)
            })
            .collect();

        // Key level
        let mut neighbors_adj = neighbours
            .iter()
            .filter(|n| {
                self.get_source_tree()
                    .get_all_keys_set()
                    .unwrap()
                    .contains(n)
                    && leaf.is_adjacent(n)
            })
            .cloned()
            .collect();

        // Parent level
        let mut parent_neighbours_adj = leaf
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| {
                self.get_source_tree()
                    .get_all_keys_set()
                    .unwrap()
                    .contains(pn)
                    && leaf.is_adjacent(pn)
            })
            .collect();

        u_list.append(&mut neighbors_children_adj);
        u_list.append(&mut neighbors_adj);
        u_list.append(&mut parent_neighbours_adj);
        u_list.push(*leaf);

        if !u_list.is_empty() {
            Some(u_list)
        } else {
            None
        }
    }
}

unsafe impl<T: FmmScalar> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: FmmScalar> Sync for SingleNodeFmmTree<T> {}
