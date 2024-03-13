//! Tree
use bempp_traits::tree::{FmmTree, Tree};
use bempp_tree::types::{domain::Domain, morton::MortonKey, single_node::SingleNodeTree};
use num::Float;
use rlst_dense::types::RlstScalar;

/// A struct that holds the single node trees associated with both sources and targets as well as their shared domain.
#[derive(Default)]
pub struct SingleNodeFmmTree<T: RlstScalar<Real = T> + Float + Default> {
    /// Source tree
    pub source_tree: SingleNodeTree<T>,
    /// Target tree
    pub target_tree: SingleNodeTree<T>,
    /// Domain
    pub domain: Domain<T>,
}

impl<T> FmmTree for SingleNodeFmmTree<T>
where
    T: RlstScalar<Real = T> + Float + Default,
{
    type Precision = T;
    type Node = MortonKey;
    type Tree = SingleNodeTree<T>;

    fn source_tree(&self) -> &Self::Tree {
        &self.source_tree
    }

    fn target_tree(&self) -> &Self::Tree {
        &self.target_tree
    }

    fn domain(&self) -> &<Self::Tree as Tree>::Domain {
        &self.domain
    }

    fn near_field(&self, leaf: &Self::Node) -> Option<Vec<Self::Node>> {
        let mut u_list = Vec::new();
        let neighbours = leaf.neighbors();

        // Child level
        let mut neighbors_children_adj = neighbours
            .iter()
            .flat_map(|n| n.children())
            .filter(|nc| {
                self.source_tree().all_keys_set().unwrap().contains(nc) && leaf.is_adjacent(nc)
            })
            .collect();

        // Key level
        let mut neighbors_adj = neighbours
            .iter()
            .filter(|n| {
                self.source_tree().all_keys_set().unwrap().contains(n) && leaf.is_adjacent(n)
            })
            .cloned()
            .collect();

        // Parent level
        let mut parent_neighbours_adj = leaf
            .parent()
            .neighbors()
            .into_iter()
            .filter(|pn| {
                self.source_tree().all_keys_set().unwrap().contains(pn) && leaf.is_adjacent(pn)
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

unsafe impl<T: RlstScalar<Real = T> + Default + Float> Send for SingleNodeFmmTree<T> {}
unsafe impl<T: RlstScalar<Real = T> + Default + Float> Sync for SingleNodeFmmTree<T> {}
