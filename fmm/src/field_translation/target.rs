use std::collections::HashSet;

use itertools::Itertools;
use rayon::prelude::*;

use bempp_traits::{
    field::SourceToTargetData, fmm::TargetTranslation, kernel::Kernel, tree::FmmTree, tree::Tree,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTreeNew};
use rlst_dense::{
    array::empty_array,
    rlst_dynamic_array2,
    traits::{MultIntoResize, RawAccess, RawAccessMut},
};

use crate::{
    constants::{L2L_MAX_CHUNK_SIZE, NSIBLINGS},
    fmm::KiFmm,
    helpers::find_chunk_size,
    traits::FmmScalar,
};

impl<T, U, V, W> TargetTranslation for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTreeNew<W>> + Send + Sync,
    U: SourceToTargetData<V> + Send + Sync,
    V: Kernel,
    W: FmmScalar,
{
    fn l2l(&self, level: u64) {
        let Some(child_targets) = self.tree.get_target_tree().get_keys(level) else {
            return;
        };

        let parent_sources: HashSet<MortonKey> =
            child_targets.iter().map(|source| source.parent()).collect();
        let mut parent_sources = parent_sources.into_iter().collect_vec();
        parent_sources.sort();
        let nparents = parent_sources.len();
        let mut parent_locals = Vec::new();
        for parent in parent_sources.iter() {
            let parent_index_pointer = *self.level_index_pointer_locals[(level - 1) as usize]
                .get(parent)
                .unwrap();
            let parent_local = &self.level_locals[(level - 1) as usize][parent_index_pointer][0];
            parent_locals.push(parent_local);
        }

        let mut max_chunk_size = nparents;
        if max_chunk_size > L2L_MAX_CHUNK_SIZE {
            max_chunk_size = L2L_MAX_CHUNK_SIZE
        }
        let chunk_size = find_chunk_size(nparents, max_chunk_size);

        let child_locals = &self.level_locals[level as usize];

        parent_locals
            .par_chunks_exact(chunk_size)
            .zip(child_locals.par_chunks_exact(NSIBLINGS * chunk_size))
            .for_each(|(parent_local_pointer_chunk, child_local_pointers_chunk)| {
                let mut parent_locals = rlst_dynamic_array2!(W, [self.ncoeffs, chunk_size]);
                for (chunk_idx, parent_local_pointer) in parent_local_pointer_chunk
                    .iter()
                    .enumerate()
                    .take(chunk_size)
                {
                    parent_locals.data_mut()
                        [chunk_idx * self.ncoeffs..(chunk_idx + 1) * self.ncoeffs]
                        .copy_from_slice(unsafe {
                            std::slice::from_raw_parts_mut(parent_local_pointer.raw, self.ncoeffs)
                        });
                }

                for i in 0..NSIBLINGS {
                    let tmp = empty_array::<W, 2>()
                        .simple_mult_into_resize(self.target_data[i].view(), parent_locals.view());

                    for j in 0..chunk_size {
                        let chunk_displacement = j * NSIBLINGS;
                        let child_displacement = chunk_displacement + i;
                        let child_local = unsafe {
                            std::slice::from_raw_parts_mut(
                                child_local_pointers_chunk[child_displacement][0].raw,
                                self.ncoeffs,
                            )
                        };
                        child_local
                            .iter_mut()
                            .zip(&tmp.data()[j * self.ncoeffs..(j + 1) * self.ncoeffs])
                            .for_each(|(l, t)| *l += *t);
                    }
                }
            });
    }

    fn l2p(&self) {}

    fn m2p(&self) {}

    fn p2p(&self) {}
}
