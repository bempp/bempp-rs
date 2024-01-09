//! Data structures FMM data and metadata.
use std::collections::HashMap;

use bempp_traits::{field::FieldTranslationData, fmm::Fmm, kernel::Kernel, tree::Tree};
use bempp_tree::types::morton::MortonKey;
use cauchy::Scalar;
use num::{Complex, Float};
use rlst_dense::{array::Array, base_array::BaseArray, data_container::VectorContainer};

/// Type alias for charge data
pub type Charge<T> = T;

/// Type alias for global index for identifying charge data with a point
pub type GlobalIdx = usize;

/// Type alias for mapping charge data to global indices.
pub type ChargeDict<T> = HashMap<GlobalIdx, Charge<T>>;

/// Type alias for approximation of FMM operator matrices.
pub type C2EType<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

pub struct FmmDataUniform<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
{
    /// The associated FMM object, which implements an FMM interface
    pub fmm: T,

    /// The multipole expansion data at each box.
    pub multipoles: Vec<U>,

    /// Multipole expansions at leaf level
    pub leaf_multipoles: Vec<SendPtrMut<U>>,

    /// Multipole expansions at each level
    pub level_multipoles: Vec<Vec<SendPtrMut<U>>>,

    /// The local expansion at each box
    pub locals: Vec<U>,

    /// Local expansions at the leaf level
    pub leaf_locals: Vec<SendPtrMut<U>>,

    /// The local expansion data at each level.
    pub level_locals: Vec<Vec<SendPtrMut<U>>>,

    /// Index pointers to each key at a given level, indexed by level.
    pub level_index_pointer: Vec<HashMap<MortonKey, usize>>,

    /// The evaluated potentials at each leaf box.
    pub potentials: Vec<U>,

    /// The evaluated potentials at each leaf box.
    pub potentials_send_pointers: Vec<SendPtrMut<U>>,

    /// All upward surfaces
    pub upward_surfaces: Vec<U>,

    /// All downward surfaces
    pub downward_surfaces: Vec<U>,

    /// Leaf upward surfaces
    pub leaf_upward_surfaces: Vec<U>,

    /// Leaf downward surfaces
    pub leaf_downward_surfaces: Vec<U>,

    /// The charge data at each leaf box.
    pub charges: Vec<U>,

    /// Index pointer between leaf keys and charges
    pub charge_index_pointer: Vec<(usize, usize)>,

    /// Scales of each leaf operator
    pub scales: Vec<U>,

    /// Global indices of each charge
    pub global_indices: Vec<usize>,
}

/// Testing for Adaptive trees
pub struct FmmDataAdaptive<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
{
    /// The associated FMM object, which implements an FMM interface
    pub fmm: T,

    /// The multipole expansion data at each box.
    pub multipoles: Vec<U>,

    /// Multipole expansions at leaf level
    pub leaf_multipoles: Vec<SendPtrMut<U>>,

    /// Multipole expansions at each level
    pub level_multipoles: Vec<Vec<SendPtrMut<U>>>,

    /// The local expansion at each box
    pub locals: Vec<U>,

    /// Local expansions at the leaf level
    pub leaf_locals: Vec<SendPtrMut<U>>,

    /// The local expansion data at each level.
    pub level_locals: Vec<Vec<SendPtrMut<U>>>,

    /// Index pointers to each key at a given level, indexed by level.
    pub level_index_pointer: Vec<HashMap<MortonKey, usize>>,

    /// The evaluated potentials at each leaf box.
    pub potentials: Vec<U>,

    /// The evaluated potentials at each leaf box.
    pub potentials_send_pointers: Vec<SendPtrMut<U>>,

    /// All upward surfaces
    pub upward_surfaces: Vec<U>,

    /// All downward surfaces
    pub downward_surfaces: Vec<U>,

    /// Leaf upward surfaces
    pub leaf_upward_surfaces: Vec<U>,

    /// Leaf downward surfaces
    pub leaf_downward_surfaces: Vec<U>,

    /// The charge data at each leaf box.
    pub charges: Vec<U>,

    /// Index pointer between leaf keys and charges
    pub charge_index_pointer: Vec<(usize, usize)>,

    /// Scales of each leaf operator
    pub scales: Vec<U>,

    /// Global indices of each charge
    pub global_indices: Vec<usize>,
}

/// Type to store data associated with the kernel independent (KiFMM) in.
pub struct KiFmmLinear<T, U, V, W>
where
    T: Tree,
    U: Kernel<T = W>,
    V: FieldTranslationData<U>,
    W: Scalar + Float + Default,
{
    /// The expansion order
    pub order: usize,

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_1: C2EType<W>,
    pub uc2e_inv_2: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_1: C2EType<W>,
    pub dc2e_inv_2: C2EType<W>,

    /// The ratio of the inner check surface diamater in comparison to the surface discretising a box.
    pub alpha_inner: W,

    /// The ratio of the outer check surface diamater in comparison to the surface discretising a box.
    pub alpha_outer: W,

    /// The multipole to multipole operator matrices, each index is associated with a child box (in sequential Morton order),
    pub m2m: C2EType<W>,

    /// The local to local operator matrices, each index is associated with a child box (in sequential Morton order).
    pub l2l: Vec<C2EType<W>>,

    /// The tree (single or multi node) associated with this FMM
    pub tree: T,

    /// The kernel associated with this FMM.
    pub kernel: U,

    /// The M2L operator matrices, as well as metadata associated with this FMM.
    pub m2l: V,
}

/// A threadsafe mutable raw pointer
#[derive(Clone, Debug, Copy)]
pub struct SendPtrMut<T> {
    pub raw: *mut T,
}

unsafe impl<T> Sync for SendPtrMut<T> {}
unsafe impl<T> Send for SendPtrMut<Complex<T>> {}

impl<T> Default for SendPtrMut<T> {
    fn default() -> Self {
        SendPtrMut {
            raw: std::ptr::null_mut(),
        }
    }
}

/// A threadsafe raw pointer
#[derive(Clone, Debug, Copy)]
pub struct SendPtr<T> {
    pub raw: *const T,
}

unsafe impl<T> Sync for SendPtr<T> {}

impl<T> Default for SendPtr<T> {
    fn default() -> Self {
        SendPtr {
            raw: std::ptr::null(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        charge::build_charge_dict,
        types::{FmmDataUniform, KiFmmLinear},
    };
    use bempp_field::types::FftFieldTranslationKiFmm;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_traits::field::FieldTranslationData;
    use bempp_traits::fmm::Fmm;
    use bempp_traits::tree::Tree;
    use bempp_tree::{
        implementations::helpers::points_fixture, types::single_node::SingleNodeTree,
    };
    use itertools::Itertools;
    use rlst_dense::traits::RawAccess;

    #[test]
    fn test_fmm_data_uniform() {
        let npoints = 10000;
        let points = points_fixture::<f64>(npoints, None, None);
        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 8;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let ncrit = 150;
        let depth = 3;

        // Uniform trees
        {
            let adaptive = false;
            let kernel = Laplace3dKernel::default();

            let tree = SingleNodeTree::new(
                points.data(),
                adaptive,
                Some(ncrit),
                Some(depth),
                &global_idxs[..],
                false,
            );

            let m2l_data_fft = FftFieldTranslationKiFmm::new(
                kernel.clone(),
                order,
                *tree.get_domain(),
                alpha_inner,
            );

            let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data_fft);
            // Form charge dict, matching charges with their associated global indices
            let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

            let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

            let ncoeffs = datatree.fmm.m2l.ncoeffs(order);
            let nleaves = datatree.fmm.tree().get_all_leaves_set().len();
            let nkeys = datatree.fmm.tree().get_all_keys_set().len();

            // Test that the number of of coefficients is being correctly assigned
            assert_eq!(datatree.multipoles.len(), ncoeffs * nkeys);
            assert_eq!(datatree.leaf_multipoles.len(), nleaves);

            // Test that leaf indices are being mapped correctly to leaf multipoles
            let idx = 0;
            let leaf_key = &datatree.fmm.tree().get_all_leaves().unwrap()[idx];
            let &leaf_idx = datatree.fmm.tree().get_index(leaf_key).unwrap();
            unsafe {
                let result = datatree.multipoles.as_ptr().add(leaf_idx * ncoeffs);
                let expected =
                    datatree.multipoles[leaf_idx * ncoeffs..(leaf_idx + 1) * ncoeffs].as_ptr();
                assert_eq!(result, expected);

                let result = datatree.locals.as_ptr().add(leaf_idx * ncoeffs);
                let expected =
                    datatree.locals[leaf_idx * ncoeffs..(leaf_idx + 1) * ncoeffs].as_ptr();
                assert_eq!(result, expected);
            }

            // Test that level expansion information is referring to correct memory, and is in correct shape
            assert_eq!(
                datatree.level_multipoles.len() as u64,
                datatree.fmm.tree().get_depth() + 1
            );
            assert_eq!(
                datatree.level_locals.len() as u64,
                datatree.fmm.tree().get_depth() + 1
            );

            assert_eq!(
                datatree.level_multipoles[(datatree.fmm.tree().get_depth()) as usize].len(),
                nleaves
            );
            assert_eq!(
                datatree.level_locals[(datatree.fmm.tree().get_depth()) as usize].len(),
                nleaves
            );

            // Test that points are being assigned correctly
            assert_eq!(datatree.potentials_send_pointers.len(), nleaves);
            assert_eq!(datatree.potentials.len(), npoints);
        }
    }
}
