//! Data structures FMM data and metadata.
use std::collections::HashMap;

use bempp_traits::fmm::KiFmm;
use bempp_traits::kernel::ScaleInvariantKernel;
use bempp_traits::{field::FieldTranslationData, fmm::Fmm, kernel::Kernel, tree::Tree};
use bempp_tree::types::morton::MortonKey;
use bempp_tree::types::single_node::SingleNodeTree;
use cauchy::Scalar;
use num::{Complex, Float};
use rlst::dense::traits::*;
use rlst::dense::{base_matrix::BaseMatrix, data_container::VectorContainer, matrix::Matrix};

/// Type alias for charge data
pub type Charge<T> = T;

/// Type alias for global index for identifying charge data with a point
pub type GlobalIdx = usize;

/// Type alias for mapping charge data to global indices.
pub type ChargeDict<T> = HashMap<GlobalIdx, Charge<T>>;

/// Type alias for approximation of FMM operator matrices.
pub type C2EType<T> = Matrix<T, BaseMatrix<T, VectorContainer<T>, Dynamic>, Dynamic>;

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

pub struct FmmDataUniformMatrix<T, U>
where
    T: Fmm,
    U: Scalar<Real = U> + Float + Default,
{
    /// The associated FMM object, which implements an FMM interface
    pub fmm: T,

    /// The multipole expansion data at each box.
    pub multipoles: Vec<U>,

    /// Multipole expansions at leaf level
    pub leaf_multipoles: Vec<Vec<SendPtrMut<U>>>,

    /// Multipole expansions at each level
    pub level_multipoles: Vec<Vec<Vec<SendPtrMut<U>>>>,

    /// The local expansion at each box
    pub locals: Vec<U>,

    /// Local expansions at the leaf level
    pub leaf_locals: Vec<Vec<SendPtrMut<U>>>,

    /// The local expansion data at each level.
    pub level_locals: Vec<Vec<Vec<SendPtrMut<U>>>>,

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

    /// Number of charge vectors being processed
    pub ncharge_vectors: usize,

    /// Number of keys
    pub nkeys: usize,

    /// Number of leaves
    pub nleaves: usize,

    /// Number of coefficients in local and multipole expansions
    pub ncoeffs: usize,

    /// Scales of each leaf operator
    pub scales: Vec<U>,

    /// Global indices of each charge
    pub global_indices: Vec<usize>,
}

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

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_2: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_1: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
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

/// Type to store data associated with the kernel independent (KiFMM) in for matrix input FMMs.
pub struct KiFmmLinearMatrix<T, U, V, W>
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

    /// The pseudo-inverse of the dense interaction matrix between the upward check and upward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub uc2e_inv_2: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_1: C2EType<W>,

    /// The pseudo-inverse of the dense interaction matrix between the downward check and downward equivalent surfaces.
    /// Store in two parts to avoid propagating error from computing pseudo-inverse
    pub dc2e_inv_2: C2EType<W>,

    /// The ratio of the inner check surface diamater in comparison to the surface discretising a box.
    pub alpha_inner: W,

    /// The ratio of the outer check surface diamater in comparison to the surface discretising a box.
    pub alpha_outer: W,

    /// The multipole to multipole operator matrices, each index is associated with a child box (in sequential Morton order),
    pub m2m: Vec<C2EType<W>>,

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

/// Implementation of the data structure to store the data for the single node KiFMM.
impl<T, U, V> FmmDataUniform<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V>,
    U: FieldTranslationData<T>,
    V: Float + Scalar<Real = V> + Default,
{
    /// Constructor fo the KiFMM's associated FmmData on a single node.
    ///
    /// # Arguments
    /// `fmm` - A single node KiFMM object.
    /// `global_charges` - The charge data associated to the point data via unique global indices.
    pub fn new(
        fmm: KiFmmLinear<SingleNodeTree<V>, T, U, V>,
        global_charges: &ChargeDict<V>,
    ) -> Result<Self, String> {
        if let Some(keys) = fmm.tree().get_all_keys() {
            let ncoeffs = fmm.m2l.ncoeffs(fmm.order);
            let nkeys = keys.len();
            let leaves = fmm.tree().get_all_leaves().unwrap();
            let nleaves = leaves.len();
            let npoints = fmm.tree().get_all_points().unwrap().len();

            let multipoles = vec![V::default(); ncoeffs * nkeys];
            let locals = vec![V::default(); ncoeffs * nkeys];

            let mut potentials = vec![V::default(); npoints];
            let mut potentials_send_pointers = vec![SendPtrMut::default(); nleaves];

            let mut charges = vec![V::default(); npoints];
            let global_indices = vec![0usize; npoints];

            // Lookup leaf coordinates, and assign charges from within the data tree.
            for (i, g_idx) in fmm
                .tree()
                .get_all_global_indices()
                .unwrap()
                .iter()
                .enumerate()
            {
                let charge = global_charges.get(g_idx).unwrap();
                charges[i] = *charge;
            }

            let mut level_multipoles = vec![Vec::new(); (fmm.tree().get_depth() + 1) as usize];
            let mut level_locals = vec![Vec::new(); (fmm.tree().get_depth() + 1) as usize];
            let mut level_index_pointer =
                vec![HashMap::new(); (fmm.tree().get_depth() + 1) as usize];

            for level in 0..=fmm.tree().get_depth() {
                let keys = fmm.tree().get_keys(level).unwrap();

                let mut tmp_multipoles = Vec::new();
                let mut tmp_locals = Vec::new();
                for (level_idx, key) in keys.iter().enumerate() {
                    let idx = fmm.tree().get_index(key).unwrap();
                    unsafe {
                        let raw = multipoles.as_ptr().add(ncoeffs * idx) as *mut V;
                        tmp_multipoles.push(SendPtrMut { raw });
                        let raw = locals.as_ptr().add(ncoeffs * idx) as *mut V;
                        tmp_locals.push(SendPtrMut { raw })
                    }
                    level_index_pointer[level as usize].insert(*key, level_idx);
                }
                level_multipoles[level as usize] = tmp_multipoles;
                level_locals[level as usize] = tmp_locals;
            }

            let mut leaf_multipoles = Vec::new();
            let mut leaf_locals = Vec::new();

            for leaf in fmm.tree.get_all_leaves().unwrap().iter() {
                let i = fmm.tree.get_index(leaf).unwrap();
                unsafe {
                    let raw = multipoles.as_ptr().add(i * ncoeffs) as *mut V;
                    leaf_multipoles.push(SendPtrMut { raw });

                    let raw = locals.as_ptr().add(i * ncoeffs) as *mut V;
                    leaf_locals.push(SendPtrMut { raw });
                }
            }

            // Create an index pointer for the charge data
            let mut index_pointer = 0;
            let mut charge_index_pointer = vec![(0usize, 0usize); nleaves];

            let mut potential_raw_pointer = potentials.as_mut_ptr();

            let mut scales = vec![V::default(); nleaves * ncoeffs];
            for (i, leaf) in leaves.iter().enumerate() {
                // Assign scales
                let l = i * ncoeffs;
                let r = l + ncoeffs;
                scales[l..r]
                    .copy_from_slice(vec![fmm.kernel.scale(leaf.level()); ncoeffs].as_slice());

                // Assign potential pointers
                let npoints;
                if let Some(points) = fmm.tree().get_points(leaf) {
                    npoints = points.len();
                } else {
                    npoints = 0;
                }

                potentials_send_pointers[i] = SendPtrMut {
                    raw: potential_raw_pointer,
                };

                let bounds = (index_pointer, index_pointer + npoints);
                charge_index_pointer[i] = bounds;
                index_pointer += npoints;
                unsafe { potential_raw_pointer = potential_raw_pointer.add(npoints) };
            }

            let dim = fmm.kernel().space_dimension();
            let mut upward_surfaces = vec![V::default(); ncoeffs * nkeys * dim];
            let mut downward_surfaces = vec![V::default(); ncoeffs * nkeys * dim];

            // For each key form both upward and downward check surfaces
            for (i, key) in keys.iter().enumerate() {
                let upward_surface =
                    key.compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_outer());

                let downward_surface =
                    key.compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_inner());

                let l = i * ncoeffs * dim;
                let r = l + ncoeffs * dim;

                upward_surfaces[l..r].copy_from_slice(&upward_surface);
                downward_surfaces[l..r].copy_from_slice(&downward_surface);
            }

            let mut leaf_upward_surfaces = vec![V::default(); ncoeffs * nleaves * dim];
            let mut leaf_downward_surfaces = vec![V::default(); ncoeffs * nleaves * dim];
            for (i, leaf) in leaves.iter().enumerate() {
                let upward_surface =
                    leaf.compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_outer());

                let downward_surface =
                    leaf.compute_surface(fmm.tree().get_domain(), fmm.order(), fmm.alpha_outer());

                let l = i * ncoeffs * dim;
                let r = l + ncoeffs * dim;
                leaf_upward_surfaces[l..r].copy_from_slice(&upward_surface);
                leaf_downward_surfaces[l..r].copy_from_slice(&downward_surface);
            }

            return Ok(Self {
                fmm,
                multipoles,
                level_multipoles,
                leaf_multipoles,
                locals,
                level_locals,
                leaf_locals,
                level_index_pointer,
                upward_surfaces,
                downward_surfaces,
                leaf_upward_surfaces,
                leaf_downward_surfaces,
                potentials,
                potentials_send_pointers,
                charges,
                charge_index_pointer,
                scales,
                global_indices,
            });
        }

        Err("Not a valid tree".to_string())
    }
}

/// Implementation of the data structure to store the data for the single node KiFMM.
impl<T, U, V> FmmDataUniformMatrix<KiFmmLinearMatrix<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V>,
    U: FieldTranslationData<T>,
    V: Float + Scalar<Real = V> + Default,
{
    /// Constructor fo the KiFMM's associated FmmData on a single node.
    ///
    /// # Arguments
    /// `fmm` - A single node KiFMM object.
    /// `global_charges` - The charge data associated to the point data via unique global indices.
    pub fn new(
        fmm: KiFmmLinearMatrix<SingleNodeTree<V>, T, U, V>,
        global_charges: &[ChargeDict<V>],
    ) -> Result<Self, String> {
        if let Some(keys) = fmm.tree().get_all_keys() {
            if !fmm.tree().adaptive {
                let ncoeffs = fmm.m2l.ncoeffs(fmm.order);
                let nkeys = keys.len();
                let leaves = fmm.tree().get_all_leaves().unwrap();
                let nleaves = leaves.len();
                let npoints = fmm.tree().get_all_points().unwrap().len();
                let ncharge_vectors = global_charges.len();

                let multipoles = vec![V::default(); ncoeffs * nkeys * ncharge_vectors];
                let locals = vec![V::default(); ncoeffs * nkeys * ncharge_vectors];

                // Indexed by charge vec idx, then leaf
                let mut potentials = vec![V::default(); npoints * ncharge_vectors];

                // Indexed by charge vec idx, then leaf.
                let mut potentials_send_pointers =
                    vec![SendPtrMut::default(); nleaves * ncharge_vectors];
                let mut scales = vec![V::default(); nleaves * ncoeffs * ncharge_vectors];

                let mut charges = vec![V::default(); npoints * ncharge_vectors];
                let global_indices = vec![0usize; npoints];

                // Lookup leaf coordinates, and assign charges from within the data tree.
                for (i, charge_dict) in global_charges.iter().enumerate() {
                    for (j, g_idx) in fmm
                        .tree()
                        .get_all_global_indices()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        let charge = charge_dict.get(g_idx).unwrap();
                        charges[i * npoints + j] = *charge;
                    }
                }

                // Indexed by level, then by box, then by charge vec.
                let mut level_multipoles = vec![Vec::new(); (fmm.tree().get_depth() + 1) as usize];
                let mut level_locals = vec![Vec::new(); (fmm.tree().get_depth() + 1) as usize];

                // Indexed by level, gives key level index
                let mut level_index_pointer =
                    vec![HashMap::new(); (fmm.tree().get_depth() + 1) as usize];

                for level in 0..=fmm.tree().get_depth() {
                    let keys = fmm.tree().get_keys(level).unwrap();

                    let mut tmp_multipoles = Vec::new();
                    let mut tmp_locals = Vec::new();
                    for key in keys.iter() {
                        let &key_idx = fmm.tree().get_index(key).unwrap();
                        let key_displacement = key_idx * ncoeffs * ncharge_vectors;
                        let mut key_multipoles = Vec::new();
                        let mut key_locals = Vec::new();
                        for charge_vec_idx in 0..ncharge_vectors {
                            let charge_vec_displacement = charge_vec_idx * ncoeffs;
                            let raw = unsafe {
                                multipoles
                                    .as_ptr()
                                    .add(key_displacement + charge_vec_displacement)
                                    as *mut V
                            };
                            key_multipoles.push(SendPtrMut { raw });

                            let raw = unsafe {
                                locals
                                    .as_ptr()
                                    .add(key_displacement + charge_vec_displacement)
                                    as *mut V
                            };
                            key_locals.push(SendPtrMut { raw });
                        }
                        tmp_multipoles.push(key_multipoles);
                        tmp_locals.push(key_locals);
                    }
                    level_multipoles[level as usize] = tmp_multipoles;
                    level_locals[level as usize] = tmp_locals;
                }

                for level in 0..=fmm.tree().get_depth() {
                    let keys = fmm.tree().get_keys(level).unwrap();
                    for (level_idx, key) in keys.iter().enumerate() {
                        level_index_pointer[level as usize].insert(*key, level_idx);
                    }
                }

                // Indexed by leaf
                let mut leaf_multipoles = vec![Vec::new(); nleaves];
                let mut leaf_locals = vec![Vec::new(); nleaves];

                for (leaf_idx, leaf) in fmm.tree().get_all_leaves().unwrap().iter().enumerate() {
                    let key_idx = fmm.tree().get_index(leaf).unwrap();
                    let key_displacement = ncharge_vectors * ncoeffs * key_idx;
                    for charge_vec_idx in 0..ncharge_vectors {
                        let charge_vec_displacement = charge_vec_idx * ncoeffs;
                        let raw = unsafe {
                            multipoles
                                .as_ptr()
                                .add(charge_vec_displacement + key_displacement)
                                as *mut V
                        };

                        leaf_multipoles[leaf_idx].push(SendPtrMut { raw });

                        let raw = unsafe {
                            locals
                                .as_ptr()
                                .add(charge_vec_displacement + key_displacement)
                                as *mut V
                        };
                        leaf_locals[leaf_idx].push(SendPtrMut { raw });
                    }
                }

                // Create an index pointer for the point data
                let mut index_pointer = 0;
                let mut charge_index_pointer = vec![(0usize, 0usize); nleaves];

                // Get raw pointers to the head of each potential vector
                let mut potential_raw_pointers = Vec::with_capacity(ncharge_vectors);
                for (charge_vec_idx, _) in (0..ncharge_vectors).enumerate() {
                    let ptr = unsafe { potentials.as_mut_ptr().add(charge_vec_idx * npoints) };
                    potential_raw_pointers.push(ptr);
                }

                for (i, leaf) in leaves.iter().enumerate() {
                    // Assign scales
                    let l = i * ncoeffs;
                    let r = l + ncoeffs;
                    scales[l..r]
                        .copy_from_slice(vec![fmm.kernel.scale(leaf.level()); ncoeffs].as_slice());

                    // Assign potential pointers
                    let npoints;
                    if let Some(points) = fmm.tree().get_points(leaf) {
                        npoints = points.len();
                    } else {
                        npoints = 0;
                    }

                    for j in 0..ncharge_vectors {
                        potentials_send_pointers[nleaves * j + i] = SendPtrMut {
                            raw: potential_raw_pointers[j],
                        };
                    }

                    // Update charge index pointer
                    let bounds = (index_pointer, index_pointer + npoints);
                    charge_index_pointer[i] = bounds;
                    index_pointer += npoints;

                    // Update raw pointers with the number of points at this leaf
                    for ptr in potential_raw_pointers.iter_mut() {
                        *ptr = unsafe { ptr.add(npoints) };
                    }
                }

                let dim = fmm.kernel().space_dimension();
                let mut upward_surfaces = vec![V::default(); ncoeffs * nkeys * dim];
                let mut downward_surfaces = vec![V::default(); ncoeffs * nkeys * dim];

                // For each key form both upward and downward check surfaces
                for (i, key) in keys.iter().enumerate() {
                    let upward_surface = key.compute_surface(
                        fmm.tree().get_domain(),
                        fmm.order(),
                        fmm.alpha_outer(),
                    );

                    let downward_surface = key.compute_surface(
                        fmm.tree().get_domain(),
                        fmm.order(),
                        fmm.alpha_inner(),
                    );

                    let l = i * ncoeffs * dim;
                    let r = l + ncoeffs * dim;

                    upward_surfaces[l..r].copy_from_slice(&upward_surface);
                    downward_surfaces[l..r].copy_from_slice(&downward_surface);
                }

                let mut leaf_upward_surfaces = vec![V::default(); ncoeffs * nleaves * dim];
                let mut leaf_downward_surfaces = vec![V::default(); ncoeffs * nleaves * dim];
                for (i, leaf) in leaves.iter().enumerate() {
                    let upward_surface = leaf.compute_surface(
                        fmm.tree().get_domain(),
                        fmm.order(),
                        fmm.alpha_outer(),
                    );

                    let downward_surface = leaf.compute_surface(
                        fmm.tree().get_domain(),
                        fmm.order(),
                        fmm.alpha_outer(),
                    );

                    let l = i * ncoeffs * dim;
                    let r = l + ncoeffs * dim;
                    leaf_upward_surfaces[l..r].copy_from_slice(&upward_surface);
                    leaf_downward_surfaces[l..r].copy_from_slice(&downward_surface);
                }

                return Ok(Self {
                    fmm,
                    multipoles,
                    level_multipoles,
                    leaf_multipoles,
                    locals,
                    level_locals,
                    leaf_locals,
                    level_index_pointer,
                    upward_surfaces,
                    downward_surfaces,
                    leaf_upward_surfaces,
                    leaf_downward_surfaces,
                    potentials,
                    potentials_send_pointers,
                    charges,
                    charge_index_pointer,
                    ncharge_vectors,
                    nkeys,
                    nleaves,
                    ncoeffs,
                    scales,
                    global_indices,
                });
            } else {
                return Err("Not a uniform tree".to_string());
            }
        }
        Err("Not a valid tree, no keys present".to_string())
    }
}

// Implementation of the data structure to store the data for the single node KiFMM.
impl<T, U, V> FmmDataAdaptive<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V>,
    U: FieldTranslationData<T>,
    V: Float + Scalar<Real = V> + Default,
{
    /// Constructor fo the KiFMM's associated FmmData on a single node.
    ///
    /// # Arguments
    /// `fmm` - A single node KiFMM object.
    /// `global_charges` - The charge data associated to the point data via unique global indices.
    pub fn new(
        fmm: KiFmmLinear<SingleNodeTree<V>, T, U, V>,
        global_charges: &ChargeDict<V>,
    ) -> Result<Self, String> {
        if let Some(keys) = fmm.tree().get_all_keys() {
            if fmm.tree().adaptive {
                let ncoeffs = fmm.m2l.ncoeffs(fmm.order);
                let nkeys = keys.len();
                let leaves = fmm.tree().get_all_leaves().unwrap();
                let nleaves = leaves.len();
                let npoints = fmm.tree().get_all_points().unwrap().len();

                let multipoles = vec![V::default(); ncoeffs * nkeys];
                let locals = vec![V::default(); ncoeffs * nkeys];

                let mut potentials = vec![V::default(); npoints];
                let mut potentials_send_pointers = vec![SendPtrMut::default(); nleaves];

                let mut charges = vec![V::default(); npoints];
                let global_indices = vec![0usize; npoints];

                // Lookup leaf coordinates, and assign charges from within the data tree.
                for (i, g_idx) in fmm
                    .tree()
                    .get_all_global_indices()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    let charge = global_charges.get(g_idx).unwrap();
                    charges[i] = *charge;
                }

                let mut level_multipoles = vec![Vec::new(); (fmm.tree().get_depth() + 1) as usize];
                let mut level_locals = vec![Vec::new(); (fmm.tree().get_depth() + 1) as usize];
                let mut level_index_pointer =
                    vec![HashMap::new(); (fmm.tree().get_depth() + 1) as usize];

                for level in 0..=fmm.tree().get_depth() {
                    let keys = fmm.tree().get_keys(level).unwrap();

                    let mut tmp_multipoles = Vec::new();
                    let mut tmp_locals = Vec::new();
                    for (level_idx, key) in keys.iter().enumerate() {
                        let &idx = fmm.tree().get_index(key).unwrap();
                        unsafe {
                            let raw = multipoles.as_ptr().add(idx * ncoeffs) as *mut V;
                            tmp_multipoles.push(SendPtrMut { raw });

                            let raw = locals.as_ptr().add(idx * ncoeffs) as *mut V;
                            tmp_locals.push(SendPtrMut { raw })
                        }
                        level_index_pointer[level as usize].insert(*key, level_idx);
                    }
                    level_multipoles[level as usize] = tmp_multipoles;
                    level_locals[level as usize] = tmp_locals;
                }

                let mut leaf_multipoles = Vec::new();
                let mut leaf_locals = Vec::new();

                for leaf in fmm.tree.get_all_leaves().unwrap().iter() {
                    let i = fmm.tree.get_index(leaf).unwrap();
                    unsafe {
                        let raw = multipoles.as_ptr().add(i * ncoeffs) as *mut V;
                        leaf_multipoles.push(SendPtrMut { raw });

                        let raw = locals.as_ptr().add(i * ncoeffs) as *mut V;
                        leaf_locals.push(SendPtrMut { raw });
                    }
                }

                // Create an index pointer for the charge data
                let mut index_pointer = 0;
                let mut charge_index_pointer = vec![(0usize, 0usize); nleaves];

                let mut potential_raw_pointer = potentials.as_mut_ptr();

                let mut scales = vec![V::default(); nleaves * ncoeffs];
                for (i, leaf) in leaves.iter().enumerate() {
                    // Assign scales
                    let l = i * ncoeffs;
                    let r = l + ncoeffs;
                    scales[l..r]
                        .copy_from_slice(vec![fmm.kernel.scale(leaf.level()); ncoeffs].as_slice());

                    // Assign potential pointers
                    let npoints;
                    if let Some(points) = fmm.tree().get_points(leaf) {
                        npoints = points.len();
                    } else {
                        npoints = 0;
                    }

                    potentials_send_pointers[i] = SendPtrMut {
                        raw: potential_raw_pointer,
                    };

                    let bounds = (index_pointer, index_pointer + npoints);
                    charge_index_pointer[i] = bounds;
                    index_pointer += npoints;
                    unsafe { potential_raw_pointer = potential_raw_pointer.add(npoints) };
                }

                let dim = fmm.kernel().space_dimension();
                let mut upward_surfaces = vec![V::default(); ncoeffs * nkeys * dim];
                let mut downward_surfaces = vec![V::default(); ncoeffs * nkeys * dim];

                // For each key form both upward and downward check surfaces
                for (i, key) in keys.iter().enumerate() {
                    let upward_surface = key.compute_surface(
                        fmm.tree().get_domain(),
                        fmm.order(),
                        fmm.alpha_outer(),
                    );

                    let downward_surface = key.compute_surface(
                        fmm.tree().get_domain(),
                        fmm.order(),
                        fmm.alpha_inner(),
                    );

                    let l = i * ncoeffs * dim;
                    let r = l + ncoeffs * dim;

                    upward_surfaces[l..r].copy_from_slice(&upward_surface);
                    downward_surfaces[l..r].copy_from_slice(&downward_surface);
                }

                let mut leaf_upward_surfaces = vec![V::default(); ncoeffs * nleaves * dim];
                let mut leaf_downward_surfaces = vec![V::default(); ncoeffs * nleaves * dim];
                for (i, leaf) in leaves.iter().enumerate() {
                    let upward_surface = leaf.compute_surface(
                        fmm.tree().get_domain(),
                        fmm.order(),
                        fmm.alpha_outer(),
                    );

                    let downward_surface = leaf.compute_surface(
                        fmm.tree().get_domain(),
                        fmm.order(),
                        fmm.alpha_outer(),
                    );

                    let l = i * ncoeffs * dim;
                    let r = l + ncoeffs * dim;
                    leaf_upward_surfaces[l..r].copy_from_slice(&upward_surface);
                    leaf_downward_surfaces[l..r].copy_from_slice(&downward_surface);
                }

                return Ok(Self {
                    fmm,
                    multipoles,
                    level_multipoles,
                    leaf_multipoles,
                    locals,
                    level_locals,
                    leaf_locals,
                    level_index_pointer,
                    upward_surfaces,
                    downward_surfaces,
                    leaf_upward_surfaces,
                    leaf_downward_surfaces,
                    potentials,
                    potentials_send_pointers,
                    charges,
                    charge_index_pointer,
                    scales,
                    global_indices,
                });
            } else {
                return Err("Not an adaptive tree".to_string());
            }
        }
        Err("Not a valid tree, no nodes present.".to_string())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        charge::build_charge_dict,
        types::{FmmDataUniform, FmmDataUniformMatrix, KiFmmLinear, KiFmmLinearMatrix},
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
    use rlst::dense::RawAccess;

    #[test]
    fn test_fmm_data_uniform_matrix() {
        let npoints = 10000;
        let ncharge_vecs = 10;
        let points = points_fixture::<f64>(npoints, None, None);
        let global_idxs = (0..npoints).collect_vec();
        let mut charge_mat = vec![vec![0.0; npoints]; ncharge_vecs];
        charge_mat
            .iter_mut()
            .enumerate()
            .for_each(|(i, charge_mat_i)| *charge_mat_i = vec![i as f64 + 1.0; npoints]);

        let order = 8;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let ncrit = 150;
        let depth = 3;

        // Uniform trees, with matrix of charges
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

            let m2l_data = FftFieldTranslationKiFmm::new(
                kernel.clone(),
                order,
                *tree.get_domain(),
                alpha_inner,
            );

            let fmm =
                KiFmmLinearMatrix::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

            // Form charge dicts, matching all charges with their associated global indices
            let charge_dicts: Vec<_> = (0..ncharge_vecs)
                .map(|i| build_charge_dict(&global_idxs, &charge_mat[i]))
                .collect();

            let datatree = FmmDataUniformMatrix::new(fmm, &charge_dicts).unwrap();

            // Test that the number of coefficients is being correctly assigned
            assert_eq!(
                datatree.multipoles.len(),
                datatree.ncoeffs * datatree.nkeys * ncharge_vecs
            );
            assert_eq!(datatree.leaf_multipoles.len(), datatree.nleaves);
            for i in 0..datatree.nleaves {
                assert_eq!(datatree.leaf_multipoles[i].len(), ncharge_vecs)
            }
            assert_eq!(
                datatree.locals.len(),
                datatree.ncoeffs * datatree.nkeys * ncharge_vecs
            );
            assert_eq!(datatree.leaf_locals.len(), datatree.nleaves);
            for i in 0..datatree.nleaves {
                assert_eq!(datatree.leaf_locals[i].len(), ncharge_vecs)
            }

            // Test that the leaf indices are being mapped correctly to leaf multipoles and locals
            for (i, leaf_key) in datatree
                .fmm
                .tree()
                .get_all_leaves()
                .unwrap()
                .iter()
                .enumerate()
            {
                let key_idx = datatree.fmm.tree().get_index(leaf_key).unwrap();
                let key_displacement = key_idx * datatree.ncoeffs * ncharge_vecs;

                for charge_vec_idx in 0..ncharge_vecs {
                    let charge_vec_displacement = charge_vec_idx * datatree.ncoeffs;

                    unsafe {
                        let result = datatree.leaf_multipoles[i][charge_vec_idx].raw as *const f64;
                        let expected = datatree
                            .multipoles
                            .as_ptr()
                            .add(charge_vec_displacement + key_displacement);

                        assert_eq!(result, expected);

                        let result = datatree.leaf_locals[i][charge_vec_idx].raw as *const f64;
                        let expected = datatree
                            .locals
                            .as_ptr()
                            .add(charge_vec_displacement + key_displacement);

                        assert_eq!(result, expected);
                    }
                }
            }

            // Test that the level expansion information is referring to the correct memory, and is in correct shape
            assert_eq!(
                datatree.level_multipoles.len() as u64,
                datatree.fmm.tree().get_depth() + 1
            );

            for level in 0..datatree.fmm.tree().get_depth() {
                assert_eq!(
                    datatree.level_multipoles[level as usize].len(),
                    datatree.fmm.tree().get_keys(level).unwrap().len()
                );
                assert_eq!(
                    datatree.level_multipoles[level as usize][0].len(),
                    ncharge_vecs
                );
            }

            assert_eq!(
                datatree.level_locals.len() as u64,
                datatree.fmm.tree().get_depth() + 1
            );

            for level in 0..datatree.fmm.tree().get_depth() {
                assert_eq!(
                    datatree.level_locals[level as usize].len(),
                    datatree.fmm.tree().get_keys(level).unwrap().len()
                );
                assert_eq!(datatree.level_locals[level as usize][0].len(), ncharge_vecs);
            }
        }
    }

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
            assert_eq!(datatree.locals.len(), ncoeffs * nkeys);
            assert_eq!(datatree.leaf_locals.len(), nleaves);

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
