// Laplace kernel
use std::collections::HashSet;
use std::vec;

use cauchy::Scalar;
use itertools::Itertools;
use ndarray::*;
use ndarray_linalg::SVDDC;

use solvers_traits::fmm::KiFmmNode;
use solvers_traits::types::{Error, EvalType, Result};
use solvers_traits::{
    fmm::{Fmm, FmmTree, Translation},
    kernel::Kernel,
};
use solvers_tree::constants::ROOT;
use solvers_tree::types::data::NodeData;

use solvers_tree::types::{
    domain::Domain,
    morton::{MortonKey, MortonKeys},
    point::{Point, Points},
};

use crate::linalg::pinv;

// TODO: Create from FMM Factory pattern, specialised for Rust in some way
pub struct KiFmm {
    pub kernel: Box<dyn Kernel<PotentialData = Vec<f64>, GradientData = Vec<[f64; 3]>>>,
    pub tree: Box<
        dyn FmmTree<
            NodeIndex = MortonKey,
            NodeIndices = MortonKeys,
            FmmNodeDataType = NodeData,
            NodeDataType = NodeData,
            Domain = Domain,
            Point = Point,
            Points = Points,
            NodeIndicesSet = HashSet<MortonKey>,
            NodeDataContainer = Vec<f64>,
            PointDataType = [f64; 2],
        >,
    >,
    pub order: usize,
    pub alpha_inner: f64,
    pub alpha_outer: f64,
    pub uc2e_inv: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),
    pub dc2e_inv: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),
    pub m2m: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>>,
    pub l2l: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>>,
    pub transfer_vectors: Vec<usize>,
    pub m2l: (
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ),
}

impl KiFmm {
    /// Algebraically defined list of transfer vectors in an octree
    fn find_unique_v_list_interactions() -> (Vec<MortonKey>, Vec<MortonKey>, Vec<usize>) {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };

        let level = 3;

        // Encode point in centre of domain
        let key = MortonKey::from_point(&point, &domain, level);

        // Add neighbours, and their resp. siblings to v list.
        let mut neighbours = key.neighbors();
        let mut keys: Vec<MortonKey> = Vec::new();
        keys.push(key);
        keys.append(&mut neighbours);

        for key in neighbours.iter() {
            let mut siblings = key.siblings();
            keys.append(&mut siblings);
        }

        // Keep only unique keys
        let keys: Vec<&MortonKey> = keys.iter().unique().collect();

        let mut transfer_vectors: Vec<usize> = Vec::new();
        let mut targets: Vec<MortonKey> = Vec::new();
        let mut sources: Vec<MortonKey> = Vec::new();

        for key in keys.iter() {
            // Dense v_list
            let v_list = key
                .parent()
                .neighbors()
                .iter()
                .flat_map(|pn| pn.children())
                .filter(|pnc| !key.is_adjacent(pnc))
                .collect_vec();

            // Find transfer vectors for everything in dense v list of each key
            let tmp: Vec<usize> = v_list
                .iter()
                .map(|v| key.find_transfer_vector(v))
                .collect_vec();

            transfer_vectors.extend(&mut tmp.iter().cloned());
            sources.extend(&mut v_list.iter().cloned());

            let tmp_targets = vec![**key; tmp.len()];
            targets.extend(&mut tmp_targets.iter().cloned());
        }

        let mut unique_transfer_vectors = Vec::new();
        let mut unique_indices = HashSet::new();

        for (i, vec) in transfer_vectors.iter().enumerate() {
            if !unique_transfer_vectors.contains(vec) {
                unique_transfer_vectors.push(*vec);
                unique_indices.insert(i);
            }
        }

        let unique_sources: Vec<MortonKey> = sources
            .iter()
            .enumerate()
            .filter(|(i, _)| unique_indices.contains(i))
            .map(|(_, x)| *x)
            .collect_vec();

        let unique_targets: Vec<MortonKey> = targets
            .iter()
            .enumerate()
            .filter(|(i, _)| unique_indices.contains(i))
            .map(|(_, x)| *x)
            .collect_vec();

        (unique_targets, unique_sources, unique_transfer_vectors)
    }

    fn ncoeffs(order: usize) -> usize {
        6 * (order - 1).pow(2) + 2
    }

    fn m2l_scale(level: u64) -> f64 {
        if level < 2 {
            panic!("M2L only performed on level 2 and below")
        }

        let m2l_scale;
        if level == 2 {
            m2l_scale = 1. / 2.
        } else {
            m2l_scale = 2.0.powf((level - 3) as f64);
        }
        m2l_scale
    }

    fn new(
        order: usize,
        alpha_inner: f64,
        alpha_outer: f64,
        tree: Box<
            dyn FmmTree<
                Domain = Domain,
                Point = Point,
                Points = Points,
                NodeIndex = MortonKey,
                NodeIndices = MortonKeys,
                NodeIndicesSet = HashSet<MortonKey>,
                NodeDataType = NodeData,
                FmmNodeDataType = NodeData,
                NodeDataContainer = Vec<f64>,
                PointDataType = [f64; 2],
            >,
        >,
        kernel: Box<dyn Kernel<PotentialData = Vec<f64>, GradientData = Vec<[f64; 3]>>>,
    ) -> KiFmm {
        // Compute equivalent and check surfaces at root level
        let upward_equivalent_surface = ROOT.compute_surface(order, alpha_inner, tree.get_domain());

        let upward_check_surface = ROOT.compute_surface(order, alpha_outer, tree.get_domain());

        let downward_equivalent_surface =
            ROOT.compute_surface(order, alpha_outer, tree.get_domain());

        let downward_check_surface = ROOT.compute_surface(order, alpha_inner, tree.get_domain());

        // Compute upward check to equivalent, and downward check to equivalent Gram matrices
        // as well as their inverses using DGESVD.
        let uc2e = kernel
            .gram(&upward_equivalent_surface, &upward_check_surface)
            .unwrap();
        let dc2e = kernel
            .gram(&downward_equivalent_surface, &downward_check_surface)
            .unwrap();

        let nrows = KiFmm::ncoeffs(order);
        let ncols = nrows;

        let uc2e = Array1::from(uc2e)
            .to_shape((nrows, ncols))
            .unwrap()
            .to_owned();

        // Store in two component format for stability, See Malhotra et. al (2015)
        let dc2e = Array1::from(dc2e)
            .to_shape((ncols, nrows))
            .unwrap()
            .to_owned();
        let (a, b, c) = pinv(&uc2e);

        let uc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        let (a, b, c) = pinv(&dc2e);
        let dc2e_inv = (a.to_owned(), b.dot(&c).to_owned());

        // Compute M2M and L2L oeprators
        let children = ROOT.children();
        let mut m2m: Vec<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = Vec::new();
        let mut l2l: Vec<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = Vec::new();

        for child in children.iter() {
            let child_upward_equivalent_surface =
                child.compute_surface(order, alpha_inner, tree.get_domain());
            let child_downward_check_surface =
                child.compute_surface(order, alpha_inner, tree.get_domain());

            let pc2ce = kernel
                .gram(&child_upward_equivalent_surface, &upward_check_surface)
                .unwrap();

            let pc2e = Array::from_shape_vec((nrows, ncols), pc2ce).unwrap();

            m2m.push(uc2e_inv.0.dot(&uc2e_inv.1.dot(&pc2e)));

            let cc2pe = kernel
                .gram(&downward_equivalent_surface, &child_downward_check_surface)
                .unwrap();
            let cc2pe = Array::from_shape_vec((ncols, nrows), cc2pe).unwrap();

            l2l.push(kernel.scale(child.level()) * dc2e_inv.0.dot(&dc2e_inv.1.dot(&cc2pe)))
        }

        // Compute unique M2L interactions at Level 3 (smallest choice with all vectors)
        let (targets, sources, transfer_vectors) = KiFmm::find_unique_v_list_interactions();

        // Compute interaction matrices between source and unique targets, defined by unique transfer vectors
        let mut se2tc: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
            Array2::zeros((nrows, ncols * sources.len()));

        for (((i, _), source), target) in transfer_vectors
            .iter()
            .enumerate()
            .zip(sources.iter())
            .zip(targets.iter())
        {
            let source_equivalent_surface =
                source.compute_surface(order, alpha_inner, tree.get_domain());

            let target_check_surface =
                target.compute_surface(order, alpha_inner, tree.get_domain());

            let tmp_gram = kernel
                .gram(&source_equivalent_surface, &target_check_surface)
                .unwrap();

            let tmp_gram = Array::from_shape_vec((nrows, ncols), tmp_gram).unwrap();
            let lidx_sources = i * ncols;
            let ridx_sources = lidx_sources + ncols;

            se2tc
                .slice_mut(s![.., lidx_sources..ridx_sources])
                .assign(&tmp_gram);
        }

        // TODO: replace with randomised SVD
        let (u, s, vt) = se2tc.svddc(ndarray_linalg::JobSvd::Some).unwrap();
        let u = u.unwrap();
        let s = Array2::from_diag(&s);
        let vt = vt.unwrap();
        let m2l = (u, s, vt);

        KiFmm {
            kernel,
            tree,
            order,
            alpha_inner,
            alpha_outer,
            uc2e_inv,
            dc2e_inv,
            m2m,
            l2l,
            m2l,
            transfer_vectors,
        }
    }
}

pub struct LaplaceKernel {
    pub dim: usize,
    pub is_singular: bool,
    pub value_dimension: usize,
}

impl LaplaceKernel {
    fn potential(&self, sources: &[[f64; 3]], charges: &[f64], targets: &[[f64; 3]]) -> Vec<f64> {
        let mut potentials: Vec<f64> = vec![0.; targets.len()];

        // TODO: Implement multithreaded
        for (i, target) in targets.iter().enumerate() {
            let mut potential = 0.0;

            for (source, charge) in sources.iter().zip(charges) {
                let mut tmp = source
                    .iter()
                    .zip(target.iter())
                    .map(|(s, t)| (s - t).powf(2.0))
                    .sum::<f64>()
                    .powf(0.5)
                    * std::f64::consts::PI
                    * 4.0;

                tmp = tmp.recip();

                if tmp.is_infinite() {
                    continue;
                } else {
                    potential += tmp * charge;
                }
            }
            potentials[i] = potential
        }
        potentials
    }

    fn gradient_kernel(&self, source: &[f64; 3], target: &[f64; 3], c: usize) -> f64 {
        let num = source[c] - target[c];
        let invdiff: f64 = source
            .iter()
            .zip(target.iter())
            .map(|(s, t)| (s - t).powf(2.0))
            .sum::<f64>()
            .sqrt()
            * std::f64::consts::PI
            * 4.0;

        let mut tmp = invdiff.recip();
        tmp *= invdiff;
        tmp *= invdiff;
        tmp *= num;

        if tmp.is_finite() {
            tmp
        } else {
            0.
        }
    }

    fn gradient(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
    ) -> Vec<[f64; 3]> {
        let mut gradients: Vec<[f64; 3]> = vec![[0., 0., 0.]; targets.len()];

        // TODO: Implement multithreaded
        for (i, target) in targets.iter().enumerate() {
            for (source, charge) in sources.iter().zip(charges) {
                gradients[i][0] -= charge * self.gradient_kernel(source, target, 0);
                gradients[i][1] -= charge * self.gradient_kernel(source, target, 1);
                gradients[i][2] -= charge * self.gradient_kernel(source, target, 2);
            }
        }

        gradients
    }
}

impl Kernel for LaplaceKernel {
    type PotentialData = Vec<f64>;
    type GradientData = Vec<[f64; 3]>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn is_singular(&self) -> bool {
        self.is_singular
    }

    fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    fn potential(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
    ) -> solvers_traits::types::Result<Self::PotentialData> {
        solvers_traits::types::Result::Ok(self.potential(sources, charges, targets))
    }

    fn gradient(
        &self,
        sources: &[[f64; 3]],
        charges: &[f64],
        targets: &[[f64; 3]],
    ) -> Result<Self::GradientData> {
        solvers_traits::types::Result::Ok(self.gradient(sources, charges, targets))
    }

    fn gram(
        &self,
        sources: &[[f64; 3]],
        targets: &[[f64; 3]],
    ) -> solvers_traits::types::Result<Self::PotentialData> {
        let mut result: Vec<f64> = Vec::new();

        for target in targets.iter() {
            let mut row: Vec<f64> = Vec::new();
            for source in sources.iter() {
                let mut tmp = source
                    .iter()
                    .zip(target.iter())
                    .map(|(s, t)| (s - t).powf(2.0))
                    .sum::<f64>()
                    .powf(0.5)
                    * std::f64::consts::PI
                    * 4.0;

                tmp = tmp.recip();

                if tmp.is_infinite() {
                    row.push(0.0);
                } else {
                    row.push(tmp);
                }
            }
            result.append(&mut row);
        }

        Result::Ok(result)
    }

    fn scale(&self, level: u64) -> f64 {
        1. / (2f64.powf(level as f64))
    }
}

impl Translation for KiFmm {
    type NodeIndex = MortonKey;

    fn p2m(&mut self, leaf: &Self::NodeIndex) {
        let upward_check_surface =
            leaf.compute_surface(self.order, self.alpha_outer, self.tree.get_domain());

        let sources = self.tree.get_points(leaf).unwrap();
        let charges: Vec<f64> = sources.iter().map(|s| s.data[0]).collect();
        let sources: Vec<[f64; 3]> = sources.iter().map(|s| s.coordinate).collect();

        // Check potential
        let check_potential = self
            .kernel
            .potential(&sources[..], &charges[..], &upward_check_surface[..])
            .unwrap();

        let check_potential = Array1::from_vec(check_potential);

        let multipole_expansion = (self.kernel.scale(leaf.level())
            * self.uc2e_inv.0.dot(&self.uc2e_inv.1.dot(&check_potential)))
        .to_vec();

        self.tree
            .set_multipole_expansion(leaf, &multipole_expansion, self.order);
    }

    fn m2m(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {
        let in_multipole = Array::from(self.tree.get_multipole_expansion(in_node).unwrap());

        let operator_index = in_node
            .siblings()
            .iter()
            .position(|&x| x == *in_node)
            .unwrap();

        let out_multipole = self.m2m[operator_index].dot(&in_multipole).to_vec();

        if let Some(curr) = self.tree.get_multipole_expansion(&out_node) {
            let curr: Vec<f64> = curr
                .iter()
                .zip(out_multipole.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            self.tree
                .set_multipole_expansion(&out_node, &curr, self.order)
        } else {
            self.tree
                .set_multipole_expansion(&out_node, &out_multipole, self.order)
        }
    }

    fn l2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {
        let in_local = Array::from(self.tree.get_local_expansion(in_node).unwrap());

        let operator_index = out_node
            .siblings()
            .iter()
            .position(|&x| x == *out_node)
            .unwrap();

        let out_local = self.l2l[operator_index].dot(&in_local).to_vec();

        if let Some(curr) = self.tree.get_local_expansion(&out_node) {
            let curr: Vec<f64> = curr
                .iter()
                .zip(out_local.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            self.tree.set_local_expansion(&out_node, &curr, self.order);
        } else {
            self.tree
                .set_local_expansion(&out_node, &out_local, self.order);
        }
    }

    fn m2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {
        let ncoeffs = KiFmm::ncoeffs(self.order);
        // Locate correct components of compressed M2L matrix.
        let transfer_vector = out_node.find_transfer_vector(in_node);
        let v_idx = self
            .transfer_vectors
            .iter()
            .position(|&x| x == transfer_vector)
            .unwrap();

        let v_lidx = v_idx * ncoeffs;
        let v_ridx = v_lidx + ncoeffs;
        let vt_sub = self.m2l.2.slice(s![.., v_lidx..v_ridx]);

        // TODO: Remove copy.
        let in_multipole = Array::from(self.tree.get_multipole_expansion(&in_node).unwrap());

        let out_local = KiFmm::m2l_scale(in_node.level())
            * self.kernel.scale(in_node.level())
            * self.dc2e_inv.0.dot(
                &self
                    .dc2e_inv
                    .1
                    .dot(&self.m2l.0.dot(&self.m2l.1.dot(&vt_sub.dot(&in_multipole)))),
            );

        let out_local = out_local.to_vec();

        if let Some(curr) = self.tree.get_local_expansion(&out_node) {
            let curr: Vec<f64> = curr
                .iter()
                .zip(out_local.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            self.tree.set_local_expansion(out_node, &curr, self.order);
        } else {
            self.tree
                .set_local_expansion(out_node, &out_local, self.order);
        }
    }

    fn l2p(&mut self, node: &Self::NodeIndex) {
        let local_expansion = self.tree.get_local_expansion(node).unwrap();
        let downward_equivalent_surface =
            node.compute_surface(self.order, self.alpha_outer, self.tree.get_domain());
        let points = self.tree.get_points(node).unwrap();
        let point_coordinates: Vec<[f64; 3]> = points.iter().map(|p| p.coordinate).collect();
        let potential = self
            .kernel
            .potential(
                &downward_equivalent_surface,
                &local_expansion,
                &point_coordinates,
            )
            .unwrap();
        let points = points
            .iter()
            .zip(potential.iter())
            .map(|(pnt, pot)| Point {
                coordinate: pnt.coordinate,
                global_idx: pnt.global_idx,
                key: pnt.key,
                data: [pnt.data[0], *pot + pnt.data[1]],
            })
            .collect_vec();

        self.tree.set_points(node, points);
    }

    fn m2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {
        let multipole_expansion = self.tree.get_multipole_expansion(in_node).unwrap();
        let upward_equivalent_surface =
            in_node.compute_surface(self.order, self.alpha_inner, self.tree.get_domain());
        let points = self.tree.get_points(out_node).unwrap();
        let point_coordinates: Vec<[f64; 3]> = points.iter().map(|p| p.coordinate).collect();
        let potential = self
            .kernel
            .potential(
                &upward_equivalent_surface,
                &multipole_expansion,
                &point_coordinates,
            )
            .unwrap();

        let points = points
            .iter()
            .zip(potential.iter())
            .map(|(pnt, pot)| Point {
                coordinate: pnt.coordinate,
                global_idx: pnt.global_idx,
                key: pnt.key,
                data: [pnt.data[0], *pot + pnt.data[1]],
            })
            .collect_vec();
        self.tree.set_points(out_node, points);
    }

    fn p2l(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {
        if let Some(points) = self.tree.get_points(in_node) {
            // TODO: Remove copy
            let point_coordinates: Vec<[f64; 3]> = points.iter().map(|p| p.coordinate).collect();
            let charges = points.iter().map(|&p| p.data[0]).collect_vec();
            let downward_check_surface =
                out_node.compute_surface(self.order, self.alpha_inner, self.tree.get_domain());
            let downward_check_potential = Array::from(
                self.kernel
                    .potential(&point_coordinates, &charges, &downward_check_surface)
                    .unwrap(),
            );
            let out_local = (self.kernel.scale(out_node.level())
                * self
                    .dc2e_inv
                    .0
                    .dot(&self.dc2e_inv.1.dot(&downward_check_potential)))
            .to_vec();

            if let Some(curr) = self.tree.get_local_expansion(&out_node) {
                let curr: Vec<f64> = curr
                    .iter()
                    .zip(out_local.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();

                self.tree.set_local_expansion(out_node, &curr, self.order);
            } else {
                self.tree
                    .set_local_expansion(out_node, &out_local, self.order);
            }
        }
    }

    fn p2p(&mut self, in_node: &Self::NodeIndex, out_node: &Self::NodeIndex) {
        if let (Some(sources), Some(targets)) = (
            self.tree.get_points(in_node),
            self.tree.get_points(out_node),
        ) {
            // TODO: Get rid of this copy
            let source_coordinates: Vec<[f64; 3]> = sources.iter().map(|p| p.coordinate).collect();
            let charges = sources.iter().map(|&p| p.data[0]).collect_vec();
            let targets_coordinates: Vec<[f64; 3]> = targets.iter().map(|p| p.coordinate).collect();

            let potential = self
                .kernel
                .potential(&source_coordinates, &charges, &targets_coordinates)
                .unwrap();

            let targets = targets
                .iter()
                .zip(potential.iter())
                .map(|(pnt, pot)| Point {
                    coordinate: pnt.coordinate,
                    global_idx: pnt.global_idx,
                    key: pnt.key,
                    data: [pnt.data[0], *pot + pnt.data[1]],
                })
                .collect_vec();

            self.tree.set_points(out_node, targets);
        }
    }
}

impl Fmm for KiFmm {
    fn upward_pass(&mut self) {
        // P2M over leaves. TODO: multithreading over all leaves
        let nleaves = self.tree.get_leaves().len();
        for i in 0..nleaves {
            self.p2m(&self.tree.get_leaves()[i].clone());
        }

        // M2M over each key in a given level.
        for level in (1..=self.tree.get_depth()).rev() {
            let keys = self.tree.get_keys(level);

            // TODO: multithreading over keys at each level
            for key in keys.iter() {
                self.m2m(key, &key.parent())
            }
        }
    }

    fn downward_pass(&mut self) {
        // Iterate down the tree (M2L/L2L)
        for level in 2..=self.tree.get_depth() {
            let keys = self.tree.get_keys(level);

            // TODO: Multithread M2L/L2L over keys at each level.
            for target in keys.iter() {
                // V List interactions

                if let Some(v_list) = self.tree.get_interaction_list(target) {
                    for source in v_list.iter() {
                        self.m2l(source, target)
                    }
                }

                // Translate parent local expansion to its children.
                let parent = target.parent();
                self.l2l(&parent, target);
            }
        }

        // Leaf level computations
        // TODO: parallelise over leaves
        let nleaves = self.tree.get_leaves().len();

        for i in 0..nleaves {
            let target = self.tree.get_leaves()[i];

            // X List interactions
            if let Some(x_list) = self.tree.get_x_list(&target) {
                // println!("Running X List");
                for source in x_list.iter() {
                    self.p2l(source, &target);
                }
            }

            // W List interactions
            if let Some(w_list) = self.tree.get_w_list(&target) {
                // println!("Running W List");
                for source in w_list.iter() {
                    self.m2p(source, &target)
                }
            }

            // U List interactions
            if let Some(u_list) = self.tree.get_near_field(&target) {
                // println!("Running U List");
                for source in u_list.iter() {
                    self.p2p(source, &target);
                }
            }

            // Translate local expansions to points, in each node.
            self.l2p(&target);
        }
    }
    fn run(&mut self) {
        self.upward_pass();
        self.downward_pass();
    }
}

mod test {
    use std::vec;

    use itertools::Itertools;
    use ndarray::*;
    use ndarray_linalg::assert;
    use ndarray_linalg::*;
    use solvers_traits::fmm::Fmm;
    use solvers_traits::fmm::KiFmmNode;
    use solvers_traits::fmm::Translation;
    use solvers_traits::kernel::Kernel;
    use solvers_traits::types::EvalType;
    use solvers_tree::constants::ROOT;
    use solvers_tree::types::domain::Domain;
    use solvers_tree::types::morton::MortonKey;
    use solvers_tree::types::point::PointType;
    use solvers_tree::types::single_node::SingleNodeTree;

    use super::{KiFmm, LaplaceKernel};
    use rand::prelude::*;
    use rand::SeedableRng;

    use float_cmp::assert_approx_eq;

    pub fn points_fixture(npoints: usize) -> Vec<[f64; 3]> {
        let mut range = StdRng::seed_from_u64(0);
        let between = rand::distributions::Uniform::from(0.0..1.0);
        let mut points: Vec<[PointType; 3]> = Vec::new();

        for _ in 0..npoints {
            points.push([
                between.sample(&mut range),
                between.sample(&mut range),
                between.sample(&mut range),
            ])
        }

        points
    }

    #[test]
    fn test_upward_pass() {
        // Create Kernel
        let kernel = Box::new(LaplaceKernel {
            dim: 3,
            is_singular: true,
            value_dimension: 3,
        });

        // Create FmmTree
        let npoints: usize = 10000;
        let points = points_fixture(npoints);
        let point_data = vec![1.0; npoints];
        let depth = 2;
        let n_crit = 150;

        let tree = Box::new(SingleNodeTree::new(
            &points,
            &point_data,
            false,
            Some(n_crit),
            Some(depth),
        ));

        // New FMM
        let mut kifmm = KiFmm::new(6, 1.05, 1.95, tree, kernel);

        kifmm.upward_pass();

        let root_multipole = kifmm.tree.get_multipole_expansion(&ROOT).unwrap();
        let upward_equivalent_surface =
            ROOT.compute_surface(kifmm.order, kifmm.alpha_inner, kifmm.tree.get_domain());

        let distant_point = [[40.0, 0., 0.]];
        let node_points = kifmm.tree.get_all_points();
        let node_point_data: Vec<f64> = node_points.iter().map(|p| p.data[0]).collect();
        let node_points: Vec<[f64; 3]> = node_points.iter().map(|p| p.coordinate).collect();

        let direct = kifmm
            .kernel
            .potential(&node_points, &node_point_data, &distant_point)
            .unwrap();

        let result = kifmm
            .kernel
            .potential(&upward_equivalent_surface, &root_multipole, &distant_point)
            .unwrap();

        for (a, b) in result.iter().zip(direct.iter()) {
            assert_approx_eq!(f64, *a, *b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_m2l_scaling() {
        let point = [0.5, 0.5, 0.5];
        let domain = Domain {
            origin: [0., 0., 0.],
            diameter: [1., 1., 1.],
        };
        let order = 2;
        let alpha_inner = 1.05;
        let alpha_outer = 1.95;

        let kernel = LaplaceKernel {
            dim: 3,
            is_singular: true,
            value_dimension: 3,
        };

        // Test that same transfer vector results in same M2L matrices
        let a = MortonKey::from_point(&point, &domain, 3);
        let other_a = a.siblings()[2];
        let res_a = a.find_transfer_vector(&other_a);

        let b = MortonKey::from_point(&point, &domain, 7);
        let other_b = b.siblings()[2];
        let res_b = b.find_transfer_vector(&other_b);
        assert_eq!(res_a, res_b);

        let target_check_surface_a = a.compute_surface(order, alpha_inner, &domain);
        let source_equivalent_surface_a = other_a.compute_surface(order, alpha_inner, &domain);
        let se2tc_a = kernel
            .gram(&source_equivalent_surface_a, &target_check_surface_a)
            .unwrap();
        let se2tc_a = Array::from(se2tc_a);

        let target_check_surface_b = b.compute_surface(order, alpha_inner, &domain);
        let source_equivalent_surface_b = other_b.compute_surface(order, alpha_inner, &domain);
        let se2tc_b = kernel
            .gram(&source_equivalent_surface_b, &target_check_surface_b)
            .unwrap();
        let se2tc_b = Array::from(se2tc_b);
        //
        // println!("SE2TC A {:?}", KiFmm::m2l_scale(7) * se2tc_a);
        // println!("SE2TC B {:?}", se2tc_b);
        // assert!(false)
    }

    #[test]
    fn test_m2l() {
        // Test that the local expansions of a given target node correspond to the
        // multipole expansions of source nodes in its v list
        // Create Kernel
        let kernel = Box::new(LaplaceKernel {
            dim: 3,
            is_singular: true,
            value_dimension: 3,
        });

        // Create FmmTree
        let npoints: usize = 1000;
        let points = points_fixture(npoints);
        let point_data = vec![1.0; npoints];
        let depth = 2;
        let n_crit = 150;

        let tree = Box::new(SingleNodeTree::new(
            &points,
            &point_data,
            false,
            Some(n_crit),
            Some(depth),
        ));

        // New FMM
        let mut kifmm = KiFmm::new(5, 1.05, 1.95, tree, kernel);

        kifmm.run();

        for key in kifmm.tree.get_leaves().iter() {
            if let Some(v_list) = kifmm.tree.get_interaction_list(key) {
                let downward_equivalent_surface =
                    key.compute_surface(kifmm.order, kifmm.alpha_outer, kifmm.tree.get_domain());
                let downward_check_surface =
                    key.compute_surface(kifmm.order, kifmm.alpha_inner, kifmm.tree.get_domain());
                let local_expansion = kifmm.tree.get_local_expansion(key).unwrap();
                let equivalent = kifmm
                    .kernel
                    .potential(
                        &downward_equivalent_surface,
                        &local_expansion,
                        &downward_check_surface,
                    )
                    .unwrap();

                let mut direct = vec![0_f64; local_expansion.len()];

                for source in v_list.iter() {
                    let upward_equivalent_surface = source.compute_surface(
                        kifmm.order,
                        kifmm.alpha_inner,
                        kifmm.tree.get_domain(),
                    );
                    let multipole_expansion = kifmm.tree.get_multipole_expansion(source).unwrap();

                    let tmp = kifmm
                        .kernel
                        .potential(
                            &upward_equivalent_surface,
                            &multipole_expansion,
                            &downward_check_surface,
                        )
                        .unwrap();

                    direct = direct.iter().zip(tmp.iter()).map(|(d, t)| d + t).collect();
                }
                for (a, b) in equivalent.iter().zip(direct.iter()) {
                    assert_approx_eq!(f64, *a, *b, epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_uniform_fmm() {
        // Create Kernel
        let kernel = Box::new(LaplaceKernel {
            dim: 3,
            is_singular: true,
            value_dimension: 3,
        });

        // Create FmmTree
        let npoints: usize = 10000;
        let points = points_fixture(npoints);
        let point_data = vec![1.0; npoints];
        let depth = 3;
        let n_crit = 150;

        let tree = Box::new(SingleNodeTree::new(
            &points,
            &point_data,
            false,
            Some(n_crit),
            Some(depth),
        ));

        // New FMM
        let mut kifmm = KiFmm::new(5, 1.05, 1.95, tree, kernel);

        kifmm.run();

        let leaves = vec![kifmm.tree.get_leaves()[0], kifmm.tree.get_leaves()[1]];
        let fmm_potential: Vec<f64> = leaves
            .iter()
            .flat_map(|l| kifmm.tree.get_point_data(l).unwrap())
            .map(|d| d[1])
            .collect();
        let node_points: Vec<_> = leaves
            .iter()
            .flat_map(|l| kifmm.tree.get_points(l).unwrap())
            .collect();
        let node_points_coordinates: Vec<[f64; 3]> =
            node_points.iter().map(|p| p.coordinate).collect();
        let direct_potential = kifmm
            .kernel
            .potential(&points, &point_data, &node_points_coordinates)
            .unwrap();

        // Test whether answers are within 4 digits of each other
        for (a, b) in fmm_potential.iter().zip(direct_potential.iter()) {
            assert_approx_eq!(f64, *a, *b, epsilon = 1.0);
        }
    }

    #[test]
    fn test_adaptive_fmm() {
        // Create Kernel
        let kernel = Box::new(LaplaceKernel {
            dim: 3,
            is_singular: true,
            value_dimension: 3,
        });

        // Create FmmTree
        let npoints: usize = 10000;
        let points = points_fixture(npoints);
        let point_data = vec![1.0; npoints];
        let depth = None;
        let n_crit = 150;

        let tree = Box::new(SingleNodeTree::new(
            &points,
            &point_data,
            true,
            Some(n_crit),
            depth,
        ));

        // New FMM
        let mut kifmm = KiFmm::new(5, 1.05, 1.95, tree, kernel);

        kifmm.run();

        let leaves: Vec<MortonKey> = kifmm.tree.get_leaves().iter().sorted().cloned().collect();

        let leaves = vec![leaves[5], leaves[6]];
        let fmm_potential: Vec<f64> = leaves
            .iter()
            .flat_map(|l| kifmm.tree.get_point_data(l).unwrap())
            .map(|d| d[1])
            .collect();
        let node_points: Vec<_> = leaves
            .iter()
            .flat_map(|l| kifmm.tree.get_points(l).unwrap())
            .collect();

        let node_points_coordinates: Vec<[f64; 3]> =
            node_points.iter().map(|p| p.coordinate).collect();
        let direct_potential = kifmm
            .kernel
            .potential(&points, &point_data, &node_points_coordinates)
            .unwrap();

        // Test whether answers are within 4 digits of each other
        for (a, b) in fmm_potential.iter().zip(direct_potential.iter()) {
            assert_approx_eq!(f64, *a, *b, epsilon = 1.0);
        }
    }
}
