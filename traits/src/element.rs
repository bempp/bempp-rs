use crate::cell::ReferenceCellType;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ElementFamily {
    Lagrange = 0,
    RaviartThomas = 1,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum MapType {
    Identity = 0,
    CovariantPiola = 1,
    ContravariantPiola = 2,
    L2Piola = 3,
}

pub struct TabulatedData<'a, F: FiniteElement> {
    data: Vec<f64>,
    element: &'a F,
    deriv_count: usize,
    point_count: usize,
    basis_count: usize,
    value_size: usize,
}

fn compute_derivative_count(nderivs: usize, cell_type: ReferenceCellType) -> Result<usize, ()> {
    match cell_type {
        ReferenceCellType::Point => Ok(0),
        ReferenceCellType::Interval => Ok(nderivs + 1),
        ReferenceCellType::Triangle => Ok((nderivs + 1) * (nderivs + 2) / 2),
        ReferenceCellType::Quadrilateral => Ok((nderivs + 1) * (nderivs + 2) / 2),
        ReferenceCellType::Tetrahedron => Ok((nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6),
        ReferenceCellType::Hexahedron => Ok((nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6),
        ReferenceCellType::Prism => Ok((nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6),
        ReferenceCellType::Pyramid => Ok((nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6),
    }
}

impl<'a, F: FiniteElement> TabulatedData<'a, F> {
    pub fn new(element: &'a F, nderivs: usize, npoints: usize) -> Self {
        let deriv_count = compute_derivative_count(nderivs, element.cell_type()).unwrap();
        let point_count = npoints;
        let basis_count = element.dim();
        let value_size = element.value_size();
        let data = vec![0.0; deriv_count * point_count * basis_count * value_size];
        Self {
            data,
            element,
            deriv_count,
            point_count,
            basis_count,
            value_size,
        }
    }

    pub fn get_mut(
        &mut self,
        deriv: usize,
        point: usize,
        basis: usize,
        component: usize,
    ) -> &mut f64 {
        // Debug here
        let index = ((deriv * self.point_count + point) * self.basis_count + basis)
            * self.value_size
            + component;
        self.data.get_mut(index).unwrap()
    }

    pub fn get(&mut self, deriv: usize, point: usize, basis: usize, component: usize) -> &f64 {
        // Debug here
        let index = ((deriv * self.point_count + point) * self.basis_count + basis)
            * self.value_size
            + component;
        self.data.get(index).unwrap()
    }

    pub fn deriv_count(&self) -> usize {
        self.deriv_count
    }
    pub fn point_count(&self) -> usize {
        self.point_count
    }
    pub fn basis_count(&self) -> usize {
        self.basis_count
    }
    pub fn value_size(&self) -> usize {
        self.value_size
    }
    pub fn element(&self) -> &'a F {
        self.element
    }
}

pub trait FiniteElement {
    fn cell_type(&self) -> ReferenceCellType;
    fn degree(&self) -> usize;
    fn highest_degree(&self) -> usize;
    fn family(&self) -> ElementFamily;
    fn dim(&self) -> usize;
    fn discontinuous(&self) -> bool;

    fn value_size(&self) -> usize;

    fn tabulate(&self, points: &[f64], nderivs: usize, data: &mut TabulatedData<Self>)
    where
        Self: Sized;

    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Vec<usize>;

    fn map_type(&self) -> MapType;
}
