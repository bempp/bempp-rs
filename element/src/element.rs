//! Finite Element definitions

use crate::cell::*;
use crate::map::*;
pub mod lagrange;
pub use lagrange::*;
pub mod raviart_thomas;
pub use raviart_thomas::*;
pub use solvers_traits::cell::ReferenceCellType;
pub use solvers_traits::element::ElementFamily;
pub use solvers_traits::element::FiniteElement;
pub use solvers_traits::element::TabulatedData;

#[cfg(test)]
mod test {
    use crate::element::*;

    #[test]
    fn test_lagrange_1() {
        let e = LagrangeElement {
            celltype: ReferenceCellType::Triangle,
            degree: 1,
        };
        assert_eq!(e.value_size(), 1);
    }
}
