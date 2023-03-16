//! Finite Element definitions

pub use bempp_traits::element::ElementFamily;
pub use bempp_traits::element::FiniteElement;
pub use bempp_traits::element::TabulatedData;
pub mod lagrange;
pub use lagrange::*;
pub mod raviart_thomas;
pub use raviart_thomas::*;

#[cfg(test)]
mod test {
    use crate::element::*;
    use bempp_traits::cell::ReferenceCellType;

    #[test]
    fn test_lagrange_1() {
        let e = LagrangeElement::new(ReferenceCellType::Triangle, 1);
        assert_eq!(e.value_size(), 1);
    }
}
