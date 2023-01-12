pub struct Array2D<T> {
    data: Vec<T>,
    shape: (usize, usize),
}
impl<T> Array2D<T> {
    pub fn new(data: Vec<T>, shape: (usize, usize)) -> Self {
        Self {
            data: data,
            shape: shape,
        }
    }
    pub fn get(&mut self, index0: usize, index1: usize) -> &T {
        self.data.get(index0 * self.shape.1 + index1).unwrap()
    }
    pub fn get_mut(&mut self, index0: usize, index1: usize) -> &mut T {
        self.data.get_mut(index0 * self.shape.1 + index1).unwrap()
    }
    pub fn row(&mut self, index: usize) -> &[T] {
        &self.data[index * self.shape.1..(index + 1) * self.shape.1]
    }
}

#[cfg(test)]
mod test {
    use crate::arrays::*;

    #[test]
    fn test_array_2d() {
        let mut arr = Array2D::new(vec![1, 2, 3, 4, 5, 6], (2, 3));
        assert_eq!(*arr.get(0, 0), 1);
        assert_eq!(*arr.get(0, 1), 2);
        assert_eq!(*arr.get(0, 2), 3);
        assert_eq!(*arr.get(1, 0), 4);
        assert_eq!(*arr.get(1, 1), 5);

        let row1 = arr.row(1);
        assert_eq!(row1.len(), 3);
        assert_eq!(row1[0], 4);
        assert_eq!(row1[1], 5);
        assert_eq!(row1[2], 6);

        *arr.get_mut(1, 2) = 7;
        assert_eq!(*arr.get(1, 2), 7);

        let mut arr2 = Array2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
        assert_eq!(*arr2.get(0, 0), 1.0);
        assert_eq!(*arr2.get(0, 1), 2.0);
        assert_eq!(*arr2.get(0, 2), 3.0);
        assert_eq!(*arr2.get(1, 0), 4.0);
        assert_eq!(*arr2.get(1, 1), 5.0);

        let row1 = arr2.row(1);
        assert_eq!(row1.len(), 3);
        assert_eq!(row1[0], 4.0);
        assert_eq!(row1[1], 5.0);
        assert_eq!(row1[2], 6.0);

        *arr2.get_mut(1, 2) = 7.;
        assert_eq!(*arr2.get(1, 2), 7.0);
    }
}
