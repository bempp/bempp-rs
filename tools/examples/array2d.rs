use solvers_tools::arrays::Array2D;

fn main() {
    // Create an 2D array of zero integers of a given size
    // A 2D array is a two-dimensional storage format that allows items to be accessed using two indices (a row and a column)
    let mut a = Array2D::<usize>::new((5, 3));

    // Set values in the first row of the array
    *a.get_mut(0, 0).unwrap() = 3;
    *a.get_mut(0, 1).unwrap() = 1;
    *a.get_mut(0, 2).unwrap() = 4;

    // Print values from the first column of the array
    println!(
        "The first column of a is {} {} {} {} {}",
        *a.get(0, 0).unwrap(),
        *a.get(1, 0).unwrap(),
        *a.get(2, 0).unwrap(),
        *a.get(3, 0).unwrap(),
        *a.get(4, 0).unwrap()
    );

    // Items can also be accessed without bound checking using the unchecked methods
    println!(
        "The entry in the first row and first column of a is {}",
        unsafe { *a.get_unchecked(0, 0) }
    );

    // Print values from the first row of the array
    // This can be done in two ways
    println!(
        "The first row of a is {} {} {}",
        *a.get(0, 0).unwrap(),
        *a.get(0, 1).unwrap(),
        *a.get(0, 2).unwrap()
    );
    let row0 = a.row(0).unwrap();
    println!("The first row of a is {} {} {}", row0[0], row0[1], row0[2]);

    // Print the shape of the array
    println!("a has {} rows and {} columns", a.shape().0, a.shape().1);

    // Iterate through the rows of a
    for (i, row) in a.iter_rows().enumerate() {
        println!(
            "The sum of the values in row {} of a is {}",
            i,
            row.iter().sum::<usize>()
        );
    }
}
