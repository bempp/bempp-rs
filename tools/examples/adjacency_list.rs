use solvers_tools::arrays::AdjacencyList;

fn main() {
    // Create an empty adjacency list
    let mut a = AdjacencyList::<usize>::new();

    // Add rows to an adjacency list
    a.add_row(&[0, 1, 4]);
    a.add_row(&[1, 5, 9, 2, 6]);

    // Set the first value in the first row of the adjacency list
    *a.get_mut(0, 0).unwrap() = 3;

    // Print values from the first row of the adjacency list
    println!(
        "The first row of a is {} {} {}",
        *a.get(0, 0).unwrap(),
        *a.get(0, 1).unwrap(),
        *a.get(0, 2).unwrap()
    );

    // Items can also be accessed without bound checking using the unchecked methods
    println!(
        "The entry in the first row and first column of a is {}",
        unsafe { *a.get_unchecked(0, 0) }
    );

    // The row can also be obtained as a slice
    let row0 = a.row(0).unwrap();
    println!("The first row of a is {} {} {}", row0[0], row0[1], row0[2]);

    // Print the number of rows of a
    println!("a has {} rows", a.num_rows());

    // Iterate through the rows of a
    for (i, row) in a.iter_rows().enumerate() {
        println!(
            "The sum of the values in row {} of a is {}",
            i,
            row.iter().sum::<usize>()
        );
    }

    // TODO: REMOVE
    // TESTING CI WILL FAIL
    println!("{}", a.len());
}
