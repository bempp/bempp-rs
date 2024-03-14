//! Grid input/output
use std::fs;

pub trait GmshIO {
    //! Grid I/O for Gmsh

    /// Generate the Gmsh string for a grid
    fn to_gmsh_string(&self) -> String;

    /// Export as Gmsh
    fn export_as_gmsh(&self, filename: String) {
        let gmsh_s = self.to_gmsh_string();
        fs::write(filename, gmsh_s).expect("Unable to write file");
    }
}
