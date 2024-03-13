//! Grid input/output
use std::fs;

pub trait GmshIO {
    fn to_gmsh_string(&self) -> String;

    fn export_as_gmsh(&self, filename: String) {
        let gmsh_s = self.to_gmsh_string();
        fs::write(filename, gmsh_s).expect("Unable to write file");
    }
}
