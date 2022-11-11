//! Data IO.
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;

use vtkio::model::*;

use serde::Serialize;

use crate::types::{domain::Domain};

// VTK compatible dataset for visualization
pub trait VTK {
    // Convert a data set to VTK format.
    fn write_vtk(&self, filename: String, domain: &Domain);
}

// JSON input and output
pub trait JSON {
    // Save data to disk in JSON.
    fn write_json(&self, filename: String) -> Result<(), std::io::Error>
    where
        Self: Serialize,
    {
        let filepath = Path::new(&filename);
        let file = File::create(filepath)?;
        let writer = BufWriter::new(file);
        let result = serde_json::to_writer(writer, self)?;

        Ok(result)
    }

    // Read data from a 1D sequence into a Rust vector.
    fn read_json<'de, P: AsRef<Path>, T: serde::de::DeserializeOwned>(
        filepath: P,
    ) -> Result<Vec<T>, std::io::Error> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let result: Vec<T> = serde_json::from_reader(reader)?;
        Ok(result)
    }
}

// HDF5 input and output wrapper for simple serializable types.
pub trait HDF5<T: hdf5::H5Type> {
    // Save a Rust data to disk.
    fn write_hdf5<P: AsRef<Path>>(&self, filename: P) -> hdf5::Result<()>;

    // Read data from a 1D sequence into a Rust vector.
    fn read_hdf5<P: AsRef<Path>>(filepath: P) -> hdf5::Result<Vec<T>>;
}
