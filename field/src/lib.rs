//! Fast Field Translations for kernel independent FMMs arranged on regular grids.
//! Examples include the Kernel Independent FMM (KiFMM) of Ying et. al (2004) and the
//! Black Box FMM (bbFMM) of Fong & Darve (2009).
pub mod array;
pub mod fft;
pub mod field;
pub mod hadamard;
pub mod surface;
pub mod transfer_vector;
pub mod types;
