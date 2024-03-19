# Bempp-rs

Bempp-rs is an open-source boundary element library written in Rust that can be used to assemble all the standard integral
kernels for Laplace and Helmholtz problems.

## Overview of components of Bempp-rs
- [bempp-traits](traits/) defines the traits use by all the other components
- [hyksort](hyksort/) is an implementation of the Hyksort algorithm
- [bempp-tools](tools/) contains functionality used by other components
- [bempp-element](element/) can create finite elements on reference cells
- [bempp-grid](grid/) can create grids of triangles and quadrilaterals
- [bempp-quadrature](quadrature/) computes quadrature rules for singular and non-singular boundary integrals
- [bempp-kernel](kernel/) defines the Green's functions for Laplace and Helmholtz problems
- [bempp-bem](bem/) creates function spaces and assembles matrices
- [bempp-field](field/) computes metadata required by field transations for FMM
- [bempp-tree](tree/) creates octrees used by FMM
- [bempp-fmm](fmm/) is an implemenation of a fast multipole method (FMM)

## Documentation
The latest documentation of the main branch of this repo is available at [bempp.github.io/bempp-rs/](https://bempp.github.io/bempp-rs/).

## Testing
The functionality of the library can be tested by running:
```bash
cargo test
```

## Getting help
Errors in the library should be added to the [GitHub issue tracker](https://github.com/bempp/bempp-rs/issues).

Questions about the library and its use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
Bempp-rs is licensed under a BSD 3-Clause licence. Full text of the licence can be found [here](LICENSE.md).
