# Bempp-rs

Bempp-rs is an open-source boundary element library written in Rust that can be used to assemble all the standard integral
kernels for Laplace and Helmholtz problems.

## Documentation
The latest documentation of the main branch of this repo is available at [bempp.github.io/bempp-rs/](https://bempp.github.io/bempp-rs/).

## Testing
The functionality of the library can be tested by running:
```bash
cargo test
```

## Examples
Examples of use can be found in the [examples folder](examples/).

## Getting help
Errors in the library should be added to the [GitHub issue tracker](https://github.com/bempp/bempp-rs/issues).

Questions about the library and its use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
Bempp-rs is licensed under a BSD 3-Clause licence. Full text of the licence can be found [here](LICENSE.md).

The *.txt files in the folder `simplex_rules` are taken from the electronic supplemental material of the paper [1],
which is licensed under a CC BY 4.0 license.

## References
[1] F. D. Witherden, P.E. Vincent, *On the identification of symmetric quadrature rules for finite element methods*,
    Computers & Mathematics with Applications 69 (2015), 1232-1241,
    [doi.org/10.1016/j.camwa.2015.03.017](https://doi.org/10.1016/j.camwa.2015.03.017).


