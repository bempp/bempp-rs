# hyksort

An implementation of the hyksort algorithm, as defined in [1].
This implementation is based on the implementation in https://github.com/hsundar/usort.
hyksort is a component of [Bempp-rs](https://github.com/bempp/bempp-rs).

## Testing
The functionality of this component can be tested by running:
```bash
cargo test
```

## Example usage
Examples of how this component can be used can be found in the [examples](examples/) folder.

## Licence
hyksort is licensed under a BSD 3-Clause licence. Full text of the licence can be found [here](../LICENSE.md).

## References
[1] H. Sundar, D. Malhotra, G. Biros, *Hyksort: a new variant of hypercube quicksort on distributed memory architectures*,
    Proceedings of the 27th international ACM conference on international conference on supercomputing (2013), 293-302,
    [doi.org/10.1145/2464996.2465442](https://doi.org/10.1145/2464996.2465442).

