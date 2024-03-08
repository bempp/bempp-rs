1. Test vector input as well as derivative evaluations
2. Add source to target translations for BLAS m2l
3. Add tests for field crate
5. Add docs for new functionality and usage
6. Add examples and benchmarks
7. Make sure displacement function in fft m2l is multithreaded
8. Fix precomputations of m2l matrices, should be taking place on level 2 not 3, makes definition of scaling function a little weird.