# Version History

## Version 0.1.5
* Added PyTorch back-end, thus enabling the use of a GPU (NVidia CUDA).
* Ensured backwards compatibility with Python 3.6.

## Version 0.1.4
* Streamlined iteration: 3x speed improvement of benchmark with PyFFTW on Intel Core i7-6700.
* Reorganized parallel_ops backend.
* Expanded unit testing.

## Version 0.1.3
* Input arguments for isotropic materials or scalar calculations do not require singleton dimensions on the left anymore.
* `macromax.solve(...)` and `macromax.Solution(...)` now take the optional input argument `refractive_index` as
an alternative to the permittivity and permeability.
* The `macromax.bound` module provides the `Bound` class and subclasses to more conveniently specify arbitrary
absorbing or periodic boundaries. Custom implementations can be specified as a subclass of `Bound`.
* Convenience class `macromax.Grid` provides an easy method to construct uniformly plaid sample grids and their Fourier-space counterparts.   

## Version 0.1.2
* The solve function and Solution constructor now take `dtype` as an 
argument. By setting dtype=np.complex64 instead of the default 
dtype=np.complex128, all calculations will be done in single precision.
This only requires half the memory, and typically also reduces the
calculation time by about 30%.

## Version 0.1.1
* A Grid object can now be used to specify the sampling grid spacing and extent.

## Version 0.1.0
* The current density can now be specified directly for `solve(...)`
instead of the `source_density`.
* Reorganized the file structure.
* Removed the `k0^2 = (2pi/wavelength)^2` factor again after the calculations.
This normalization factor was introduced to avoid underflow in the
iteration. Now it is removed as intended, after the iteration, or when an
intermediate solution is queried.

## Version 0.0.9
* Made the dependency on the multiprocessing module optional.


## Version 0.0.8
* Extended unit tests, including the ```utils''' sub-module.
* Improved logging.
* Removed unused ```conductive''' indicator property from solver.
* Brought naming in line with Python conventions.

## Version 0.0.6
* Initial version on production PyPI repository.
