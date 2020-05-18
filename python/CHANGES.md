# Macroscopic Maxwell Solver Change Log

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