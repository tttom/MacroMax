# Macroscopic Maxwell Solver Change Log

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