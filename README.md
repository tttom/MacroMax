# Macroscopic Maxwell Solver

## Introduction
Efficient solving the macroscopic Maxwell equations in complex dielectric materials - in [Python](python) and [Matlab](matlab).

The material properties are defined on a rectangular grid (1D, 2D, or 3D) for which each voxel defines an isotropic or anisotropic permittivity. Optionally, a heterogeneous (anisotropic) permeability as well as bi-anisotropic coupling factors may be specified (e.g. for chiral media). The source, such as an incident laser field, is specified as an oscillating current-density distribution.

The method iteratively corrects an estimated solution for the electric field (default: all zero). Its memory requirements are on the order of the storage requirements for the material properties and the electric field within the calculation volume. Full details can be found in the [open-access](https://doi.org/10.1364/OE.27.011946) manuscript ["Calculating coherent light-wave propagation in large heterogeneous media."](https://doi.org/10.1364/OE.27.011946)

The [source code](https://github.com/tttom/MacroMax) is available on [GitHub](https://github.com/tttom/MacroMax) under the
**[MIT License](https://opensource.org/licenses/MIT): [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)**

### Pure Python and Matlab implementations
Please follow the links for further information and the source code:
* [Python](python)
* [Matlab](matlab)