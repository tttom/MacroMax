# Macroscopic Maxwell Solver

## Introduction
Efficient solving the macroscopic Maxwell equations in complex dielectric materials - in [Python](python) and [Matlab](matlab).

The material properties are defined on a rectangular grid (1D, 2D, or 3D) for which each voxel defines an isotropic or anisotropic permittivity. Optionally, a heterogeneous (anisotropic) permeability as well as bi-anisotropic coupling factors may be specified (e.g. for chiral media). The source, such as an incident laser field, is specified as an oscillating current-density distribution.

The method iteratively corrects an estimated solution for the electric field (default: all zero). Its memory
requirements are on the order of the storage requirements for the material properties and the electric field within the
calculation volume. Full details can be found in the [open-access](https://doi.org/10.1364/OE.27.011946) manuscript
["Calculating coherent light-wave propagation in large heterogeneous media"](https://doi.org/10.1364/OE.27.011946). When the machine learning library PyTorch is detected, the wave equations can also be solved on the cloud or a local GPU, as described in the paper [doi:10.34133/icomputing.0098](https://doi.org/10.34133/icomputing.0098).

The [source code](https://github.com/tttom/MacroMax) is available on [GitHub](https://github.com/tttom/MacroMax) under the
**[MIT License](https://opensource.org/licenses/MIT): [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)**

## When to use this method?
The algorithm is particularly efficient for solving complex scattering:
* wave problems such as those encountered in electromagnetism and acoustics,
* subject to temporal coherent (monochromatic) irradiation,
* and a limited variation of a gain-free potential (permittivity),
* in complex, heterogeneous, materials.

Although incoherent and aperiodic problems can be solved by solving multiple coherent problems,
alternative methods might be more appropriate for this type of problems.

## When not to use this method?
With the exception of gain materials, MacroMax works for a wide variety of problems. However, more appropriate solutions may exist when:
* an approximate solution is sufficient, &rarr; e.g. the beam propagation method
* the material has a simple structure, &rarr; e.g. Mie scattering from a perfect sphere
* coherence is not important, &rarr; e.g. ray tracing
* the variation in the complex potential (permittivity) is very large, e.g. due to the presence of conductors &rarr; e.g. finite element methods
* the material has gain, e.g. an active laser cavity &rarr; e.g. finite element method
* aperiodic time-dependence is important, &rarr; e.g. finite difference method
The convergence rate of MacroMax is approximately inversely proportional to the variation in the potential. Materials with a refractive index or extinction coefficients larger than 5 will generally lead to slow convergence. By consequence, the presence of a superconductor would lead to infinitely slow convergence (a.k.a. divergence).

## Pure Python and Matlab implementations
Please follow the links for further information, source code, and examples:
* [Python](python)
* [Matlab](matlab)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/macromax)](https://www.python.org/downloads)
[![PyPI - License](https://img.shields.io/pypi/l/macromax)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/macromax?label=version&color=808000)](https://github.com/corilim/MacroMax/tree/master/python)
[![PyPI - Status](https://img.shields.io/pypi/status/macromax)](https://pypi.org/project/macromax/tree/master/python)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/macromax?label=python%20wheel)](https://pypi.org/project/macromax/#files)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/macromax)](https://pypi.org/project/macromax/)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/corilim/MacroMax)](https://github.com/corilim/MacroMax)
[![GitHub last commit](https://img.shields.io/github/last-commit/corilim/MacroMax)](https://github.com/corilim/MacroMax)
[![Libraries.io dependency status for latest release](https://img.shields.io/librariesio/release/pypi/macromax)](https://libraries.io/pypi/macromax)
[![Documentation Status](https://readthedocs.org/projects/macromax/badge/?version=latest)](https://readthedocs.org/projects/macromax)
