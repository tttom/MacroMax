# Macroscopic Maxwell Solver

## Introduction
This Matlab library enables solving the macroscopic Maxwell equations in complex dielectric materials.

The material properties are defined on a rectangular grid (1D, 2D, or 3D) for which each voxel defines an isotropic or anistropic permittivity. Optionally, a heterogeneous (anisotropic) permeability as well as bi-anisotropic coupling factors may be specified (e.g. for chiral media). The source, such as an incident laser field, is specified as an oscillating current-density distribution.

The method iteratively corrects an estimated solution for the electric field (default: all zero). Its memory requirements are on the order of the storage requirements for the material properties and the electric field within the calculation volume. Full details can be found in the [open-access](https://doi.org/10.1364/OE.27.011946) manuscript ["Calculating coherent light-wave propagation in large heterogeneous media."](https://doi.org/10.1364/OE.27.011946)

**[MIT License](https://opensource.org/licenses/MIT): [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)**

## Source organization
The root folder contains ````solveMacroscopicMaxwell.m```` with the main function. It also contains two functions to set and reset the Matlab path to this project: ````setPath```` and ````resetPath````.
Auxiliary functions are located in the ````utils```` subfolder.

Examples of usage can be found in the ````examples```` folder.

## Getting started
  1. Check out or download the source code.
  
  2. Open the dowload folder in Matlab.
  
  3. Run ````setPath```` 
  
  4. Run ````exampleScalar```
  
  5. ...
  
  6. Run ````resetPath````

## Matlab documentation
  Code documentation is included with the sources. Type F1 or
  ````sh
  helpwin solveMacroscopicMaxwell
  ````
  
  The input and output arguments are:
  ````
  [E, rmsError, itIdx] = solveMacroscopicMaxwell(ranges, k0, epsilon, xi, zeta, mu, sourceDistribution, progressFunction, E, lowPassBands)
 
  Calculates the vector field, E, that satisfies the Helmholtz equation for a
  potential and source's current density distribution given by chi and calcS at the coordinates specified by the ranges.
 
  The direct implementation keeps everything in memory all the time.
  This function assumes periodic boundary conditions. Add absorbing or PML
  layers to simulate other situations.
 
  Input parameters:
      ranges: a cell array of ranges to calculate the solution at. In the
              case of 1D, a simple vector may be provided. The length of the ranges
              determines the data_shape, which must match the dimensions of
              (the output of) epsilon, xi, zeta, mu, source, and the
              optional start field, unless these are singletons.
 
      k0: the wavenumber in vacuum = 2pi / wavelength.
              The wavelength in the same units as used for the other inputs/outputs.
 
      epsilon: an array or function that returns the (tensor) permittivity at the
              points indicated by the ranges specified as its input arguments.
      xi: an array or function that returns the (tensor) xi for bi-(an)isotropy at the
              points indicated by the ranges specified as its input arguments.
      zeta: an array or function that returns the (tensor) zeta for bi-(an)isotropy at the
              points indicated by the ranges specified as its input arguments.
      mu: an array or function that returns the (tensor) permeability at the
              points indicated by the ranges specified as its input arguments.
 
      sourceDistribution: an array or function that returns the (vector) source current density at the
              points indicated by the ranges specified as its input arguments. The
              source values relate to the current density, j, as source = 1i *
              angularFrequency * Const.mu_0 * j
 
      progressFunction: if specified (and not []), a function called after
              every iteration. The function should return true if the iteration is to continue, false otherwise.
              If instead of a function handle numeric values are provided, these are interpreted as the stop criterion
              maximum iteration and root-mean-square error as a 2-vector.
 
      E: (optional) the (vector) field to start the calculation from.
              Anything close to the solution will reduce the number of iterations
              required.
 
      lowPassBands: (optional, default: 'none') flag to indicate how to band
              limit the inputs and calculation to prevent aliasing. It can
              be specified as a vector with the low-pass fractions per band:
              [final, source, material, iteration], where final is the
              fraction of the final output to be retained, source that of
              the source at the start of the calculation, material that of
              the epsilon, xi, zeta, and mu, and iteration indicates
              whether to band limit at each step of the process. Note that
              the band-limiting of the material properties may introduce
              gain and thus divergence. The strings 'none', 'source',
              'susceptibility', 'input', 'iteration', can be provided as
              shorthands for 50% filtering of the corresponding.
 
  Output parameters:
      resultE: The solution field as an n-d array. If the source is
              vectorial, the first dimension is 3. If the source is scalar,
              the first dimension corresponds to the first range in ranges.
      rmsError: an estimate of the relative error
      itIdx: the iteration number at which the algorithm terminated
      
  ````
