# Macroscopic Maxwell Solver

## Introduction
This Python 3 package enables solving the macroscopic Maxwell equations in complex dielectric materials.

The material properties are defined on a rectangular grid (1D, 2D, or 3D) for which each voxel defines an isotropic or
anisotropic permittivity. Optionally, a heterogeneous (anisotropic) permeability as well as bi-anisotropic coupling
factors may be specified (e.g. for chiral media). The source, such as an incident laser field, is specified as an
oscillating current-density distribution.

The method iteratively corrects an estimated solution for the electric field (default: all zero). Its memory
requirements are on the order of the storage requirements for the material properties and the electric field within the
calculation volume. Full details can be found in the [open-access](https://doi.org/10.1364/OE.27.011946) manuscript
["Calculating coherent light-wave propagation in large heterogeneous media"](https://doi.org/10.1364/OE.27.011946).
Automatic leveraging of detected GPU/Cloud is implemented using PyTorch (for further details [follow this link](https://arxiv.org/abs/2208.01118)).

Examples of usage can be found in [the examples/ sub-folder](examples). The [Complete MacroMax Documentation](https://macromax.readthedocs.io)
can be found at [https://macromax.readthedocs.io](https://macromax.readthedocs.io).
All [source code](https://github.com/corilim/MacroMax) is available on [GitHub](https://github.com/corilim/MacroMax) under the
**[MIT License](https://opensource.org/licenses/MIT): [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)**

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


## Installation
### Prerequisites

This library requires Python 3 with the ````numpy```` and ````scipy```` packages for the main calculations. These modules will be automatically installed.
The modules ````multiprocessing````, ````torch````,  ````pyfftw````, and ````mkl-fft```` (Intel(R) CPU specific) can significantly speed up the calculations.

The examples require ````matplotlib```` for displaying the results.
In the creation of this package for distribution, the ````pypandoc```` package is used for translating this document to
other formats. This is only necessary for software development.

The code has been tested on Python 3.7 and 3.10, though it is expected to work on versions 3.6 and above.

### Installing
Installing the ````macromax```` package and its mandatory dependencies is as straightforward as running the following command in a terminal:
````sh
pip install macromax
````
While this is sufficient to get started, optional packages are useful to display the results and to speed-up the calculations.


#### Optimizing execution speed on a CPU
The calculation time can be reduced to a fraction by ensuring that you have the fastest libraries installed for your system. In particular the FFTW library can easily halve the calculation time on a CPU:
````sh
pip install macromax pyFFTW
````
On some systems the [pyFFTW](https://pypi.org/project/pyFFTW/) Python package requires the separate installation of the [FFTW library](http://www.fftw.org/download.html); however, it is easy to install it using Anaconda with the commands:
```conda install fftw```, or on Debian-based systems with ```sudo apt-get install fftw```.

Alternatively, the [mkl-fft](https://github.com/IntelPython/mkl_fft) package is available for Intel(R) CPUs, though it may require compilation or relying on the [Anaconda](https://www.anaconda.com/) or [Intel Python](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html) distributions:
````sh
conda install -c intel intelpython
````

#### Leveraging a machine learning framework for GPU and cloud-based calculations
The calculation time can be reduced by several orders of magnitude using the PyTorch machine learning library. This can be as straightforward as using the appropriate runtime on [Google Colab](https://colab.research.google.com/). The MacroMax library can here be installed by prepending the command with an exclamation mark as follows:
````sh
!pip install macromax
````
For more details, check out the [Google Colab deployment example](examples/Introduction%20to%20MacroMax.ipynb).

Local GPUs can also be used provided that PyTorch has a compatible implementation. At the time of writing, these includes NVidia's CUDA-enabled GPU as well AMD's ROCm-enabled GPUs (on Linux). Prior to installing the PyTorch module following the [PyTorch Guide](https://pytorch.org/), install the appropriate  [CUDA](https://www.nvidia.co.uk/Download/index.aspx?lang=en-uk) or [ROCm drivers](https://docs.amd.com/) for your GPU.
Note that for PyTorch to work correctly, Nvidia drivers need to be up to date and match the installed CUDA version. At the time of writing, for CUDA version 11.6, PyTorch can be installed as follows using pip:
````sh
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
````
Specifics for your CUDA version and operating system are listed on [PyTorch Guide](https://pytorch.org/).

When PyTorch and a compatible GPU are detected, these will be used by default. If not, FFTW and mkl-fft will be used if available. Otherwise, NumPy and SciPy will be used as a fallback.
The default backend can be set at the start of your code, or by creating a text file named ```backend_config.json``` in the current working directory with contents as:
````json
[
  {"type": "torch", "device": "cuda"},
  {"type": "numpy"}
]
````
to choose PyTorch when a compatible GPU is available, and NumPy otherwise. Although this machine learning library can be used without a hardware accelerator, we found that NumPy (with FFTW) can be faster when no GPU is available. This backend selection rule is used by default.

Experimental support exists for alternative backend implementations such as [TensorFlow](https://www.tensorflow.org/). Please refer to the [source code](https://github.com/corilim/MacroMax/tree/master/python/macromax/backend) for the latest work in progress. Pull requests are always welcome.

#### Additional packages
The package comes with a submodule containing example code that should run as-is on most desktop installations of Python.
Some systems may require the installation of the ubiquitous ````matplotlib```` graphics library:
````sh
pip install matplotlib
````

The output logs can be colored by installing the coloredlogs packaged:
````sh
pip install coloredlogs
````

Building and distributing the library may require further packages as indicated below.

## Usage
The basic calculation procedure consists of the following steps:

1. define the material

2. define the coherent light source

3. call ````solution = macromax.solve(...)````

4. display the solution

The ````macromax```` package must be imported to be able to use the ````solve```` function. The package also contains several utility functions that may help in defining the property and source distributions.

Examples can be found in [the examples package in the examples/ folder](examples). Ensure that the entire `examples/` folder
is downloaded, including the `__init__.py` file with general definitions. Run the examples from the parent folder using e.g. `python -m examples.air_glass_air_1D`.

The complete functionality is described in the [Library Reference Documentation](https://macromax.readthedocs.io) at [https://macromax.readthedocs.io](https://macromax.readthedocs.io).


### Loading the Python 3 package
The ````macromax```` package can be imported using:
```python
import macromax
```

**Optional:**
If the package is installed without a package manager, it may not be on Python's search path.
If necessary, add the library to Python's search path, e.g. using:

```python
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
```

Reminder: this library requires Python 3, ````numpy````, and ````scipy````. Optionally, ````pyfftw```` can be used to speed up the calculations. The examples also require ````matplotlib````.

### Specifying the material
#### Defining the sampling grid
The material properties are sampled on a plaid uniform rectangular grid of voxels. The sample points are defined by one or more linearly increasing coordinate ranges, one range per dimensions. The coordinates must be specified in meters, e.g.:

```python
import numpy as np
x_range = 50e-9 * np.arange(1000)
```

Ranges for multiple dimensions can be passed to ````solve(...)```` as a tuple of ranges:
````ranges = (x_range, y_range)````, or the convenience object `Grid` in
the `macromax.utils.array` sub-package. The latter can be used as follows:

```python
data_shape = (200, 400)
sample_pitch = 50e-9  # or (50e-9, 50e-9)
grid = macromax.Grid(data_shape, sample_pitch)
```
This defines a uniformly spaced plaid grid, centered around the origin,
unless specified otherwise.


#### Defining the material property distributions
The material properties are defined by ndarrays of 2+N dimensions, where N can be up to 3 for three-dimensional samples. In each sample point, or voxel, a complex 3x3 matrix defines the anisotropy at that point in the sample volume. The first two dimensions of the ndarray are used to store the 3x3 matrix, the following dimensions are the spatial indices x, y, and z. Optionally, four complex ndarrays can be specified: ````epsilon````, ````mu````, ````xi````, and ````zeta````. These ndarrays represent the permittivity, permeability, and the two coupling factors, respectively.

When the first two dimensions of a property are found to be both a singleton, i.e. 1x1, that property is assumed to be isotropic. Similarly, singleton spatial dimensions are interpreted as homogeneity in that property.
The default permeability `mu` is 1, and the coupling constants are zero by default.

##### Boundary conditions
The underlying algorithm assumes [periodic boundary conditions](https://en.wikipedia.org/wiki/Periodic_boundary_conditions).
Alternative boundary conditions can be implemented by surrounding the calculation area with absorbing (or reflective) layers.
Back reflections can be suppressed by e.g. linearly increasing the imaginary part of the permittivity with depth into a boundary with a thickness of a few wavelengths.

### Defining the source
The coherent source is defined by as a spatially-variant free current density.
Although the current density may be non-zero in all of space, it is more common to
define a source at one of the edges of the volume, to model e.g. an incident laser beam;
or even as a single voxel, to simulate a dipole emitter.
The source density can be specified as a complex number, indicating the phase
and amplitude of the current at each point. If an extended source is defined,
care should be taken so that the source currents constructively interfere
in the desired direction. I.e. the current density at neighboring voxels should
have a phase difference matching the k-vector in the background medium.
Optionally, instead of a current density, the internally-used source distribution may be
specified directly. It is related to the current density as follows: `S = i omega mu_0 J` with units of rad s^-1 H m^-1 A m^-2 = rad V m^-3,
where `omega` is the angular frequency, and `mu_0` is the vacuum permeability, mu_0.

The source distribution is stored as a complex ndarray with 1+N dimensions.
The first dimension contains the current 3D direction and amplitude for each voxel. The complex argument indicates the relative phase at each voxel.

### Calculating the electromagnetic light field
Once the ````macromax```` module is imported, the solution satisfying the macroscopic Maxwell's equations is calculated by calling:

```python
solution = macromax.solve(...)
```

The function arguments to ````macromax.solve(...)```` can be the following:

* ````grid|x_range````: A Grid object, a vector (1D), or tuple of vectors (2D, or 3D) indicating the spatial coordinates of the sample points. Each vector must be a uniformly increasing array of coordinates, sufficiently dense to avoid aliasing artefacts.

* ````vacuum_wavelength|wave_number|anguler_frequency````: The wavelength in vacuum of the coherent illumination in units of meters.

* ````current_density```` or ````source_distribution````: An ndarray of complex values indicating the source value and direction at each sample point. The source values define the free current density in the sample. The first dimension contains the vector index, the following dimensions contain the spatial dimensions.
If the source distribution is not specified, it is calculated as
:math:`-i c k0 mu_0 J`, where `i` is the imaginary constant, `c`, `k0`, and `mu_0`, the light-speed, wavenumber, and permeability in
vacuum. Finally, `J` is the free current density (excluding the movement of bound charges in a dielectric), specified as the input argument current_density.
These input arguments should be ```numpy.ndarray```s with a shape as specified by the `grid` input argument, or have one extra dimension on the left to indicate the polarization. If polarization is not specified the solution to the _scalar_ wave equation is calculated. However, when polarization is specified the _vectorial_ problem is solved.
  The returned ```macromax.Solution``` object has the property ```vectorial``` to indicate whether polarization is accounted for or not.    

* ````refractive_index````: A complex ```numpy.ndarray``` of a shape as indicated by the `grid` argument. Each value indicates the refractive at the corresponding spatial grid point. Real values indicate a loss-less material. A positive imaginary part indicates the absorption coefficient, :math:`\kappa`. This input argument is not required if the permittivity, `epsilon` is specified.

* ````epsilon````: (optional, default: :math:`n^2`) A complex ```numpy.ndarray``` of a shape as indicated by the `grid` argument for _isotropic_ media, or a shape with two extra dimensions on the left to indicate _anisotropy/birefringence_. The array values indicate the relative permittivity at all sample points in space. The optional two first (left-most) dimensions may contain a 3x3 matrix at each spatial location to indicate the anisotropy/birefringence. By default the 3x3 identity matrix is assumed, scaled by the scalar value of the array without the first two dimensions. Real values indicate loss-less permittivity. This input argument is unit-less, it is relative to the vacuum permittivity.

Optionally one can also specify magnetic and coupling factors:

* ````mu````: A complex ndarray that defines the 3x3 permeability matrix at all sample points. The first two dimensions contain the matrix indices, the following dimensions contain the spatial dimensions.

* ````xi```` and ````zeta````: Complex ndarray that define the 3x3 coupling matrices at all sample points. This may be useful to model chiral materials. The first two dimensions contain the matrix indices, the following dimensions contain the spatial dimensions.

It is often useful to also specify a callback function that tracks progress. This can be done by defining the ````callback````-argument as a function that takes an intermediate solution as argument. This user-defined callback function can display the intermediate solution and check if the convergence is adequate. The callback function should return ````True```` if more iterations are required, and ````False```` otherwise. E.g.:

```python
callback=lambda s: s.residue > 0.01 and s.iteration < 1000
```
will iterate until the residue is at most 1% or until the number of iterations reaches 1,000.

The solution object (of the Solution class) fully defines the state of the iteration and the current solution as described below.

The ````macromax.solve(...)```` function returns a solution object. This object contains the electric field vector distribution as well as diagnostic information such as the number of iterations used and the magnitude of the correction applied in the last iteration. It can also calculate the displacement, magnetizing, and magnetic fields on demand. These fields can be queried as follows:
* ````solution.E````: Returns the electric field distribution.
* ````solution.H````: Returns the magnetizing field distribution.
* ````solution.D````: Returns the electric displacement field distribution.
* ````solution.B````: Returns the magnetic flux density distribution.
* ````solution.S````: The Poynting vector distribution in the sample.

The field distributions are returned as complex ````numpy```` ndarrays in which the first dimensions is the polarization or direction index. The following dimensions are the spatial dimensions of the problem, e.g. x, y, and z, for three-dimensional problems.

The solution object also keeps track of the iteration itself. It has the following diagnostic properties:
* ````solution.iteration````: The number of iterations performed.
* ````solution.residue````: The relative magnitude of the correction during the previous iteration.
and it can be used as a Python iterator.

Further information can be found in the [examples](https://github.com/corilim/MacroMax/python/examples/) and the [signatures of each function and class](https://github.com/corilim/MacroMax/python/macromax/).


### Complete Example
The following code loads the library, defines the material and light source, calculates the result, and displays it.
To keep this example as simple as possible, the calculation is limited to one dimension. Higher dimensional calculations
simply require the definition of the material and light source in 2D or 3D.

The first section of the code loads the ````macromax```` library module as well as its ````utils```` submodule. More

```python
import macromax

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook  # Uncomment this line in an iPython Jupyter notebook

#
# Define the material properties
#
wavelength = 500e-9  # [ m ] In SI units as everything else here
source_polarization = np.array([0, 1, 0])[:, np.newaxis]  # y-polarized

# Set the sampling grid
nb_samples = 1024
sample_pitch = wavelength / 10  # [ m ]  # Sub-sample for display
boundary_thickness = 5e-6  # [ m ]
x_range = sample_pitch * np.arange(nb_samples) - boundary_thickness  # [ m ]

# Define the medium as a spatially-variant permittivity
# Don't forget absorbing boundary:
dist_in_boundary = np.maximum(0, np.maximum(-x_range,
                                            x_range - (x_range[-1] - boundary_thickness)
                                            ) / boundary_thickness)
permittivity = 1.0 + 0.25j * dist_in_boundary  # unit-less, relative to vacuum permittivity
# glass has a refractive index of about 1.5
permittivity[(x_range >= 20e-6) & (x_range < 30e-6)] += 1.5**2
permittivity = permittivity[np.newaxis, np.newaxis, ...]  # Define an isotropic material

#
# Define the illumination source
#
# point source at x = 0
current_density = source_polarization * (np.abs(x_range) < sample_pitch/4)

#
# Solve Maxwell's equations
#
# (the actual work is done in this line)
solution = macromax.solve(x_range, vacuum_wavelength=wavelength,
                          current_density=current_density, epsilon=permittivity)

#
# Display the results
#
fig, ax = plt.subplots(2, 1, frameon=False, figsize=(8, 6))

x_range = solution.grid[0]  # coordinates
E = solution.E[1, :]  # Electric field in y
H = solution.H[2, :]  # Magnetizing field in z
S = solution.S_forw[0, :]  # Poynting vector in x
f = solution.f[0, :]  # Optical force in x
# Display the field for the polarization dimension
field_to_display = E
max_val_to_display = np.amax(np.abs(field_to_display))
poynting_normalization = np.amax(np.abs(S)) / max_val_to_display
ax[0].plot(x_range * 1e6,
           np.abs(field_to_display) ** 2 / max_val_to_display,
           color=[0, 0, 0])
ax[0].plot(x_range * 1e6, np.real(S) / poynting_normalization,
           color=[1, 0, 1])
ax[0].plot(x_range * 1e6, np.real(field_to_display),
           color=[0, 0.7, 0])
ax[0].plot(x_range * 1e6, np.imag(field_to_display),
           color=[1, 0, 0])
figure_title = "Iteration %d, " % solution.iteration
ax[0].set_title(figure_title)
ax[0].set_xlabel("x  [$\mu$m]")
ax[0].set_ylabel("I, E  [a.u., V/m]")
ax[0].set_xlim(x_range[[0, -1]] * 1e6)

ax[1].plot(x_range[-1] * 2e6, 0,
           color=[0, 0, 0], label='I')
ax[1].plot(x_range[-1] * 2e6, 0,
           color=[1, 0, 1], label='$S_{real}$')
ax[1].plot(x_range[-1] * 2e6, 0,
           color=[0, 0.7, 0], label='$E_{real}$')
ax[1].plot(x_range[-1] * 2e6, 0,
           color=[1, 0, 0], label='$E_{imag}$')
ax[1].plot(x_range * 1e6, permittivity[0, 0].real,
           color=[0, 0, 1], label='$\epsilon_{real}$')
ax[1].plot(x_range * 1e6, permittivity[0, 0].imag,
           color=[0, 0.5, 0.5], label='$\epsilon_{imag}$')
ax[1].set_xlabel('x  [$\mu$m]')
ax[1].set_ylabel('$\epsilon$')
ax[1].set_xlim(x_range[[0, -1]] * 1e6)
ax[1].legend(loc='upper right')

plt.show(block=True)  # Not needed for iPython Jupyter notebook
```

### Optimization of time and memory efficiency
Electromagnetic calculations tend to test the limits of the hardware.
Two factors should be considered when optimizing the calculation: computation and memory.
Naturally, the number of operations and the duration of each operation should be minimized.
However, the latter is often dominated by memory accesses and copying of arrays.
The memory usage therefore does not only affect the size of the problems that can be solved,
it also tends to have an important impact on the total calculation time.

A straightforward method to reduce memory usage is to switch from 128-bit
precision complex numbers to 64-bit. By default, the precision of the
source_density is used, which is typically `np.complex128` or its real
equivalent. The `Solution`'s default `dtype` can be overridden by specifying
it as `solve(... dtype=np.complex64)`. Halving the storage requirements
can eliminate additional copies between the main memory and CPU cache.
In extreme cases it can also avoid swapping. Lower precision math also
executes faster on many architectures.

While oversampling to less than 1/10th of the wavelength may aid visualization,
it is often sufficient to sample at a quarter of the wavelength.
The sample solution represents a sinc-interpolated continuous function.
The final result can be visualized with arbitrary resolution using interpolation.

The number of operations can be kept to a minimum by:
* using non-magnetic and non-chiral materials,
* using isotropic materials,
* limiting the largest difference in permittivity (including the absorbing boundary), and
* using a scalar approximation
whenever possible.

Optimization of the implementation is another route to consider.
Potentially areas of improvement are:
* Profiling of memory usage and elimination of redundant temporary copies
* In-place fast-Fourier transforms. When available, the [FFTW](http://fftw.org/) library is used;
however, the drop-in fft and ifft replacements are used at the moment.
* Moving the calculations to a GPU or a cloud-computing environment.
Since the copying-overheads may quickly become a bottleneck, it is important
to consider the memory requirements for the problem you want to solve.


## Development
The [Library API Documentation](https://macromax.readthedocs.io) can be found at [https://macromax.readthedocs.io](https://macromax.readthedocs.io).
### Source code organization
The source code is organized as follows:
* [/](.) (root): Module description and distribution files.
* [macromax/](macromax/): The iterative solver.
    * [macromax/utils/](macromax/utils/): Helper functionality used in the solver and to use the solver.
* [examples/](examples/): Examples of how the solver can be used.
* [tests/](tests/): Automated unit tests of the solver's functionality. Use this after making modifications to the solver and extend it if new functionality is added.

The library functions are contained in ````macromax/````:
* [solver](macromax/solver.py): Defines the ````solve(...)```` function and the ````Solution```` class.
* [backend](macromax/backend/numpy.py): Defines linear algebra functions to work efficiently with large arrays of 3x3 matrices and 3-vectors.
* [utils/](macromax/utils/): Defines utility functions that can be used to prepare and interpret function arguments.

The included examples in the [examples/](examples/) folder are:
* [notebook_example.ipynb](examples/notebook_example.ipynb): An iPython notebook demonstrating basic usage of the library.
* [air_glass_air_1D.py](examples/air_glass_air_1D.py): Calculation of the back reflection from an air-glass interface (one-dimensional calculation)
* [air_glass_air_2D.py](examples/air_glass_air_2D.py): Calculation of the refraction and reflection of light hitting a glass window at an angle (two-dimensional calculation)
* [birefringent_crystal.py](examples/birefringent_crystal.py): Demonstration of how an anisotropic permittivity can split a diagonally polarized Gaussian beam into ordinary and extraordinary beams.
* [polarizer.py](examples/polarizer.py): Calculation of light wave traversing a set of two and a set of three polarizers as a demonstration of anisotropic absorption (non-Hermitian permittivity)
* [rutile.py](examples/rutile.py): Scattering from disordered collection of birefringent rutile (TiO2) particles.
* [benchmark.py](examples/benchmark.py): Timing of a simple two-dimensional calculation for comparison between versions.

### Testing
Unit tests are contained in ````macromax/tests````. The ````BackEnd```` class in ````backend.py```` is well covered and
specific tests have been written for the ````Solution```` class in ````solver.py````.

To run the tests, make sure that the `pytest` package is installed, and
run the following commands from the `Macromax/python/` directory:
```sh
pip install pytest
pytest --ignore=tests/test_backend_tensorflow.py
```

The benchmark script in the `examples/` folder can be used to compare performance for different problems and architectures.
Performance issues can be debugged using profilers as `pprofile` and `memory_profiler`, installed with:
````sh
pip install pprofile memory_profiler 
````

### Documentation
The `make` scripts in the `docs/` subdirectory automatically generate the documentation.
This uses Sphinx and extensions that can be installed with
````sh
pip install sphinx==5.3.0 sphinx_autodoc_typehints sphinxcontrib_mermaid sphinx-rtd-theme recommonmark
````
Examples of use can be found in the `examples/` and `tests/` folders. The former is more didactic, while the latter is more complete.

### Building and Distributing
The [source code](https://github.com/corilim/MacroMax) consists of pure Python 3, hence only packaging is required for distribution.
A package is generated by ````setup.py````, which relies on the ````m2r2```` package to translate `.md` files to `.rst` format:
````sh
pip install m2r2==0.3.2  # for compatibility with sphinx-rtd-theme, which requires an older version of docutils.
````

To prepare a package for distribution, increase the `__version__` number in [macromax/\_\_init\_\_.py](macromax/__init__.py), and run:

```sh
pip install build --upgrade
python -m build -nsw
pip install . --upgrade
```
The second line installs the newly-forged `macromax` package for testing.

The package can then be uploaded to a test repository as follows:
```sh
pip install twine --upgrade
twine upload -r testpypi dist/*
```

Installing from the test repository is done as follows:
```sh
pip install -i https://test.pypi.org/simple/ macromax --upgrade
```

To facilitate importing the code, IntelliJ IDEA/PyCharm project files can be found in ```MacroMax/python/```: ```MacroMax/python/python.iml``` and the folder ```MacroMax/python/.idea```.
