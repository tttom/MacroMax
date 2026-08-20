# Contributing
The [Library API Documentation](https://macromax.readthedocs.io) can be found at [https://macromax.readthedocs.io].
## Source code organization
The source code is organized as follows:
* [/](https://github.com/corilim/MacroMax/tree/master/python/) (root): Module description and distribution files.
* [macromax/](https://github.com/corilim/MacroMax/tree/master/python/macromax/): The iterative solver.
    * [macromax/utils/](https://github.com/corilim/MacroMax/tree/master/python/macromax/utils/): Helper functionality used in the solver and to use the solver.
* [examples/](https://github.com/corilim/MacroMax/tree/master/python/examples/): Examples of how the solver can be used.
* [tests/](https://github.com/corilim/MacroMax/tree/master/python/tests/): Automated unit tests of the solver's functionality. Use this after making modifications to the solver and extend it if new functionality is added.

The library functions are contained in ````macromax/````:
* [solver](https://github.com/corilim/MacroMax/tree/master/python/macromax/solver.py): Defines the ````solve(...)```` function and the ````Solution```` class.
* [backend](https://github.com/corilim/MacroMax/tree/master/python/macromax/backend/numpy.py): Defines linear algebra functions to work efficiently with large arrays of 3x3 matrices and 3-vectors.
* [utils/](https://github.com/corilim/MacroMax/tree/master/python/macromax/utils/): Defines utility functions that can be used to prepare and interpret function arguments.

The included examples in the [examples/](https://github.com/corilim/MacroMax/tree/master/python/examples/) folder are:
* [notebook_example.ipynb](https://github.com/corilim/MacroMax/tree/master/python/examples/notebook_example.ipynb): An iPython notebook demonstrating basic usage of the library.
* [air_glass_air_1D.py](https://github.com/corilim/MacroMax/tree/master/python/examples/air_glass_air_1D.py): Calculation of the back reflection from an air-glass interface (one-dimensional calculation)
* [air_glass_air_2D.py](https://github.com/corilim/MacroMax/tree/master/python/examples/air_glass_air_2D.py): Calculation of the refraction and reflection of light hitting a glass window at an angle (two-dimensional calculation)
* [birefringent_crystal.py](https://github.com/corilim/MacroMax/tree/master/python/examples/birefringent_crystal.py): Demonstration of how an anisotropic permittivity can split a diagonally polarized Gaussian beam into ordinary and extraordinary beams.
* [polarizer.py](https://github.com/corilim/MacroMax/tree/master/python/examples/polarizer.py): Calculation of light wave traversing a set of two and a set of three polarizers as a demonstration of anisotropic absorption (non-Hermitian permittivity)
* [rutile.py](https://github.com/corilim/MacroMax/tree/master/python/examples/rutile.py): Scattering from disordered collection of birefringent rutile (TiO2) particles.
* [benchmark.py](https://github.com/corilim/MacroMax/tree/master/python/examples/benchmark.py): Timing of a simple two-dimensional calculation for comparison between versions.

## Testing
Unit tests are contained in ````macromax/tests````. The ````BackEnd```` class in ````backend.py```` is well covered and
specific tests have been written for the ````Solution```` class in ````solver.py````.

To run the tests, make sure to install [`Astral's uv`](https://docs.astral.sh/uv/),
```sh
pipx install uv  # or alternatively: https://docs.astral.sh/uv/getting-started/installation/
```
and set up the virtual environment as follows:
```sh
uv venv .venv --allow-existing --managed-python
uv lock --upgrade
uv sync --extra torch --group docs --group examples
```
where the extras are optional.
Then run the tests as follows from the root of the [Git](https://git-scm.com/) repository:
```sh
uv run --frozen pytest --ignore=tests/test_matrix.py
```
Running all tests can take several minutes, in particular the `test_matrix` takes significant time and is therefore ignored in the above.
Some tests are backend-specific and will fail if e.g. PyTorch is not installed; however, this should not affect the other tests.

The benchmark script in the `examples/` folder can be used to compare performance for different problems and architectures.
Performance issues can be debugged using profilers as `pprofile` and `memory_profiler`, installed with the default 
````sh
uv sync --extra dev
````
where the `--extra dev` can be omitted.

## Documentation
The `make` scripts in the `docs/` subdirectory automatically generate the documentation.
This uses Sphinx and its extensions, which are installed with
````sh
uv sync --group docs
````
Next, generate the documentation with
````sh
uv run python docs/make.py
````
You can find the html documentation in `docs/build/html/index.html`.

Examples of use can be found in the `examples/` and `tests/` folders. The former is more didactic, while the latter is more complete.

## Building and Distributing
The [source code] consists of pure Python 3, hence only packaging is required for distribution. A package is generated by ````setup.py````.

To prepare a package for distribution:

0. Make sure all tests pass and that the documentation builds, as described above.

0. Add the description of the new version at the top of the CHANGES.md file.

0. Build the source and wheel distributions:
```sh
uv build
```

0. If the build succeeds, commit and push your final changes:
```sh
git commit -am "Description of the changes."
git push
```

0. Tag the git commit with the next version and a release candidate number:
```sh
git tag v_._._rc1
git push v_._._rc1
```

0. Check that the package on PyPI works with pip install and check the documentation on readthedocs.com.