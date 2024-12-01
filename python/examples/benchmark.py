#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import numpy as np
import time

import macromax
from macromax.utils.ft import Grid
from macromax.bound import LinearBound
try:
    from examples import log
except ImportError:
    from macromax import log  # Fallback in case this script is not started as part of the examples package.

try:
    import multiprocessing
    nb_threads = multiprocessing.cpu_count()
    import os
    log.info(f'Setting maximum number of threads to {nb_threads}.')
    for _ in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[_] = str(nb_threads)
    import mkl
    mkl.set_num_threads(nb_threads)
except (ImportError, TypeError):
    pass


def calculate(dtype=np.complex64, magnetic=False, birefringent=False, vectorial=False, ndim=2) -> float:
    wavelength = 500e-9
    boundary_thickness = 2e-6
    beam_diameter = 1e-6
    k0 = 2 * np.pi / wavelength
    data_shape = np.ones(ndim, dtype=int) * 128
    sample_pitch = wavelength / 4
    grid = Grid(data_shape, sample_pitch)
    log.info(f'Testing a calculation volume of {"x".join(f"{_:0.0f}" for _ in grid.extent / wavelength)} wavelengths...')

    # Define source
    source = np.exp(1j * k0 * grid[1])  # propagate along axis 1
    # Aperture the incoming beam
    source = source * np.exp(-0.5*(np.abs(grid[1] - (grid[1].ravel()[0]+boundary_thickness))/wavelength)**2)
    source = source * np.exp(-0.5*((grid[0] - grid[0].ravel()[int(len(grid[0])*1/2)])/(beam_diameter/2))**2)
    if vectorial:  # polarize along axis 0
        polarization = np.array([1, 0, 0])
        for _ in range(ndim):
            polarization = polarization[:, np.newaxis]
        source = polarization * source

    # Define material with scatterer
    sphere_mask = np.sqrt(sum(_**2 for _ in grid)) < 0.5*np.amin(data_shape * sample_pitch)/2
    if not birefringent:  # An isotropic material
        permittivity = np.ones(data_shape)
        permittivity[sphere_mask] = 1.5**2
    else:  # An anisotropic material
        def rot_z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        epsilon_crystal = rot_z(-np.pi/4) @ np.diag((1.5, 1.3, 1.1))**2 @ rot_z(np.pi/4)
        permittivity = np.zeros((3, 3, *data_shape))
        for _ in range(permittivity.shape[0]):
            permittivity[_, _, ...] = 1.0
        permittivity[:, :, sphere_mask] = epsilon_crystal[..., np.newaxis]

    if magnetic:
        permeability = permittivity
    else:
        permeability = 1.0

    # Add boundary
    bound = LinearBound(grid, thickness=boundary_thickness, max_extinction_coefficient=0.3)

    start_time = time.perf_counter()
    solution = macromax.solve(grid, vacuum_wavelength=wavelength, source_distribution=source, bound=bound,
                              epsilon=permittivity, mu=permeability, dtype=dtype,
                              callback=lambda s: s.iteration < 1000 and s.residue > 1e-6
                              )
    total_time = time.perf_counter() - start_time

    log.info(f'Total time: {total_time:0.3f} s for {solution.iteration} iterations:' +
             f' ({1000 * total_time / solution.iteration:0.3f} ms) for a residue of {solution.residue:0.6f}.')

    return total_time / solution.iteration


def measure(dtype=np.complex64, magnetic=False, birefringent=False, vectorial=False, ndim=2, nb_trials: int = 10) -> float:
    if magnetic or birefringent:
        vectorial = True
    log.info(f'{nb_trials} ' + ('vectorial' if vectorial else 'scalar') + (' and magnetic' if magnetic else '') + (' and birefringent' if birefringent else '') + f' {ndim}D calculations with {dtype.__name__}...')
    iteration_time = min(calculate(dtype=dtype, magnetic=magnetic, birefringent=birefringent, vectorial=vectorial, ndim=ndim) for _ in range(nb_trials))
    log.info(f'Minimum iteration time: {iteration_time * 1000:0.3f} ms.')
    return iteration_time


if __name__ == '__main__':
    nb_trials = 10
    ndim = 3
    dtype = np.complex64  # np.complex128

    log.info(f'MacroMax version {macromax.__version__}')
    #
    measure(dtype=dtype, vectorial=False, ndim=ndim, nb_trials=nb_trials)
    measure(dtype=dtype, vectorial=True, ndim=ndim, nb_trials=nb_trials)
    measure(dtype=dtype, birefringent=True, ndim=ndim, nb_trials=nb_trials)
    measure(dtype=dtype, magnetic=True, ndim=ndim, nb_trials=nb_trials)
    measure(dtype=dtype, magnetic=True, birefringent=True, ndim=ndim, nb_trials=nb_trials)
