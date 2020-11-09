#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
try:
    import multiprocessing
    nb_threads = multiprocessing.cpu_count()
    import os
    for _ in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[_] = str(nb_threads)
    import mkl
    mkl.set_num_threads(nb_threads)
except ModuleNotFoundError:
    pass
import numpy as np
import time

import macromax
import logging
macromax.log.setLevel(logging.WARNING)  # Suppress MacroMax information logs
from examples import log
from macromax.utils.array import Grid
from macromax.utils.bound import LinearBound


def calculate(dtype=np.complex64, vectorial=False, ndim=2) -> float:
    wavelength = 500e-9
    boundary_thickness = 2e-6
    beam_diameter = 1e-6
    k0 = 2 * np.pi / wavelength
    data_shape = np.ones(ndim, dtype=int) * 128
    if ndim < 3:
        data_shape[1] *= 2
    sample_pitch = wavelength / 4
    grid = Grid(data_shape, sample_pitch)

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
    refractive_index = np.ones(data_shape)
    refractive_index[np.sqrt(sum(_**2 for _ in grid)) < 0.5*np.amin(data_shape * sample_pitch)/2] = 1.5

    # Add boundary
    bound = LinearBound(grid, thickness=boundary_thickness, max_extinction_coefficient=0.3)

    start_time = time.perf_counter()
    solution = macromax.solve(grid, vacuum_wavelength=wavelength, source_distribution=source, bound=bound,
                              refractive_index=refractive_index, dtype=dtype,
                              callback=lambda s: s.iteration < 1000 and s.residue > 1e-5
                              )
    total_time = time.perf_counter() - start_time

    log.debug(f'Total time: {total_time:0.3f} s for {solution.iteration} iterations:' +
             f' ({1000 * total_time / solution.iteration:0.3f} ms) for a residue of {solution.residue:0.6f}.')

    return total_time / solution.iteration


def measure(dtype=np.complex64, vectorial=False, ndim=2, nb_trials: int = 10) -> float:
    log.info(('Vectorial' if vectorial else 'Scalar') + f' {ndim}D calculation with {dtype.__name__}...')
    iteration_time = min(calculate(dtype=dtype, vectorial=vectorial, ndim=ndim) for _ in range(nb_trials))
    log.info(f'Iteration time: {iteration_time * 1000:0.3f} ms.')
    return iteration_time


if __name__ == '__main__':
    nb_trials = 10
    benchmark_fft = False
    ndim = 2
    log.info(f'MacroMax version {macromax.__version__}')

    if benchmark_fft:
        log.info('Benchmarking FFT...')
        from macromax.utils import ft

        data_shape = np.array([1024, 1200, 6], dtype=np.int)
        nb_iterations = 100
        res = np.random.randn(*data_shape) + 1j * np.random.randn(*data_shape)

        start_time = time.perf_counter()
        for _ in range(nb_iterations):
            res = ft.fftn(res)
        total_time = time.perf_counter() - start_time
        log.info(f'Total time for FFT: {total_time:0.3f} s for {nb_iterations:d} iterations: ({total_time / nb_iterations:0.3f} ms).')

    measure(dtype=np.complex128, vectorial=True, ndim=ndim, nb_trials=nb_trials)
    measure(dtype=np.complex64, vectorial=True, ndim=ndim, nb_trials=nb_trials)
    measure(dtype=np.complex128, vectorial=False, ndim=ndim, nb_trials=nb_trials)
    measure(dtype=np.complex64, vectorial=False, ndim=ndim, nb_trials=nb_trials)
