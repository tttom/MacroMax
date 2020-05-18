#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import numpy as np
import time

import macromax
import logging
macromax.log.setLevel(logging.WARNING)  # Suppress MacroMax information logs
from examples import log
from macromax.utils.array import Grid


def calculate(dtype=np.complex128, vectorial=False):
    wavelength = 500e-9
    boundary_thickness = 2e-6
    beam_diameter = 1e-6
    k0 = 2 * np.pi / wavelength
    data_shape = np.array([128, 256])
    sample_pitch = np.ones(2) * wavelength / 4
    grid = Grid(data_shape, sample_pitch)

    current_density = np.exp(1j * k0 * grid[1])  # propagate along axis 1
    # Aperture the incoming beam
    current_density = current_density * np.exp(-0.5*(np.abs(grid[1] - (grid[1].ravel()[0]+boundary_thickness))/wavelength)**2)
    current_density = current_density * np.exp(-0.5*((grid[0] - grid[0].ravel()[int(len(grid[0])*1/2)])/(beam_diameter/2))**2)
    if vectorial:
        # polarize along axis 0
        current_density = np.array([1, 0, 0])[:, np.newaxis, np.newaxis] * current_density
    else:
        current_density = current_density[np.newaxis, ...]
    current_density = current_density.astype(dtype=dtype, copy=True)

    permittivity = np.ones(data_shape)
    # Add scatterer
    permittivity[np.sqrt(grid[0]**2 + grid[1]**2) < 0.5*np.amin(data_shape * sample_pitch)/2] = 1.5**2
    permittivity = permittivity[np.newaxis, np.newaxis, ...]  # mark this as isotropic

    # Add boundary
    dist_in_boundary = np.maximum(
        np.maximum(0.0, -(grid[0] - (grid[0].ravel()[0]+boundary_thickness)))
        + np.maximum(0.0, grid[0] - (grid[0].ravel()[-1]-boundary_thickness)),
        np.maximum(0.0,-(grid[1] - (grid[1].ravel()[0]+boundary_thickness)))
        + np.maximum(0.0, grid[1] - (grid[1].ravel()[-1]-boundary_thickness))
    )
    weight_boundary = dist_in_boundary / boundary_thickness
    permittivity = permittivity + 0.3j * weight_boundary  # boundary

    display = False
    if display:
        import matplotlib.pyplot as plt
        from macromax.utils import complex2rgb, grid2extent

        # Prepare the display
        fig, axs = plt.subplots(1, 2, frameon=False, figsize=(12, 6))
        for ax in axs.ravel():
            ax.set_xlabel('y [$\mu$m]')
            ax.set_ylabel('x [$\mu$m]')
            ax.set_aspect('equal')

        images = axs[0].imshow(complex2rgb(np.zeros(data_shape), 1), extent=grid2extent(grid[0].ravel(), grid[1].ravel()) * 1e6)
        axs[1].imshow(complex2rgb(permittivity[0, 0], 1), extent=grid2extent(grid[0].ravel(), grid[1].ravel()) * 1e6)
        axs[1].set_title('$\chi$')

        #
        # Display the current solution
        #
        def display(s):
            log.info("2D: Displaying iteration %d: error %0.1f%%" % (s.iteration, 100 * s.residue))
            images.set_data(complex2rgb(s.E[0], 1))
            figure_title = '$E' + "$ it %d: rms error %0.1f%% " % (s.iteration, 100 * s.residue)
            axs[0].set_title(figure_title)

            plt.draw()
            plt.pause(0.001)

        def update_function(s):
            if np.mod(s.iteration, 10) == 0:
                log.info("Iteration %0.0f: rms error %0.1f%%" % (s.iteration, 100 * s.residue))
            if np.mod(s.iteration, 10) == 0:
                display(s)

            return s.residue > 1e-3 and s.iteration < 1000
    else:
        def update_function(s):
            return s.residue > 1e-4 and s.iteration < 1000

    start_time = time.perf_counter()
    solution = macromax.solve((grid[0].ravel(), grid[1].ravel()), vacuum_wavelength=wavelength, current_density=current_density, epsilon=permittivity,
                              callback=update_function
                              )
    total_time = time.perf_counter() - start_time

    return total_time, solution.iteration, solution.residue


if __name__ == '__main__':
    log.info(f'MacroMax version {macromax.__version__}')

    log.info('Vectorial 2D calculation with np.complex128...')
    total_time, nb_iterations, residue = calculate(vectorial=True)
    log.info('Total time: %0.3f s for %d iterations: (%0.3f ms) for a residue of %0.6f.'
             % (total_time, nb_iterations, 1000 * total_time / nb_iterations, residue))

    log.info('Vectorial 2D calculation with np.complex64...')
    total_time, nb_iterations, residue = calculate(dtype=np.complex64, vectorial=True)
    log.info('Total time: %0.3f s for %d iterations: (%0.3f ms) for a residue of %0.6f.'
             % (total_time, nb_iterations, 1000 * total_time / nb_iterations, residue))

    log.info('Scalar 2D calculation with np.complex128...')
    total_time, nb_iterations, residue = calculate()
    log.info('Total time: %0.3f s for %d iterations: (%0.3f ms) for a residue of %0.6f.'
             % (total_time, nb_iterations, 1000 * total_time / nb_iterations, residue))

    log.info('Scalar 2D calculation with np.complex64...')
    total_time, nb_iterations, residue = calculate(dtype=np.complex64)
    log.info('Total time: %0.3f s for %d iterations: (%0.3f ms) for a residue of %0.6f.'
             % (total_time, nb_iterations, 1000 * total_time / nb_iterations, residue))



