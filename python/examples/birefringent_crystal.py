#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Example code showing double refraction in a birefringent crystal


import matplotlib.pyplot as plt
import numpy as np
import time

import macromax
import macromax.bound
from macromax.utils.ft import Grid
from macromax.utils.display import complex2rgb, grid2extent
try:
    from examples import log
except ImportError:
    from macromax import log  # Fallback in case this script is not started as part of the examples package.


def show_birefringence():
    #
    # Medium settings
    #
    data_shape = np.array([128, 256]) * 2
    wavelength = 500e-9
    boundary_thickness = 5e-6
    beam_diameter = 2.5e-6
    k0 = 2 * np.pi / wavelength
    sample_pitch = np.array([1, 1]) * wavelength / 8
    grid = Grid(data_shape, sample_pitch)
    incident_angle = 0 * np.pi / 180

    def rot_z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    incident_k = rot_z(incident_angle) * k0 @ np.array([0, 1, 0])
    source_polarization = (rot_z(incident_angle) @ np.array([1, 0, 1]) / np.sqrt(2))[:, np.newaxis, np.newaxis]  # diagonally polarized beam
    current_density = np.exp(1j * (incident_k[0]*grid[0] + incident_k[1]*grid[1]))
    # Aperture the incoming beam
    current_density = current_density * np.exp(-0.5*(np.abs(grid[1] - (grid[1].ravel()[0]+boundary_thickness))/wavelength)**2)
    current_density = current_density * np.exp(-0.5*((grid[0] - grid[0].ravel()[int(len(grid[0])/4)])/(beam_diameter/2))**2)
    current_density = source_polarization * current_density

    permittivity = np.tile(np.eye(3, dtype=np.complex128)[:, :, np.newaxis, np.newaxis], (1, 1, *data_shape))
    # Add prism
    epsilon_crystal = rot_z(-np.pi/4) @ np.diag((1.486, 1.658, 1.658))**2 @ rot_z(np.pi/4)
    permittivity[:, :, :, int(data_shape[1]*(1-5/8)/2)+np.arange(int(data_shape[1]*5/8))] = \
        np.tile(epsilon_crystal[:, :, np.newaxis, np.newaxis], (1, 1, data_shape[0], int(data_shape[1]*5/8)))

    # Define boundaries
    bound = macromax.bound.LinearBound(grid, thickness=boundary_thickness, max_extinction_coefficient=0.1)

    # Prepare the display
    fig, axs = plt.subplots(3, 2, frameon=False, figsize=(12, 9))

    for ax in axs.ravel():
        ax.set_xlabel('y [$\mu$m]')
        ax.set_ylabel('x [$\mu$m]')
        ax.set_aspect('equal')

    images = [axs[dim_idx][0].imshow(complex2rgb(np.zeros(data_shape), 1),
                                     extent=np.array([*grid[1].ravel()[[0, -1]], *grid[0].ravel()[[0, -1]]]) * 1e6, origin='lower')
              for dim_idx in range(3)]
    axs[0][1].imshow(complex2rgb(permittivity[0, 0], 1),
                     extent=grid2extent(grid, origin_lower=True) * 1e6, origin='lower')
    axs[2][1].imshow(complex2rgb(current_density[0], 1),
                     extent=grid2extent(grid, origin_lower=True) * 1e6, origin='lower')
    axs[0][1].set_title('$\chi$')
    axs[1][1].axis('off')
    axs[2][1].set_title('source and S')
    Y, X = np.broadcast_arrays(*grid)
    arrow_sep = np.array([1, 1], dtype=int) * 30
    quiver = axs[2][1].quiver(X[::arrow_sep[0], ::arrow_sep[1]]*1e6, Y[::arrow_sep[0], ::arrow_sep[1]]*1e6,
                              X[::arrow_sep[0], ::arrow_sep[1]]*0, Y[::arrow_sep[0], ::arrow_sep[1]]*0,
                              pivot='mid', scale=1.0, scale_units='x', units='x', color=np.array([1, 0, 1, 0.5]))

    for dim_idx in range(3):
        for col_idx in range(2):
            axs[dim_idx][col_idx].autoscale(False, tight=True)

    #
    # Display the current solution
    #
    def display(s):
        log.info("Displaying iteration %d: error %0.1f%%" % (s.iteration, 100 * s.residue))
        nb_dims = s.E.shape[0]
        for dim_idx in range(nb_dims):
            images[dim_idx].set_data(complex2rgb(s.E[dim_idx], 1))
            figure_title = '$E_' + 'xyz'[dim_idx] + "$ it %d: rms error %0.1f%% " % (s.iteration, 100 * s.residue)
            axs[dim_idx][0].set_title(figure_title)

        S = s.S
        S /= np.sqrt(np.max(np.sum(np.abs(S) ** 2, axis=0))) / (sample_pitch[0] * arrow_sep[0])  # Normalize
        U = S[0, ...]
        V = S[1, ...]
        quiver.set_UVC(V[::arrow_sep[0], ::arrow_sep[1]]*1e6, U[::arrow_sep[0], ::arrow_sep[1]]*1e6)

        plt.draw()
        plt.pause(0.001)

    #
    # Display the (intermediate) result
    #
    def update_function(s):
        if np.mod(s.iteration, 10) == 0:
            log.info("Iteration %0.0f: rms error %0.1f%%" % (s.iteration, 100 * s.residue))
        if np.mod(s.iteration, 10) == 0:
            display(s)

        return s.residue > 1e-3 and s.iteration < 1e4

    # The actual work is done here:
    start_time = time.perf_counter()
    solution = macromax.solve(grid, vacuum_wavelength=wavelength, current_density=current_density,
                              epsilon=permittivity, bound=bound, callback=update_function, dtype=np.complex64
                              )
    log.info("Calculation time: %0.3fs." % (time.perf_counter() - start_time))

    # Show final result
    log.info('Displaying final result.')
    display(solution)
    plt.show(block=True)


if __name__ == "__main__":
    start_time = time.perf_counter()
    show_birefringence()
    log.info("Total time: %0.3fs." % (time.perf_counter() - start_time))
