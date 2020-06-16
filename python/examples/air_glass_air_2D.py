#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Example code showing reflection and refraction at a glass pane, in two dimensions

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pathlib

import macromax
from macromax.utils.array import Grid
from macromax.utils.display import complex2rgb, grid2extent
from examples import log


def show_scatterer(vectorial=True):
    output_path = pathlib.Path('output').absolute()
    output_filepath = pathlib.PurePath(output_path, 'air_glass_air_2D')

    #
    # Medium settings
    #
    scale = 2
    wavelength = 500e-9
    medium_refractive_index = 1.0  # 1.4758, 2.7114
    boundary_thickness = 2e-6
    beam_diameter = 1.0e-6 * scale
    plate_thickness = 2.5e-6 * scale

    k0 = 2 * np.pi / wavelength
    grid = Grid(np.array([256, 256]) * scale, wavelength / 16)
    incident_angle = 30 * np.pi / 180
    # incident_angle = np.arctan(1.5)  # Brewster's angle

    log.info('Calculating fields over a %0.1fum x %0.1fum area...' % tuple(grid.extent * 1e6))

    def rot_Z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    incident_k = rot_Z(incident_angle) * k0 @ np.array([1, 0, 0])
    source_polarization = (rot_Z(incident_angle) @ np.array([0, 1, 1j]) / np.sqrt(2))[:, np.newaxis, np.newaxis]
    current_density = np.exp(1j * (incident_k[0]*grid[0] + incident_k[1]*grid[1]))
    # Aperture the incoming beam
    # source = source * np.exp(-0.5*(np.abs(grid[1] - (grid[1].ravel()[0]+boundary_thickness))
    #                                * medium_refractive_index / wavelength)**2)  # source position
    source_pixel = grid.shape[0] - int(boundary_thickness / grid.step[0])
    current_density[:source_pixel, :] = 0
    current_density[source_pixel+1:, :] = 0
    current_density = current_density * np.exp(-0.5*((grid[1] - grid[1].ravel()[int(len(grid[0])*1/4)])/(beam_diameter/2))**2)  # beam aperture
    current_density = current_density[np.newaxis, ...]
    if vectorial:
        current_density = current_density * source_polarization

    # define the glass plate
    refractive_index = 1.0 + 0.5 * np.ones(grid[1].shape) * (np.abs(grid[0]) < plate_thickness/2)
    permittivity = np.array(refractive_index**2, dtype=np.complex64)
    permittivity = permittivity[np.newaxis, np.newaxis, :, :]

    # Add boundary
    dist_in_boundary = np.maximum(
        np.maximum(0.0, -(grid[0] - (grid[0].ravel()[0]+boundary_thickness)))
        + np.maximum(0.0, grid[0] - (grid[0].ravel()[-1]-boundary_thickness)),
        np.maximum(0.0, -(grid[1] - (grid[1].ravel()[0]+boundary_thickness)))
        + np.maximum(0.0, grid[1] - (grid[1].ravel()[-1]-boundary_thickness))
    )
    weight_boundary = dist_in_boundary / boundary_thickness
    for axis in range(permittivity.shape[0]):
        permittivity[axis, axis, :, :] += 0.5j * weight_boundary  # boundary

    # Prepare the display
    def add_rectangle_to_axes(axes):
        rectangle = plt.Rectangle(np.array((grid[1].ravel()[0], -plate_thickness / 2))*1e6,
                                  (grid.extent[1])*1e6, plate_thickness*1e6,
                                  edgecolor=np.array((0, 1, 1, 0.25)), linewidth=1, fill=True,
                                  facecolor=np.array((0, 1, 1, 0.05)))
        axes.add_artist(rectangle)

    fig, axs = plt.subplots(1 + vectorial, 2, frameon=False, figsize=(12, 9), sharex='all', sharey='all')
    for ax in axs.ravel():
        ax.set_xlabel('y [$\mu$m]')
        ax.set_ylabel('x [$\mu$m]')
        ax.set_aspect('equal')

    images = [ax.imshow(complex2rgb(np.zeros(grid.shape), 1, inverted=True), extent=grid2extent(grid) * 1e6)
              for ax in axs.ravel()]

    axs.ravel()[-1].set_title('$||E||^2$')

    # Display the medium without the boundaries
    for idx in range(axs.size):
        axs.ravel()[idx].set_xlim((grid[1].flatten()[0] + boundary_thickness) * 1e6,
                                       (grid[1].flatten()[-1] - boundary_thickness) * 1e6)
        axs.ravel()[idx].set_ylim((grid[0].flatten()[0] + boundary_thickness) * 1e6,
                                       (grid[0].flatten()[-1] - boundary_thickness) * 1e6)
        axs.ravel()[idx].autoscale(False)

    #
    # Display the current solution
    #
    def display(s):
        log.info('Displaying iteration %d: error %0.1f%%' % (s.iteration, 100 * s.residue))
        nb_dims = s.E.shape[0]
        for dim_idx in range(nb_dims):
            images[dim_idx].set_data(complex2rgb(s.E[dim_idx], 1, inverted=True))
            figure_title = '$E_' + 'xyz'[dim_idx] + "$ it %d: rms error %0.1f%% " % (s.iteration, 100 * s.residue)
            add_rectangle_to_axes(axs.ravel()[dim_idx])
            axs.ravel()[dim_idx].set_title(figure_title)
        intensity = np.linalg.norm(s.E, axis=0)
        intensity /= np.max(intensity)
        intensity_rgb = np.concatenate((intensity[:, :, np.newaxis], intensity[:, :, np.newaxis], intensity[:, :, np.newaxis]), axis=2)
        images[-1].set_data(intensity_rgb)
        add_rectangle_to_axes(axs.ravel()[-1])
        axs.ravel()[-1].set_title('I')

        plt.draw()
        plt.pause(0.001)

    #
    # Display progress and the (intermediate) result
    #
    residues = []
    times = []

    def update_function(s):
        # Log progress
        times.append(time.time())
        residues.append(s.residue)

        if np.mod(s.iteration, 10) == 0:
            log.info("Iteration %0.0f: rms error %0.3f%%" % (s.iteration, 100 * s.residue))
        if np.mod(s.iteration, 10) == 1:
            display(s)

        return s.residue > 1e-4 and s.iteration < 1e4

    # The actual work is done here:
    start_time = time.time()
    solution = macromax.solve(grid, vacuum_wavelength=wavelength, current_density=current_density,
                              epsilon=permittivity, callback=update_function, dtype=np.complex64
                              )
    log.info("Calculation time: %0.3fs." % (time.time() - start_time))

    # Display how the method converged
    times = np.array(times) - start_time
    log.info("Calculation time: %0.3fs." % times[-1])

    # Show final result
    log.info('Displaying final result.')
    display(solution)
    plt.show(block=False)
    # Save the individual images
    log.info('Saving results to %s...' % output_filepath.as_posix())
    output_path.mkdir(parents=True, exist_ok=True)
    for axis in range(solution.E.shape[0]):
        plt.imsave(output_filepath.as_posix() + '_E%s.png' % chr(ord('x') + axis), complex2rgb(solution.E[axis], 1, inverted=True),
                   vmin=0.0, vmax=1.0, cmap=None, format='png', origin=None, dpi=600)
    # Save the figure
    plt.ioff()
    fig.savefig(output_filepath.as_posix() + '.pdf', bbox_inches='tight', format='pdf')
    plt.ion()

    return times, residues


def conj_transpose(a):
    return np.conj(a).swapaxes(-2, -1)


def conj_inner(a, b):
    return np.sum(np.conj(a) * b, axis=-1, keepdims=True)


if __name__ == "__main__":
    start_time = time.time()
    # times, residues = show_scatterer(vectorial=False)
    times, residues = show_scatterer()
    log.info("Total time: %0.3fs." % (time.time() - start_time))

    # Display how the method converged
    fig_summary, axs_summary = plt.subplots(1, 2, frameon=False, figsize=(12, 9))
    axs_summary[0].semilogy(times, residues)
    axs_summary[0].scatter(times[::100], residues[::100])
    axs_summary[0].set_xlabel('t [s]')
    axs_summary[0].set_ylabel(r'$||\Delta E|| / ||E||$')
    colormap_ranges = [-(np.arange(256) / 256 * 2 * np.pi - np.pi), np.linspace(0, 1, 256)]
    colormap_image = complex2rgb(
        colormap_ranges[1][np.newaxis, :] * np.exp(1j * colormap_ranges[0][:, np.newaxis]),
        inverted=True)
    axs_summary[1].imshow(colormap_image, extent=grid2extent(*colormap_ranges))

    plt.show(block=True)
