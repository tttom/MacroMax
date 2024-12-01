#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Example code showing light scattering by a layer of rutile (TiO2) particles.

import matplotlib.pyplot as plt
import numpy as np
import time
import pathlib

import macromax
from macromax.bound import LinearBound
from macromax.utils.ft import Grid
from macromax.utils.display import complex2rgb, grid2extent
try:
    from examples import log
except ImportError:
    from macromax import log  # Fallback in case this script is not started as part of the examples package.
from examples.utils.sphere_packing import pack


def calculate_and_display_scattering(vectorial=True, anisotropic=True):
    if not vectorial:
        anisotropic = False

    output_path = pathlib.Path('output').absolute()
    output_filepath = pathlib.PurePath(output_path, 'rutile')

    #
    # Medium settings
    #
    scale = 2
    wavelength = 500e-9
    medium_refractive_index = 1.0
    boundary_thickness = 2e-6
    beam_diameter = 1.0e-6 * scale
    layer_thickness = 2.5e-6 * scale

    k0 = 2 * np.pi / wavelength
    grid = Grid(np.array([128, 256]) * scale, wavelength / 16)
    incident_angle = 0 * np.pi / 180

    log.info('Calculating fields over a %0.1fμm x %0.1fμm area...' % tuple(grid.extent * 1e6))

    def rot_Z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    incident_k = rot_Z(incident_angle) * k0 @ np.array([0, 1, 0])
    source_polarization = (rot_Z(incident_angle) @ np.array([1, 0, 1j]) / np.sqrt(2))[:, np.newaxis, np.newaxis]
    current_density = np.exp(1j * (incident_k[0]*grid[0] + incident_k[1]*grid[1]))
    # Aperture the incoming beam
    current_density = current_density * np.exp(-0.5*(np.abs(grid[1] - (grid[1].ravel()[0]+boundary_thickness))
                                   * medium_refractive_index/ wavelength)**2)  # source position
    current_density = current_density * np.exp(-0.5*((grid[0] - grid[0].ravel()[int(len(grid[0])*2/4)])/(beam_diameter/2))**2)  # beam aperture
    current_density = current_density[np.newaxis, ...]
    if vectorial:
        current_density = current_density * source_polarization

    # Place randomly oriented TiO2 particles
    start_time = time.perf_counter()
    permittivity, orientation, grain_pos, grain_rad, grain_dir = \
        generate_birefringent_random_layer(grid, layer_thickness=layer_thickness, radius_mean=0.5e-6,
                                           radius_std=0.1e-6, normal_dim=1,
                                           birefringent=anisotropic, medium_refractive_index=medium_refractive_index)
    log.info(f'{time.perf_counter() - start_time:0.6}s to generate layer with {grain_pos.shape[0]} grains.')

    if not anisotropic:
        permittivity = permittivity[:1, :1, ...]
    log.info('Sample ready.')

    # Prepare the display
    def add_circles_to_axes(axes):
        for r, pos in zip(grain_rad, grain_pos):
            circle = plt.Circle(pos[::-1]*1e6, r*1e6,
                                edgecolor=np.array((1, 1, 1))*0.0, facecolor=None, alpha=0.25, fill=False, linewidth=1)
            axes.add_artist(circle)

    fig, axs = plt.subplots(3, 2, frameon=False, figsize=(12, 9), sharex='all', sharey='all')
    for ax in axs.ravel():
        ax.set_xlabel(r'y [$\mu$m]')
        ax.set_ylabel(r'x [$\mu$m]')
        ax.set_aspect('equal')

    images = [axs[dim_idx][0].imshow(complex2rgb(np.zeros(grid.shape), 1, inverted=True),
                                     extent=grid2extent(grid) * 1e6)
              for dim_idx in range(3)]

    epsilon_abs = np.abs(permittivity[0, 0]) - 1
    # rgb_image = colors.hsv_to_rgb(np.stack((np.mod(direction / (2*np.pi), 1), 1+0*direction, epsilon_abs), axis=2))
    axs[0][1].imshow(complex2rgb(epsilon_abs * np.exp(1j * orientation), normalization=True, inverted=True),
                     zorder=0, extent=grid2extent(grid) * 1e6)
    add_circles_to_axes(axs[0][1])
    axs[1][1].imshow(complex2rgb(permittivity[0, 0], 1, inverted=True), extent=grid2extent(grid) * 1e6)
    axs[2][1].imshow(complex2rgb(current_density[0], 1, inverted=True), extent=grid2extent(grid) * 1e6)
    axs[0][1].set_title('crystal axis orientation')
    axs[1][1].set_title(r'$\chi$')
    axs[2][1].set_title('source')

    # Display the medium without the boundaries
    for dim_idx in range(len(axs)):
        for col_idx in range(len(axs[dim_idx])):
            axs[dim_idx][col_idx].set_xlim((grid[1].ravel()[0] + boundary_thickness) * 1e6,
                                           (grid[1].ravel()[-1] - boundary_thickness) * 1e6)
            axs[dim_idx][col_idx].set_ylim((grid[0].ravel()[0] + boundary_thickness) * 1e6,
                                           (grid[0].ravel()[-1] - boundary_thickness) * 1e6)
            axs[dim_idx][col_idx].autoscale(False)

    #
    # Display the current solution
    #
    def display(s):
        log.info(f'Displaying iteration {s.iteration}: update = {s.residue * 100:0.1f}%.')
        nb_dims = s.E.shape[0]
        for dim_idx in range(nb_dims):
            images[dim_idx].set_data(complex2rgb(s.E[dim_idx], 1, inverted=True))
            figure_title = '$E_' + 'xyz'[dim_idx] + f'$ it {s.iteration}: update = {s.residue * 100:0.1f}%'
            add_circles_to_axes(axs[dim_idx][0])
            axs[dim_idx][0].set_title(figure_title)

        plt.draw()
        plt.pause(0.001)

    #
    # Display progress and the (intermediate) result
    #
    residues = []
    times = []

    def update_function(s):
        # Log progress
        times.append(time.perf_counter())
        residues.append(s.residue)

        if np.mod(s.iteration, 10) == 0:
            log.info(f'Iteration {s.iteration}: relative residue = {s.residue * 100:0.1f}%, residue = {s.residue * 100:0.1f}%')
        if np.mod(s.iteration, 100) == 0:
            display(s)

        return s.residue > 1e-4 and s.iteration < 1e4

    #
    # Calculate the field produced by the current density source.
    # The actual work is done here.
    #
    start_time = time.perf_counter()
    solution = macromax.solve(grid, vacuum_wavelength=wavelength, current_density=current_density,
                              epsilon=permittivity, callback=update_function, dtype=np.complex64,
                              bound=LinearBound(grid, thickness=boundary_thickness, max_extinction_coefficient=0.5)
                              )

    # Display how the method converged
    times = np.array(times) - start_time
    log.info(f'Calculation time: {times[-1]:0.3f} s.')

    # Calculate total energy flow in the propagation direction
    forward_poynting_vector = np.mean(solution.S, axis=1)  # average over dimension x
    forward_poynting_vector = forward_poynting_vector[1 * vectorial, :]  # Ignore if not vectorial
    forward_poynting_vector_after_layer =\
        forward_poynting_vector[(grid[1].ravel() > layer_thickness / 2) &
                                (grid[1].ravel() < grid[1].ravel()[-1] - boundary_thickness)]
    forward_poynting_vector_after_layer = forward_poynting_vector_after_layer[int(len(forward_poynting_vector_after_layer)/2)]
    log.info('Forward Poynting vector: %g' % forward_poynting_vector_after_layer)
    fig_S = plt.figure(frameon=False, figsize=(12, 9))
    ax_S = fig_S.add_subplot(111)
    ax_S.plot(grid[1].ravel() * 1e6, forward_poynting_vector)
    ax_S.set_xlabel(r'$z [\mu m]$')
    ax_S.set_ylabel(r'$S_z$')

    # Show final result
    log.info('Displaying final result.')
    display(solution)
    plt.show(block=False)
    # Save the individual images
    log.info('Saving results to %s...' % output_filepath.as_posix())
    output_path.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_filepath.as_posix() + '_orientation.png',
               complex2rgb(epsilon_abs * np.exp(1j * orientation), normalization=True, inverted=True),
               vmin=0.0, vmax=1.0, cmap=None, format='png', origin=None, dpi=600)
    for dim_idx in range(solution.E.shape[0]):
        plt.imsave(output_filepath.as_posix() + '_E%s.png' % chr(ord('x') + dim_idx), complex2rgb(solution.E[dim_idx], 1, inverted=True),
                   vmin=0.0, vmax=1.0, cmap=None, format='png', origin=None, dpi=600)
    # Save the figure
    plt.ioff()
    fig.savefig(output_filepath.as_posix() + '.pdf', bbox_inches='tight', format='pdf')
    plt.ion()

    return times, residues, forward_poynting_vector


def generate_birefringent_random_layer(grid, layer_thickness, radius_mean, radius_std=0.0, normal_dim=0,
                                       birefringent=True, medium_refractive_index=1.0):
    random_seed = 0
    rng = np.random.RandomState(seed=random_seed)  # Make sure that this is exactly reproducible

    if birefringent:
        log.info(f'Generating a {layer_thickness / 1e-3:0.3f}μm-thick layer of randomly placed, sized and oriented rutile (TiO2) particles...')
    else:
        log.info(f'Generating a {layer_thickness / 1e-3:0.3f}μm-thick layer of randomly placed and sized particles...')

    layer_grid = Grid(2, extent=(*grid.extent[:normal_dim], layer_thickness, *grid.extent[normal_dim+1:]))
    spheres = pack(layer_grid, radius_mean=radius_mean, radius_std=radius_std, seed=random_seed)  # Make sure that this is exactly reproducible
    grain_radius = np.asarray([_.radius for _ in spheres])
    grain_position = np.asarray([_.position for _ in spheres])

    # If birefringent, pick random crystal axes
    # 1. Generate random rotation matrices
    # 2. Apply diagonal matrices to randomly oriented coordinate system
    # nb_pos = positions.shape[0]
    # nb_dims = positions.shape[1]
    def conj_transpose(a):
        return np.conj(a).swapaxes(-2, -1)

    def orth(A):
        """
        Orthonormalizes the matrices represented in the array A.
        The vectors in the final dimension will all have normal length and be orthogonal to those found along
        the penultimate dimension.
        :param A: An array of which the matrices in the final two dimensions are to be orthonormalized.
        :return: A reference to the same array, now orthonormalized.
        """
        conj_inner = lambda a, b: np.sum(np.conj(a) * b, axis=-1, keepdims=True)
        nb_dims = A.shape[-2]
        for dim_idx in range(nb_dims-1):
            ref = A[..., dim_idx, np.newaxis, :]
            rest = A[..., dim_idx+1:, :]
            projection = ref * (conj_inner(ref, rest) / conj_inner(ref, ref))
            rest -= projection
        # Normalize
        A /= np.linalg.norm(A, axis=-1, keepdims=True)

        return A

    nb_pol = 3
    rot_matrices = orth(rng.normal(0.0, 1.0, [len(spheres), nb_pol, nb_pol]))
    # rot_matrices = np.tile(np.eye(3)[np.newaxis, :, :], [rot_matrices.shape[0], 1, 1])
    # Rutile @500nm:
    medium_permittivity = medium_refractive_index ** 2
    n_o = 2.7114
    n_e = 3.0335
    if birefringent:
        log.debug('Determining a random rotation per rutile grain...')
        eps_eye = np.diag((n_o, n_o, n_e)) ** 2
    else:
        log.debug('Changing extraordinary propagation speed to the ordinary one.')
        eps_eye = np.diag((n_o, n_o, n_o))**2  # Make isotropic
    eps_grain = rot_matrices @ eps_eye @ conj_transpose(rot_matrices)
    grain_direction = rot_matrices[..., 2]

    # Place a matrix at every point in the simulation volume
    log.debug('Rasterizing permittivity tensor...')
    epsilon = np.tile(medium_permittivity * np.eye(nb_pol, dtype=np.complex128)[:, :, np.newaxis, np.newaxis], (1, 1, *grid.shape))
    direction = np.zeros(grid.shape)
    for pos_idx, pos in enumerate(grain_position):
        R2 = (grid[0] - pos[0]) ** 2 + (grid[1] - pos[1]) ** 2
        inside = np.where(R2 < (grain_radius[pos_idx]**2))
        for row_idx in range(nb_pol):
            for col_idx in range(nb_pol):
                epsilon[row_idx, col_idx][inside] = eps_grain[pos_idx][row_idx, col_idx]
                direction[inside] = np.arctan2(grain_direction[pos_idx][1], grain_direction[pos_idx][0])

    return epsilon, direction, grain_position, grain_radius, grain_direction


if __name__ == '__main__':
    start_time = time.perf_counter()
    times, residues, forward_poynting_vector = calculate_and_display_scattering(vectorial=False)  # calc time small 2.9s, large: 23.5s (320 MB)
    times, residues, forward_poynting_vector = calculate_and_display_scattering(anisotropic=False)  # calc time small 11.2s, large: 96.1 (480MB)
    times, residues, forward_poynting_vector = calculate_and_display_scattering(anisotropic=True)  # calc time small 55.9s, large: 198.8s (740MB)
    log.info(f'Total time: {time.perf_counter() - start_time:0.3f}s.')

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
