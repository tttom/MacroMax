#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Example code showing light scattering by a layer of rutile (TiO2) particles.


import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.constants as const
import time

import macromax
from macromax import utils
from macromax.examples import log
from macromax.parallel_ops_column import ParallelOperations


def show_scatterer(vectorial=True, anisotropic=True, scattering_layer=True):
    if not vectorial:
        anisotropic = False

    #
    # Medium settings
    #
    scale = 2
    data_shape = np.array([128, 256]) * scale
    wavelength = 500e-9
    medium_refractive_index = 1.0  # 1.4758, 2.7114
    boundary_thickness = 2e-6
    beam_diameter = 1.0e-6 * scale
    layer_thickness = 2.5e-6 * scale

    k0 = 2 * np.pi / wavelength
    angular_frequency = const.c * k0
    source_amplitude = 1j * angular_frequency * const.mu_0
    sample_pitch = np.array([1, 1]) * wavelength / 15
    ranges = utils.calc_ranges(data_shape, sample_pitch)
    incident_angle = 0 * np.pi / 180

    log.info('Calculating fields over a %0.1fum x %0.1fum area...' % tuple(data_shape * sample_pitch * 1e6))

    def rot_Z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    incident_k = rot_Z(incident_angle) * k0 @ np.array([0, 1, 0])
    p_source = rot_Z(incident_angle) @ np.array([1, 0, 1j]) / np.sqrt(2)
    source = -source_amplitude * np.exp(1j * (incident_k[0]*ranges[0][:, np.newaxis] + incident_k[1]*ranges[1][np.newaxis, :]))
    # Aperture the incoming beam
    source = source * np.exp(-0.5*(np.abs(ranges[1][np.newaxis, :] - (ranges[1][0]+boundary_thickness))
                                   * medium_refractive_index/ wavelength)**2)  # source position
    source = source * np.exp(-0.5*((ranges[0][:, np.newaxis] - ranges[0][int(len(ranges[0])*2/4)])/(beam_diameter/2))**2)  # beam aperture
    source = source[np.newaxis, ...]
    if vectorial:
        source = source * p_source[:, np.newaxis, np.newaxis]

    # Place randomly oriented TiO2 particles
    permittivity, orientation, grain_pos, grain_rad, grain_dir = \
        generate_random_layer(data_shape, sample_pitch, layer_thickness=layer_thickness, grain_mean=1e-6,
                              grain_std=0.2e-6, normal_dim=1,
                              birefringent=anisotropic, medium_refractive_index=medium_refractive_index,
                              scattering_layer=scattering_layer)

    if not anisotropic:
        permittivity = permittivity[:1, :1, ...]
    log.info('Sample ready.')

    # for r, pos in zip(grain_rad, grain_pos):
    #     plot_circle(plt, radius=r*1e6, origin=pos[::-1]*1e6)
    # epsilon_abs = np.abs(permittivity[0, 0]) - 1
    # rgb_image = colors.hsv_to_rgb(np.stack((np.mod(direction / (2*np.pi),1), 1+0*direction, epsilon_abs), axis=2))
    # plt.imshow(rgb_image, zorder=0, extent=utils.ranges2extent(*ranges)*1e6)
    # plt.axis('equal')
    # plt.pause(0.01)
    # plt.show(block=True)

    # Add boundary
    dist_in_boundary = np.maximum(
        np.maximum(0.0, -(ranges[0][:, np.newaxis] - (ranges[0][0]+boundary_thickness)))
        + np.maximum(0.0, ranges[0][:, np.newaxis] - (ranges[0][-1]-boundary_thickness)),
        np.maximum(0.0, -(ranges[1][np.newaxis, :] - (ranges[1][0]+boundary_thickness)))
        + np.maximum(0.0, ranges[1][np.newaxis, :] - (ranges[1][-1]-boundary_thickness))
    )
    weight_boundary = dist_in_boundary / boundary_thickness
    for dim_idx in range(permittivity.shape[0]):
        permittivity[dim_idx, dim_idx, :, :] += -1.0 + (1.0 + 0.5j * weight_boundary)  # boundary

    # Prepare the display
    def add_circles_to_axes(axes):
        for r, pos in zip(grain_rad, grain_pos):
            circle = plt.Circle(pos[::-1]*1e6, r*1e6,
                                edgecolor=np.array((1, 1, 1))*0.0, facecolor=None, alpha=0.25, fill=False, linewidth=1)
            axes.add_artist(circle)

    fig, axs = plt.subplots(3, 2, frameon=False, figsize=(12, 9), sharex=True, sharey=True)
    for ax in axs.ravel():
        ax.set_xlabel('y [$\mu$m]')
        ax.set_ylabel('x [$\mu$m]')
        ax.set_aspect('equal')

    images = [axs[dim_idx][0].imshow(utils.complex2rgb(np.zeros(data_shape), 1, inverted=True),
                                     extent=utils.ranges2extent(*ranges)*1e6)
              for dim_idx in range(3)]

    epsilon_abs = np.abs(permittivity[0, 0]) - 1
    # rgb_image = colors.hsv_to_rgb(np.stack((np.mod(direction / (2*np.pi), 1), 1+0*direction, epsilon_abs), axis=2))
    axs[0][1].imshow(utils.complex2rgb(epsilon_abs * np.exp(1j * orientation), normalization=True, inverted=True),
                     zorder=0, extent=utils.ranges2extent(*ranges)*1e6)
    add_circles_to_axes(axs[0][1])
    axs[1][1].imshow(utils.complex2rgb(permittivity[0, 0], 1, inverted=True), extent=utils.ranges2extent(*ranges) * 1e6)
    axs[2][1].imshow(utils.complex2rgb(source[0], 1, inverted=True), extent=utils.ranges2extent(*ranges) * 1e6)
    axs[0][1].set_title('crystal axis orientation')
    axs[1][1].set_title('$\chi$')
    axs[2][1].set_title('source')

    # Display the medium without the boundaries
    for dim_idx in range(len(axs)):
        for col_idx in range(len(axs[dim_idx])):
            axs[dim_idx][col_idx].set_xlim((ranges[1].flatten()[0] + boundary_thickness) * 1e6,
                                           (ranges[1].flatten()[-1] - boundary_thickness) * 1e6)
            axs[dim_idx][col_idx].set_ylim((ranges[0].flatten()[0] + boundary_thickness) * 1e6,
                                           (ranges[0].flatten()[-1] - boundary_thickness) * 1e6)
            axs[dim_idx][col_idx].autoscale(False)

    #
    # Display the current solution
    #
    def display(s):
        log.info('Displaying iteration %d: error %0.1f%%' % (s.iteration, 100 * s.residue))
        nb_dims = s.E.shape[0]
        for dim_idx in range(nb_dims):
            images[dim_idx].set_data(utils.complex2rgb(s.E[dim_idx], 1, inverted=True))
            figure_title = '$E_' + 'xyz'[dim_idx] + "$ it %d: rms error %0.1f%% " % (s.iteration, 100 * s.residue)
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
        times.append(time.time())
        residues.append(s.residue)

        if np.mod(s.iteration, 10) == 0:
            log.info("Iteration %0.0f: rms error %0.3f%%" % (s.iteration, 100 * s.residue))
        if np.mod(s.iteration, 10) == 1:
            display(s)

        return s.residue > 1e-5 and s.iteration < 1e4

    # The actual work is done here:
    start_time = time.time()
    solution = macromax.solve(ranges, vacuum_wavelength=wavelength, source_distribution=source,
                              epsilon=permittivity, callback=update_function
                              )
    log.info("Calculation time: %0.3fs." % (time.time() - start_time))

    # Display how the method converged
    times = np.array(times) - start_time
    log.info("Calculation time: %0.3fs." % times[-1])

    # Calculate total energy flow in propagation direction
    # forward_poynting_vector = np.sum(solution.S[1, :, :], axis=0)
    forward_E = np.mean(solution.E, axis=1)  # average over dimension x
    forward_H = np.mean(solution.H, axis=1)  # average over dimension x
    forward_poynting_vector = (0.5 / const.mu_0) * ParallelOperations.cross(forward_E, np.conj(forward_H)).real
    forward_poynting_vector = forward_poynting_vector[1, :]
    forward_poynting_vector_after_layer =\
        forward_poynting_vector[(ranges[1] > layer_thickness / 2) & (ranges[1] < ranges[1][-1] - boundary_thickness)]
    forward_poynting_vector_after_layer = forward_poynting_vector_after_layer[int(len(forward_poynting_vector_after_layer)/2)]
    log.info('Forward Poynting vector: %g' % forward_poynting_vector_after_layer)
    fig_S = plt.figure(frameon=False, figsize=(12, 9))
    ax_S = fig_S.add_subplot(111)
    ax_S.plot(ranges[1] * 1e6, forward_poynting_vector)
    ax_S.set_xlabel(r'$z [\mu m]$')
    ax_S.set_ylabel(r'$S_z$')

    # Show final result
    log.info('Displaying final result.')
    display(solution)
    plt.show(block=False)
    # Save the individual images
    log.info('Saving results to folder %s...' % os.getcwd())
    plt.imsave('rutile_orientation.png',
               utils.complex2rgb(epsilon_abs * np.exp(1j * orientation), normalization=True, inverted=True),
               vmin=0.0, vmax=1.0, cmap=None, format='png', origin=None, dpi=600)
    for dim_idx in range(solution.E.shape[0]):
        plt.imsave('rutile_E%s.png' % chr(ord('x') + dim_idx), utils.complex2rgb(solution.E[dim_idx], 1, inverted=True),
                   vmin=0.0, vmax=1.0, cmap=None, format='png', origin=None, dpi=600)
    # Save the figure
    plt.ioff()
    fig.savefig('rutile.svgz', bbox_inches='tight', format='svgz')
    plt.ion()

    return times, residues, forward_poynting_vector


def generate_random_layer(data_shape, sample_pitch, layer_thickness, grain_mean, grain_std=0, normal_dim=0,
                          birefringent=True, medium_refractive_index=1.0, scattering_layer=True):
    rng = np.random.RandomState()
    rng.seed(0)  # Make sure that this is exactly reproducible

    if birefringent:
        log.info('Generating layer of randomly placed, sized and oriented rutile (TiO2) particles...')
    else:
        log.info('Generating layer of randomly placed and sized particles...')

    volume_dims = data_shape * sample_pitch
    layer_dims = np.array([*volume_dims[:normal_dim], layer_thickness, *volume_dims[normal_dim+1:]])

    nb_dims = data_shape.size
    # Choose a set of random positions for the nuclei
    grain_position = np.zeros([0, nb_dims])
    grain_radius = np.zeros(0)
    nb_pos = 0
    failures = 0
    max_energy = (grain_mean / 100) ** 2

    def relax(positions, radii, max_iterations=100):
        relax_idx = 0
        energy = np.inf
        while relax_idx < max_iterations and energy > max_energy:
            relax_idx += 1
            forces, energy = calc_forces_between_spheres(positions, radii, layer_dims)
            positions += forces * (1.0 ** (-relax_idx))
        return energy, positions

    log.debug('Determining random sphere positions and diameters without overlap in the layer...')
    while failures < 10 and scattering_layer:
        nb_pos += 1
        # Insert a new random grain
        grain_position = np.concatenate((grain_position, rng.uniform(-0.5, 0.5, [1, nb_dims]) * layer_dims))
        grain_radius = np.concatenate((grain_radius, rng.normal(grain_mean/2, grain_std/2, 1)))

        energy, grain_position = relax(grain_position, grain_radius)

        if energy > max_energy:
            grain_position = grain_position[:-2, :]
            grain_radius = grain_radius[:-2]
            nb_pos -= 1
            failures += 1
            energy, grain_position = relax(grain_position, grain_radius)

    energy, grain_position = relax(grain_position, grain_radius, max_iterations=1000)

    # Pick random crystal axes
    # 1. Generate random rotation matrices
    # 2. Apply diagonal matrices to randomly oriented coordinate system
    # nb_pos = positions.shape[0]
    # nb_dims = positions.shape[1]
    if birefringent:
        log.debug('Determining a random rotation per rutile grain...')
    nb_pol = 3
    rot_matrices = orth(rng.normal(0.0, 1.0, [nb_pos, nb_pol, nb_pol]))
    # rot_matrices = np.tile(np.eye(3)[np.newaxis, :, :], [rot_matrices.shape[0], 1, 1])
    # Rutile @500nm:
    medium_permittivity = medium_refractive_index ** 2
    n_o = 2.7114
    n_e = 3.0335
    if birefringent:
        eps_eye = np.diag((n_o, n_o, n_e)) ** 2
    else:
        log.debug('Changing extraordinary propagation speed to the ordinary one.')
        eps_eye = np.diag((n_o, n_o, n_o))**2  # Make isotropic
    eps_grain = rot_matrices @ eps_eye @ conj_transpose(rot_matrices)
    grain_direction = rot_matrices[..., 2]

    # Place a matrix at every point in the simulation volume
    log.debug('Rasterizing permittivity tensor...')
    epsilon = np.tile(medium_permittivity * np.eye(nb_pol, dtype=np.complex128)[:, :, np.newaxis, np.newaxis], (1, 1, *data_shape))
    direction = np.zeros(data_shape)
    x_range, y_range = utils.calc_ranges(data_shape, sample_pitch)
    for pos_idx, pos in enumerate(grain_position):
        R2 = (x_range[:, np.newaxis] - pos[0]) ** 2 + (y_range[np.newaxis, :] - pos[1]) ** 2
        inside = np.where(R2 < (grain_radius[pos_idx]**2))
        for row_idx in range(nb_pol):
            for col_idx in range(nb_pol):
                epsilon[row_idx, col_idx][inside] = eps_grain[pos_idx][row_idx, col_idx]
                direction[inside] = np.arctan2(grain_direction[pos_idx][1], grain_direction[pos_idx][0])

    return epsilon, direction, grain_position, grain_radius, grain_direction


def calc_forces_between_spheres(positions, radii, volume_dims):
    nb_pos = positions.shape[0]
    nb_dims = positions.shape[1]
    inner_forces = np.zeros([nb_pos, nb_dims])
    # Add forces between particles
    for pos_idx in range(nb_pos):
        pos = positions[pos_idx]
        vectors = positions - pos
        distances = np.sqrt(np.sum(vectors ** 2, axis=1))
        normals = vectors / (distances + (distances == 0))[:, np.newaxis]
        intersection = np.minimum(0.0, distances - (radii[pos_idx] + radii))
        # intersection[pos_idx] = 0
        inner_forces -= normals * intersection[:, np.newaxis]
    # Add forces from box
    outer_forces = np.zeros([nb_pos, nb_dims])
    for dim_idx in range(nb_dims):
        dist_beyond_border = np.minimum(0.0, positions[:, dim_idx] - radii - -(volume_dims[dim_idx] / 2)) \
                             + np.maximum(0.0, positions[:, dim_idx] + radii - (volume_dims[dim_idx] / 2))
        direction = np.zeros(nb_dims)
        direction[dim_idx] = -1
        outer_forces += direction * dist_beyond_border[:, np.newaxis]

    forces = inner_forces + outer_forces

    inner_energy = np.sum(inner_forces.flatten() ** 2)
    outer_energy = np.sum(outer_forces.flatten() ** 2)
    energy = inner_energy + outer_energy

    return forces, energy


def conj_transpose(a):
    return np.conj(a).swapaxes(-2, -1)


def conj_inner(a, b):
    return np.sum(np.conj(a) * b, axis=-1, keepdims=True)


def orth(A):
    """
    Orthonormalizes the matrices represented in the array A.
    The vectors in the final dimension will all have normal length and be orthogonal to those found along
    the penultimate dimension.
    :param A: An array of which the matrices in the final two dimensions are to be orthonormalized.
    :return: A reference to the same array, now orthonormalized.
    """
    nb_dims = A.shape[-2]
    for dim_idx in range(nb_dims-1):
        ref = A[..., dim_idx, np.newaxis, :]
        rest = A[..., dim_idx+1:, :]
        projection = ref * (conj_inner(ref, rest) / conj_inner(ref, ref))
        rest -= projection
    # Normalize
    A /= np.linalg.norm(A, axis=-1, keepdims=True)

    return A


def plot_circle(ax, radius=1, origin=(0, 0), nb_segments=40):
    thetas = 2 * np.pi * np.mod(np.arange(nb_segments+1) / nb_segments, 1.0)
    return ax.plot(origin[0] + radius * np.cos(thetas), origin[1] + radius * np.sin(thetas))


if __name__ == "__main__":
    start_time = time.time()
    # times, residues = show_scatterer(vectorial=False)  # calc time small 2.9s, large: 23.5s (320 MB)
    # times, residues = show_scatterer(anisotropic=False)  # calc time small 11.2s, large: 96.1 (480MB)
    times, residues, forward_poynting_vector = show_scatterer(anisotropic=True, scattering_layer=True)  # calc time small 55.9s, large: 198.8s (740MB)
    log.info("Total time: %0.3fs." % (time.time() - start_time))

    # Display how the method converged
    fig_summary, axs_summary = plt.subplots(1, 2, frameon=False, figsize=(18, 9))
    axs_summary[0].semilogy(times, residues)
    axs_summary[0].scatter(times[::100], residues[::100])
    axs_summary[0].set_xlabel('t [s]')
    axs_summary[0].set_ylabel(r'$||\Delta E|| / ||E||$')
    colormap_ranges = [-(np.arange(256) / 256 * 2 * np.pi - np.pi), np.linspace(0, 1, 256)]
    colormap_image = utils.complex2rgb(
        colormap_ranges[1][np.newaxis, :] * np.exp(1j * colormap_ranges[0][:, np.newaxis]),
        inverted=True)
    axs_summary[1].imshow(colormap_image, extent=utils.ranges2extent(*colormap_ranges))

    plt.show(block=True)
