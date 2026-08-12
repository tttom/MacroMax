#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Example code showing light scattering by a layer of rutile (TiO2) particles.
from __future__ import annotations

import collections
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import time
from typing import Sequence

import macromax
from macromax.bound import LinearBound
from macromax.utils.ft import Grid
from macromax.utils.display import complex2rgb, grid2extent
from macromax import log


class Sphere:
    def __init__(self, radius: float = 1.0, position: np.ndarray = np.zeros(2), neighbors: list = None, fails: int = 0):
        self.radius = radius  # The radius
        self.position = np.asarray(position).ravel()  # The center position
        self.__neighbors = collections.OrderedDict()
        self.__sorted_neighbors = None
        self.add_neighbors(neighbors)
        self.fails = fails

    @property
    def neighbors(self) -> list:
        if self.__sorted_neighbors is None:
            self.__sorted_neighbors = [_[0] for _ in sorted(self.__neighbors.items(), key=lambda kv: kv[1])]
        return self.__sorted_neighbors

    def add_neighbors(self, new_neighbors):
        if new_neighbors is not None:
            if isinstance(new_neighbors, Sphere):
                new_neighbors = [new_neighbors]
            for neighbor in new_neighbors:
                self.__neighbors[neighbor] = self.center_distance(neighbor)
            self.__sorted_neighbors = None

    @property
    def neighbors_of_neighbors(self) -> list:
        """Excludes this sphere and its neighborhood!"""
        n_of_n = self.neighbors.copy()
        [n_of_n.__iadd__(_.neighbors) for _ in self.neighbors]
        n_of_n = set(n_of_n)
        n_of_n -= set(self.neighbors)
        n_of_n -= {self}
        n_of_n = sorted(n_of_n, key = lambda _: _.center_distance(self))
        return n_of_n

    def center_distance(self, other_sphere: Sphere) -> float:
        if other_sphere in self.__neighbors:
            return self.__neighbors[other_sphere]
        else:
            return np.linalg.norm(self.position - other_sphere.position)

    def shell_distance(self, other_sphere: Sphere) -> float:
        return self.center_distance(other_sphere) - (self.radius + other_sphere.radius)

    def overlap(self, other_sphere: Sphere) -> bool:
        return self.shell_distance(other_sphere) < 0

    def __repr__(self) -> str:
        return f'Sphere({self.radius},{self.position.tolist()},{self.neighbors},{self.fails})'

    def __str__(self) -> str:
        return f'Sphere({self.radius},{self.position.tolist()})'

    def __hash__(self) -> int:
        """ Only hashes the radius and position! """
        return hash(self.position.tobytes())


def pack(grid: Grid, radius_mean: float = 1.0, radius_std: float = 0.0, seed: int | None = None) -> Sequence[Sphere]:
    rng = np.random.Generator(np.random.PCG64(seed=seed))  # Set seed to make sure that this is reproducible

    layer_extent = grid.extent

    # radii = rng.normal(radius_mean, radius_std, nb_spheres)
    # positions = grid.first + rng.uniform(0.0, 1.0, [nb_spheres, grid.ndim]) * (layer_extent - radii)

    neighborhood_radius = 3 * (radius_mean + 5 * radius_std)

    # Start with random sphere at a random place in the volume
    radius = rng.normal(radius_mean, radius_std)
    spheres = [Sphere(radius=radius, position=grid.first + radius + rng.uniform(0.0, 1.0, [1, grid.ndim]) * (layer_extent - 2 * radius))]

    # Create a new sphere for the first iteration
    new_sphere = Sphere(radius=rng.normal(radius_mean, radius_std))
    while min(_.fails for _ in spheres) < 2 * (5 ** (grid.ndim - 1)):
        # Pick a random radius
        # new_sphere.radius = rng.normal(radius_mean, radius_std)
        # Pick a potential neighbor
        # log.info('Picking a potential neighbor.')
        contact_sphere = min(spheres, key=lambda _: _.fails)
        for trial_idx in range(2 * (5 ** (grid.ndim - 1))):
            # Place sphere at random position but touching this sphere
            random_direction = rng.normal(0.0, 1.0, grid.ndim)
            random_direction /= np.linalg.norm(random_direction)
            new_sphere.position = contact_sphere.position + random_direction * (contact_sphere.radius + new_sphere.radius)
            # Check if inside box
            if np.all(grid.first <= new_sphere.position - new_sphere.radius) \
                    and np.all(new_sphere.position + new_sphere.radius < grid.first + grid.extent):
                # Check for overlap with known neighbors
                neighbor_overlap = any(_.overlap(new_sphere) for _ in contact_sphere.neighbors)
                if not neighbor_overlap:
                    # log.info('No overlap with neighbor.')
                    if contact_sphere.radius + 2 * new_sphere.radius < neighborhood_radius:
                        other_spheres = []
                    else:
                        # Check with all other spheres
                        other_spheres = [_ for _ in spheres if _ is not contact_sphere and _ not in contact_sphere.neighbors]
                    other_overlap = any(new_sphere.overlap(_) for _ in other_spheres)
                    if not other_overlap:
                        # log.info('No overlap with other either.')
                        # All good! Now add the sphere to the set of spheres!
                        nearby_spheres = [_ for _ in spheres if new_sphere.center_distance(_) - _.radius < neighborhood_radius]
                        new_sphere.add_neighbors(nearby_spheres)  # Add reference back to contact sphere as well as all spheres in neighborhood
                        for _ in nearby_spheres:
                            _.add_neighbors(new_sphere)  # Add reference to new sphere for all neighbors
                        spheres.append(new_sphere)  # Add the new sphere to the list
                        if len(spheres) % 100 == 0:
                            log.info(f'Packed {len(spheres)} spheres so far.')

                        # Create a new sphere for the next iteration
                        new_sphere = Sphere(radius=rng.normal(radius_mean, radius_std))
                        continue  # Try to add another one

            # Some overlap was detected somewhere
            contact_sphere.fails += 1

    return spheres


def calculate_and_display_scattering(vectorial=True, anisotropic=True):
    if not vectorial:
        anisotropic = False

    output_path = pathlib.Path('results').absolute()
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
