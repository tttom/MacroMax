#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Utility class and functions to pack spheres, rasterize, and display.
# This can be used to create random materials.
#

from __future__ import annotations

from typing import Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np
import time
import pathlib
import collections
import logging

from macromax.utils.ft import Grid
from macromax.utils.display import complex2rgb, grid2extent

log = logging.getLogger(__file__)


__all__ = ['pack', 'rasterize', 'pack_and_rasterize', 'draw_spheres', 'Sphere']


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


def pack(grid: Grid, radius_mean: float = 1.0, radius_std: float = 0.0, seed: Optional[int] = None) -> Sequence[Sphere]:
    rng = np.random.Generator(np.random.PCG64(seed=seed))  # Set seed to make sure that this is reproducible

    layer_extent = grid.extent

    # radii = rng.normal(radius_mean, radius_std, nb_spheres)
    # positions = grid.first + rng.uniform(0.0, 1.0, [nb_spheres, grid.ndim]) * (layer_extent - radii)

    neighborhood_radius = 3 * (radius_mean + 5 * radius_std)

    # Start with random sphere at a random place in the volume
    radius = rng.normal(radius_mean, radius_std)
    spheres = [Sphere(radius=radius, position=grid.first + radius + rng.uniform(0.0, 1.0, [1, grid.ndim]) * (layer_extent - 2 * radius))]

    # Create a new sphere for the first iteration
    new_sphere = Sphere(radius = rng.normal(radius_mean, radius_std))
    while min(_.fails for _ in spheres) < 2 * (5 ** (grid.ndim - 1)):
        # Pick a random radius
        # new_sphere.radius = rng.normal(radius_mean, radius_std)
        # Pick a potential neighbor
        # log.info('Picking a potential neighbor.')
        contact_sphere = min(spheres, key = lambda _: _.fails)
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


def rasterize(grid, spheres) -> np.ndarray:
    img = np.zeros(grid.shape)
    for _ in spheres:
        R2 = sum((rng - coord) ** 2 for rng, coord in zip(grid, _.position))
        inside = np.where(R2 < _.radius ** 2)
        img[inside] += 1

    log.info(f'Done rasterizing {len(spheres)} spheres.')
    return img


def pack_and_rasterize(grid, radius_mean: float = 1.0, radius_std: float = 0.0) -> np.ndarray:
    return rasterize(grid, pack(grid, radius_mean, radius_std))


def draw_spheres(ax, spheres: Sequence[Sphere], scale: float = 1.0):
    for _ in spheres:
        ax.add_artist(plt.Circle(_.position[::-1] / scale, _.radius / scale, facecolor=[1, 1, 1, 0], edgecolor=[1, 0, 0, 0.75], linewidth=2))
    return ax


if __name__ == "__main__":
    radius_mean = 1.0e-6 / 2
    radius_std = 0.25e-6 / 2

    grid = Grid(extent=(10e-6, 10e-6), step=0.500e-6 / 4)

    # import pprofile
    log.info('Calculating sphere stacking...')
    start_time = time.perf_counter()
    spheres = pack(grid, radius_mean, radius_std)
    packing_time = time.perf_counter() - start_time
    log.info(f'Packing time: {packing_time:0.3f}s for {len(spheres)} spheres.')
    start_time = time.perf_counter()
    img = rasterize(grid, spheres)
    rasterizing_time = time.perf_counter() - start_time
    log.info(f'Rasterization time: {rasterizing_time:0.3f}s for {len(spheres)} spheres.')

    # Display how the method converged
    log.info('Displaying...')
    fig_summary, axs = plt.subplots(1, 1, frameon=False, figsize=(12, 9))
    start_time = time.perf_counter()
    axs.imshow(complex2rgb(img, normalization=1.0), extent=grid2extent(grid / 1e-6))
    draw_spheres(axs, spheres, 1e-6)
    log.info(f'Total drawing time: {time.perf_counter() - start_time:0.3f}s for {len(spheres)} spheres.')

    # Save the results
    output_path = pathlib.Path('output').absolute()
    output_filepath = pathlib.PurePath(output_path, f'packed_spheres_{radius_mean*2*1e9:0.0f}nm_{"x".join(str(_) for _ in grid.shape)}')
    log.info(f'Saving results to {output_filepath.as_posix()}...')
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(output_filepath, positions=np.array([_.position for _ in spheres]), radii=np.array([_.radius for _ in spheres]), image=img,
             radius_mean=radius_mean, radius_std=radius_std,
             grid_shape=grid.shape, grid_step=grid.step,
             packing_time=packing_time, rasterizing_time=rasterizing_time)
    log.info(f'Saved results to {output_filepath.as_posix()}.')

    plt.show(block=True)
