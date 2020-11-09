from __future__ import annotations

import numpy as np
from typing import Union, Sequence, Callable

from macromax.utils import log
from macromax.utils.array import Grid


class Electric:
    """ Mixin for Bound to indicate that the electric susceptibility is non-zero."""
    pass


class Magnetic:
    """ Mixin for Bound to indicate that the magnetic susceptibility is non-zero."""
    pass


class Bound:
    """
    A base class to represent calculation-volume-boundaries.
    Use the sub-classes for practical implementations.
    """
    def __init__(self, grid: Union[Grid, Sequence, np.ndarray]=None,
                 thickness: Union[float, Sequence, np.ndarray]=0.0,
                 background_permittivity: float=1.0):
        """
        :param grid: The Grid to which to the boundaries will be applied.
        :param thickness: The thickness as a scalar, vector, or 2d-array (axes x side). Broadcasting is used as necessary.
        :param background_permittivity: The background permittivity of the boundary (default: 1.0 for vacuum). This is
        only used when the absolute permittivity is requested.
        """
        if not isinstance(grid, Grid):
            grid = Grid.from_ranges(grid)
        self.__grid = grid
        self.__thickness = np.broadcast_to(thickness, (self.grid.ndim, 2))
        self.__background_permittivity = background_permittivity

    @property
    def grid(self):
        return self.__grid

    @property
    def thickness(self) -> np.ndarray:
        """
        The thickness as a 2D-array `thickness[axis, front_back]` in meters.
        """
        return self.__thickness.copy()

    @property
    def background_permittivity(self) -> float:
        """A complex scalar indicating the permittivity of the background."""
        return self.__background_permittivity

    @property
    def electric_susceptibility(self) -> np.ndarray:
        """
        The electric susceptibility, chi_E, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        return np.zeros(self.grid.shape)

    @property
    def permittivity(self) -> np.ndarray:
        """
        The electric permittivity, epsilon, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        return self.background_permittivity + self.electric_susceptibility

    @property
    def magnetic_susceptibility(self) -> np.ndarray:
        """
        The magnetic susceptibility, chi_H, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        return np.zeros(self.grid.shape)

    @property
    def permeability(self) -> np.ndarray:
        """
        The magnetic permeability, mu, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        return 1.0 + self.magnetic_susceptibility


class PeriodicBound(Bound):
    def __init__(self, grid: Union[Grid, Sequence, np.ndarray]):
        """
        Constructs an object that represents periodic boundaries.
        
        :param grid: The Grid to which to the boundaries will be applied.
        """
        super().__init__(grid=grid, thickness=0.0)


class AbsorbingBound(Bound, Electric):
    def __init__(self, grid: Union[Grid, Sequence, np.ndarray], thickness: Union[float, Sequence, np.ndarray]=0.0,
                 extinction_coefficient_function: Union[Callable, Sequence, np.ndarray]=lambda rel_depth: rel_depth,
                 background_permittivity: float=1.0):
        """
        Constructs a boundary with depth-dependent extinction coefficient, kappa(rel_depth).

        :param grid: The Grid to which to the boundaries will be applied.
        :param thickness: The boundary thickness(es) in meters. This can be specified as a 2d-array [axis, side].
        Singleton dimensions are broadcast.
        :param extinction_coefficient_function: A function that returns the extinction coefficient as function of
        the depth in the boundary relative to the total thickness of the boundary.
        :param background_permittivity: (default: 1.0 for vacuum)
        """
        super().__init__(grid=grid, thickness=thickness, background_permittivity=background_permittivity)
        self.__extinction_coefficient_functions = np.broadcast_to(extinction_coefficient_function, (self.grid.ndim, 2))

    @property
    def is_electric(self) -> bool:
        return True

    @property
    def extinction(self) -> np.ndarray:
        """
        Determines the extinction coefficient, kappa, of the boundary on a plaid grid.
        The only non-zero values are found in the boundaries. At the corners, the maximum extinction value of the
        overlapping dimensions is returned.

        Note that the returned array may have singleton dimensions that must be broadcast!

        :return: An nd-array with the extinction coefficient, kappa.
        """
        kappa = 0.0
        for axis, rng in enumerate(self.grid):
            for back_side in range(2):
                thickness = self.thickness[axis, back_side] * np.sign(self.grid.step[axis])
                if not back_side:
                    new_depth_in_boundary = (rng.ravel()[0] + thickness) - rng
                else:
                    new_depth_in_boundary = rng - (rng.ravel()[-1] - thickness)
                new_depth_in_boundary *= np.sign(self.grid.step[axis])
                in_boundary = new_depth_in_boundary > 0
                if np.any(in_boundary):
                    rel_depth = in_boundary * new_depth_in_boundary / thickness
                    kappa_function = self.__extinction_coefficient_functions[axis, back_side]
                    kappa = np.maximum(kappa, kappa_function(rel_depth) * in_boundary)
        return kappa

    @property
    def electric_susceptibility(self) -> np.ndarray:
        """
        The electric susceptibility, chi_E, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        n = np.lib.scimath.sqrt(self.background_permittivity)
        epsilon = (n + 1j * self.extinction)**2
        return epsilon - self.background_permittivity


class LinearBound(AbsorbingBound):
    def __init__(self, grid: Union[Grid, Sequence, np.ndarray], thickness: Union[float, Sequence, np.ndarray]=0.0,
                 max_extinction_coefficient: Union[float, Sequence, np.ndarray]=1.0,
                 background_permittivity: float=1.0):
        """
        Constructs a boundary with linearly increasing extinction coefficient, kappa.

        :param grid: The Grid to which to the boundaries will be applied.
        :param thickness: The boundary thickness(es) in meters. This can be specified as a 2d-array [axis, side].
        Singleton dimensions are broadcast.
        :param max_extinction_coefficient: The maximum extinction coefficient, reached at the deepest point of the
        boundary at the edge of the calculation volume.
        :param background_permittivity: (default: 1.0 for vacuum)
        """
        # Define a linear function for every axis and every side
        kappa_function = np.vectorize(lambda kappa_max: lambda rel_depth: kappa_max * rel_depth)\
            (max_extinction_coefficient)
        super().__init__(grid=grid, thickness=thickness,
                         extinction_coefficient_function=kappa_function,
                         background_permittivity=background_permittivity)

