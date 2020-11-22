#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Example code showing reflection at a glass-air interface in one dimension


import matplotlib.pyplot as plt
import numpy as np
import time

import macromax
from macromax import Grid
try:
    from examples import log
except ImportError:
    from macromax import log  # Fallback in case this script is not started as part of the examples package.


def show_air_glass_transition(impedance_matched=False, birefringent=False):
    wavelength = 488e-9
    source_polarization = np.array([0.0, 1.0, 1.0j])[:, np.newaxis]  # y-polarized

    # Set the sampling grid
    nb_samples = 1024
    sample_pitch = wavelength / 16
    x_range = sample_pitch * np.arange(nb_samples) - 5e-6

    # define the source
    current_density = (np.abs(x_range) < sample_pitch / 2)  # point source at 0
    current_density = source_polarization * current_density

    # define the medium
    epsilon_material = np.array([1.5, 1.48, 1.5]) ** 2
    has_object = (x_range >= 10e-6) & (x_range < 200e-6)
    permittivity = np.ones(len(x_range), dtype=np.complex64)
    # absorbing boundary
    boundary_thickness = 5e-6
    bound = macromax.bound.LinearBound(x_range, thickness=boundary_thickness, max_extinction_coefficient=0.25)

    nb_pol_dims = 1 + 2 * birefringent
    permittivity = np.eye(nb_pol_dims)[:, :, np.newaxis] * permittivity
    for dim_idx in range(nb_pol_dims):
        permittivity[dim_idx, dim_idx, has_object] += epsilon_material[dim_idx]
    # permittivity = bandpass_and_remove_gain(permittivity, 2, x_range, wavelength/2)

    if impedance_matched:
        # Impedance matched everywhere
        permeability = permittivity
    else:
        # Non-impedance matched glass
        permeability = 1.0  # The display function below would't expect a scalar

    # Prepare the display
    fig, ax = plt.subplots(2, 1, frameon=False, figsize=(12, 9), sharex='all')
    abs_line = ax[0].plot(x_range * 1e6, x_range * 0, color=[0, 0, 0])[0]
    poynting_line = ax[0].plot(x_range * 1e6, x_range * 0, color=[1, 0, 1])[0]
    energy_line = ax[0].plot(x_range * 1e6, x_range * 0, color=[0, 1, 1])[0]
    real_line = ax[0].plot(x_range * 1e6, x_range * 0, color=[0, 0.7, 0])[0]
    imag_line = ax[0].plot(x_range * 1e6, x_range * 0, color=[1, 0, 0])[0]
    ax[0].set_xlabel("x  [$\mu$m]")
    ax[0].set_ylabel("E, S  [a.u.]")
    ax[0].set_xlim(x_range[[0, -1]] * 1e6)

    ax[1].plot(x_range[-1] * 2e6, 0, color=[0, 0, 0], label='|E|')
    ax[1].plot(x_range[-1] * 2e6, 0, color=[1, 0, 1], label='S')
    ax[1].plot(x_range[-1] * 2e6, 0, color=[0, 1, 1], label='u')
    ax[1].plot(x_range[-1] * 2e6, 0, color=[0, 0.7, 0], label='$E_{real}$')
    ax[1].plot(x_range[-1] * 2e6, 0, color=[1, 0, 0], label='$E_{imag}$')
    ax[1].plot(x_range * 1e6, permittivity[0, 0].real, color=[0, 0, 1], linewidth=2.0, label='$\\epsilon_{real}$')
    ax[1].plot(x_range * 1e6, permittivity[0, 0].imag, color=[0, 0.5, 0.5], linewidth=2.0, label='$\\epsilon_{imag}$')
    if impedance_matched:
        ax[1].plot(x_range * 1e6, permeability[0, 0].real, color=[0.5, 0.25, 0], label='$\\mu_{real}$')
        ax[1].plot(x_range * 1e6, permeability[0, 0].imag, color=[0.5, 1, 0], label='$\\mu_{imag}$')
    ax[1].set_xlabel('x  [$\mu$m]')
    ax[1].set_ylabel('$\epsilon$, $\mu$')
    ax[1].set_xlim(x_range[[0, -1]] * 1e6)
    ax[1].legend(loc='upper right')

    plt.ion()

    def display(s):
        E = s.E[1, :]
        H = s.H[2, :]
        S = s.S[0, :]
        u = s.energy_density

        log.info("1D: Displaying iteration %0.0f: error %0.1f%%" % (s.iteration, 100 * s.residue))
        field_to_display = E  # The source is polarized along this dimension
        max_val_to_display = np.maximum(np.max(np.abs(field_to_display)), np.finfo(field_to_display.dtype).eps)
        poynting_normalization = np.max(np.abs(S)) / max_val_to_display
        energy_normalization = np.max(np.abs(u)) / max_val_to_display

        abs_line.set_ydata(np.abs(field_to_display) ** 2 / max_val_to_display)
        poynting_line.set_ydata(np.real(S) / poynting_normalization)
        energy_line.set_ydata(np.real(u) / energy_normalization)
        real_line.set_ydata(np.real(field_to_display))
        imag_line.set_ydata(np.imag(field_to_display))
        ax[0].set_ylim(np.array((-1, 1)) * np.maximum(np.max(np.abs(field_to_display)), np.max(abs(field_to_display) ** 2 / max_val_to_display)) * 1.05 )
        figure_title = "Iteration %d, " % s.iteration
        ax[0].set_title(figure_title)

        plt.pause(0.001)

    #
    # What to do after each iteration
    #
    def update_function(s):
        if np.mod(s.iteration, 100) == 0:
            log.info("Iteration %0.0f: rms error %0.1f%%" % (s.iteration, 100 * s.residue))
        if np.mod(s.iteration, 100) == 0:
            display(s)

        return s.residue > 1e-5 and s.iteration < 1e4

    # The actual work is done here:
    start_time = time.perf_counter()
    solution = macromax.solve(x_range, vacuum_wavelength=wavelength, current_density=current_density,
                              epsilon=permittivity, mu=permeability, bound=bound, callback=update_function
                              )
    log.info("Calculation time: %0.3fs." % (time.perf_counter() - start_time))

    # Show final result
    log.info('Displaying final result.')
    display(solution)
    plt.show(block=False)


def bandpass_and_remove_gain(v, dims, ranges, periods):
    """
    Helper function to smoothen inputs to the scale of the wavelength
    :param v: the input array.
    :param dims: the dimension to smoothen
    :param ranges: the coordinate vectors of the dimensions to smoothen
    :param periods: the standard deviation(s).
    :return: The smoothened array.
    """
    if np.isscalar(dims):
        dims = [dims]
    if np.isscalar(ranges[0]):
        ranges = [ranges]
    grid = Grid.from_ranges(*ranges)
    if np.isscalar(periods):
        periods = [periods] * len(ranges)
    v_ft = np.fft.fftn(v, axes=dims)
    for dim_idx in range(v.ndim - 2):
        v_ft *= np.exp(-0.5*(np.abs(grid.f[dim_idx]) * periods[dim_idx])**2)
    v = np.fft.ifftn(v_ft, axes=dims)
    v[v.imag < 0] = v.real[v.imag < 0]  # remove and gain (may be introduced by rounding errors)
    return v


if __name__ == "__main__":
    start_time = time.perf_counter()
    show_air_glass_transition(impedance_matched=False, birefringent=False)
    log.info("Total time: %0.3fs." % (time.perf_counter() - start_time))
    plt.show(block=True)
