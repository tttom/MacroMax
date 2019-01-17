#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Example code showing light propagating through a set of polarizers

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import time

import macromax
from macromax import utils
from macromax.examples import log


def show_polarizer(center_polarizer=True):
    wavelength = 500e-9
    angular_frequency = 2 * const.pi * const.c / wavelength
    source_amplitude = 1j * angular_frequency * const.mu_0
    p_source = np.array([0, 1, 1]) / np.sqrt(2)

    # Set the sampling grid
    nb_samples = 2 * 1024
    sample_pitch = wavelength / 8
    boundary_thickness = 5e-6

    x_range = utils.calc_ranges(nb_samples, sample_pitch, np.floor(1 + nb_samples / 2) * sample_pitch - boundary_thickness)

    # define the source
    source = -source_amplitude * sample_pitch * (np.abs(x_range) < sample_pitch/4)  # point source at 0
    source = p_source[:, np.newaxis] * source[np.newaxis, :]

    # define the medium
    eps_pol = np.eye(3, dtype=np.complex64)
    eps_pol[2, 2] = 1.0 + 0.1j
    rot_x = lambda a: np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]], dtype=np.complex64)

    permittivity = np.tile(np.eye(3, dtype=np.complex64)[:, :, np.newaxis], [1, 1, nb_samples])
    for pos_idx, pos in enumerate(x_range):
        if abs(pos - 30e-6) < 5e-6:
            permittivity[:, :, pos_idx] = eps_pol
        elif center_polarizer and abs(pos - 60e-6) < 5e-6:
            permittivity[:, :, pos_idx] = rot_x(-np.pi/4) @ eps_pol @ rot_x(np.pi/4)
        elif abs(pos - 90e-6) < 5e-6:
            permittivity[:, :, pos_idx] = rot_x(-np.pi/2) @ eps_pol @ rot_x(np.pi/2)

    dist_in_boundary = np.maximum(-(x_range - (x_range[0] + boundary_thickness)), x_range - (x_range[-1]-boundary_thickness)) / boundary_thickness
    for dim_idx in range(3):
        permittivity[dim_idx, dim_idx, ...] += (0.8j * np.maximum(0.0, dist_in_boundary))  # absorbing boundary

    # # Impedance matched everywhere
    # permeability = 1.0
    # # Non-impedance matched glass
    # permeability = np.ones((1, 1, len(x_range)), dtype=np.complex64)
    # permeability[:, :, (x_range < -1e-6) | (x_range > 26e-6)] = np.exp(0.2j)  # absorbing boundary
    # permeability[:, :, (x_range >= 10e-6) & (x_range < 20e-6)] = 1.0
    # permeability = bandpass_and_remove_gain(permeability, 2, x_range)
    # No magnetic component
    permeability = 1.0

    # Prepare the display
    fig, axs = plt.subplots(3, 2, frameon=False, figsize=(12, 9))
    abs_line = []
    real_line = []
    imag_line = []
    for plot_idx in range(3):
        field_ax = axs[plot_idx][0]
        abs_line.append(field_ax.plot(x_range * 1e6, x_range * 0, color=[0, 0, 0])[0])
        real_line.append(field_ax.plot(x_range * 1e6, x_range * 0, color=[0, 0.7, 0])[0])
        imag_line.append(field_ax.plot(x_range * 1e6, x_range * 0, color=[1, 0, 0])[0])
        field_ax.set_xlabel("x  [$\mu$m]")
        field_ax.set_ylabel("$I_" + 'xyz'[plot_idx] + "$, $E_" + 'xyz'[plot_idx] + "$  [a.u.]")

        ax_m = axs[plot_idx][1]
        ax_m.plot(x_range[-1] * 2e6, 0, color=[0, 0, 0], label='I')
        ax_m.plot(x_range[-1] * 2e6, 0, color=[0, 0.7, 0], label='$E_{real}$')
        ax_m.plot(x_range[-1] * 2e6, 0, color=[1, 0, 0], label='$E_{imag}$')
        ax_m.plot(x_range * 1e6, permittivity[plot_idx, plot_idx].real, color=[0, 0, 1], label='$\epsilon_{real}$')
        ax_m.plot(x_range * 1e6, permittivity[plot_idx, plot_idx].imag, color=[0, 0.5, 0.5], label='$\epsilon_{imag}$')
        # ax_m.plot(x_range * 1e6, permeability[plot_idx, plot_idx].real, color=[0.5, 0.25, 0], label='$\mu_{real}$')
        # ax_m.plot(x_range * 1e6, permeability[plot_idx, plot_idx].imag, color=[0.5, 1, 0], label='$\mu_{imag}$')
        ax_m.set_xlabel('x  [$\mu$m]')
        ax_m.set_ylabel('$\epsilon$, $\mu$')
        ax_m.set_xlim(x_range[[0, -1]] * 1e6)
        ax_m.legend(loc='upper right')

    plt.ion()

    #
    # Display the (intermediate) result
    #
    def display(s):
        E = s.E
        log.info("Total power: %0.3g", np.linalg.norm(E) ** 2)
        log.info("Displaying it %0.0f: error %0.1f%%" % (s.iteration, 100 * s.residue))

        for plot_idx in range(3):
            ax = axs[plot_idx][0]
            field_to_display = E[plot_idx, :]
            max_val_to_display = np.maximum(np.max(np.abs(field_to_display)), np.finfo(field_to_display.dtype).eps)

            abs_line[plot_idx].set_ydata(np.abs(field_to_display) ** 2 / max_val_to_display)
            real_line[plot_idx].set_ydata(np.real(field_to_display))
            imag_line[plot_idx].set_ydata(np.imag(field_to_display))
            ax.set_ylim(np.array((-1, 1)) * np.maximum(np.max(np.abs(field_to_display)), np.max(abs(field_to_display) ** 2 / max_val_to_display)) * 1.05 )
            figure_title = "$E_" + 'xyz'[plot_idx] + "$ Iteration %d, " % s.iteration
            ax.set_title(figure_title)

        plt.draw()
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
    solution = macromax.solve(x_range, vacuum_wavelength=wavelength, source_distribution=source,
                              epsilon=permittivity, mu=permeability, callback=update_function
                              )

    # Show final result
    log.info('Displaying final result.')
    display(solution)
    plt.show()


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
    if np.isscalar(periods):
        periods = [periods] * len(ranges)
    v_ft = np.fft.fftn(v, axes=dims)
    for dim_idx in range(v.ndim - 2):
        f_range = utils.calc_frequency_ranges(ranges[dim_idx])[0]
        filt = np.exp(-0.5*(np.abs(f_range) * periods[dim_idx])**2)
        v_ft *= utils.to_dim(filt, v_ft.ndim, dims[dim_idx])
    v = np.fft.ifftn(v_ft, axes=dims)
    v[v.imag < 0] = v.real[v.imag < 0]  # remove and gain (may be introduced by rounding errors)
    return v


if __name__ == "__main__":
    start_time = time.time()
    show_polarizer(center_polarizer=False)
    show_polarizer(center_polarizer=True)
    log.debug("Total time: %0.3fs." % (time.time() - start_time))
    plt.show(block=True)
