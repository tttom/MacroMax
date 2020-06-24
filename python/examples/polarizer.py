#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Example code showing light propagating through a set of polarizers

import matplotlib.pyplot as plt
import numpy as np
import time

import macromax
from examples import log


def show_polarizer(center_polarizer=True):
    wavelength = 500e-9
    source_polarization = np.array([0, 1, 1])[:, np.newaxis] / np.sqrt(2)

    # Set the sampling grid
    nb_samples = 2 * 1024
    sample_pitch = wavelength / 8
    boundary_thickness = 5e-6

    x_range = np.arange(nb_samples) * sample_pitch - boundary_thickness

    # define the source
    current_density = source_polarization * (np.abs(x_range) < sample_pitch/4)  # point source at 0

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
        ax_m.plot(x_range[-1] * 2e6, 0, color=[0, 0, 0], label='I')  # Add a dummy line outside the FOV for the legend
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
    solution = macromax.solve(x_range, vacuum_wavelength=wavelength, current_density=current_density,
                              epsilon=permittivity, mu=permeability, callback=update_function
                              )

    # Show final result
    log.info('Displaying final result.')
    display(solution)
    plt.show()


if __name__ == "__main__":
    start_time = time.perf_counter()
    show_polarizer(center_polarizer=False)
    show_polarizer(center_polarizer=True)
    log.debug("Total time: %0.3fs." % (time.perf_counter() - start_time))
    plt.show(block=True)
