#!/usr/bin/env python3
"""
Example code showing reflection at a glass-air interface in one dimension
"""
import matplotlib.pyplot as plt
import numpy as np

import macromax
from macromax import log


def show_air_glass_transition(impedance_matched=False, birefringent=False):
    wavelength = 500e-9
    source_polarization = np.array([0.0, 1.0, 1.0j])[:, np.newaxis]  # y-polarized

    # Set the sampling grid
    nb_samples = 1024
    sample_pitch = wavelength / 16
    x_range = sample_pitch * np.arange(nb_samples) - 5e-6

    # define the source
    current_density = source_polarization * (abs(x_range) < sample_pitch / 2)  # point source at x=0

    # define the medium
    epsilon_material = np.array([1.5, 1.48, 1.5]) ** 2
    has_object = (x_range >= 10e-6) & (x_range < 200e-6)
    permittivity = np.ones(len(x_range), dtype=np.complex64)
    bound = macromax.bound.LinearBound(x_range, thickness=5e-6)  # absorbing boundary

    nb_pol_dims = 1 + 2 * birefringent
    permittivity = np.eye(nb_pol_dims)[:, :, np.newaxis] * permittivity
    for dim_idx in range(nb_pol_dims):
        permittivity[dim_idx, dim_idx, has_object] += epsilon_material[dim_idx]

    permeability = permittivity if impedance_matched else 1

    #
    # Calculate the electric field
    #
    solution = macromax.solve(
        grid=x_range, vacuum_wavelength=wavelength, current_density=current_density,
        refractive_index=permittivity**0.5, mu=permeability, bound=bound, 
        callback=lambda s: s.residue > 1e-5 and s.iteration < 1e4, dtype=np.complex64
    )

    #
    # Show the result now
    #
    E = solution.E[1, :]
    # H = solution.H[2, :]
    S = solution.S[0, :]
    u = solution.energy_density

    field_to_display = E  # The source is polarized along this dimension
    max_val_to_display = np.maximum(np.amax(abs(field_to_display)), np.finfo(field_to_display.dtype).eps)
    poynting_normalization = np.amax(abs(S)) / max_val_to_display
    energy_normalization = np.amax(abs(u)) / max_val_to_display

    # Prepare the display
    fig, ax = plt.subplots(2, 1, frameon=False, figsize=(12, 9), sharex='all')
    ax[0].plot(x_range * 1e6, abs(field_to_display) ** 2 / max_val_to_display, color=[0, 0, 0])[0]
    ax[0].plot(x_range * 1e6, np.real(S) / poynting_normalization, color=[1, 0, 1])[0]
    ax[0].plot(x_range * 1e6, np.real(u) / energy_normalization, color=[0, 1, 1])[0]
    ax[0].plot(x_range * 1e6, np.real(field_to_display), color=[0, 0.7, 0])[0]
    ax[0].plot(x_range * 1e6, np.imag(field_to_display), color=[1, 0, 0])[0]
    ax[0].set_xlabel('x  [$\\mu$m]')
    ax[0].set_ylabel('E, S  [a.u.]')
    ax[0].set_xlim(x_range[[0, -1]] * 1e6)
    ax[0].set_ylim(np.array((-1, 1)) * np.maximum(np.amax(abs(field_to_display)), np.amax(abs(field_to_display) ** 2 / max_val_to_display)) * 1.05)
    ax[1].plot(x_range[-1] * 2e6, 0, color=[0, 0, 0], label='|E|')
    ax[1].plot(x_range[-1] * 2e6, 0, color=[1, 0, 1], label='S')
    ax[1].plot(x_range[-1] * 2e6, 0, color=[0, 1, 1], label='u')
    ax[1].plot(x_range[-1] * 2e6, 0, color=[0, 0.7, 0], label='$E_{real}$')
    ax[1].plot(x_range[-1] * 2e6, 0, color=[1, 0, 0], label='$E_{imag}$')
    ax[1].plot(x_range * 1e6, permittivity[0, 0].real, color=[0, 0, 1], linewidth=2.0, label=r'$\epsilon_{real}$')
    ax[1].plot(x_range * 1e6, permittivity[0, 0].imag, color=[0, 0.5, 0.5], linewidth=2.0, label=r'$\epsilon_{imag}$')
    if impedance_matched:
        ax[1].plot(x_range * 1e6, permeability[0, 0].real, color=[0.5, 0.25, 0], label=r'$\mu_{real}$')
        ax[1].plot(x_range * 1e6, permeability[0, 0].imag, color=[0.5, 1, 0], label=r'$\mu_{imag}$')
    ax[1].set_xlabel(r'x  [$\mu$m]')
    ax[1].set_ylabel(r'$\epsilon$, $\mu$')
    ax[1].set_xlim(x_range[[0, -1]] * 1e6)
    ax[1].legend(loc='upper right')

if __name__ == '__main__':
    show_air_glass_transition(impedance_matched=False, birefringent=False)
    log.info('Displaying final result. Close window to exit.')
    plt.show(block=True)
