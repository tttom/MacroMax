#
# Definitions of neural networks that can solve the wave equations for scattering in a heterogeneous material,
# and their comparison for a simple system.
#
# This is a simplified implementation that ignores polarization, birefringence, and magnetic properties. It uses PyTorch
# on the CPU. A more complete and optimized implementation is integrated in the macromax library.
#
# The main code at the bottom of this file defines a material structure and a source. It determines the electric field
# (a) by training a neural network that represents the preconditioned helmholtz equation,
# (b) infering it using a recurrent neural network that takes the source current density as input.
#
# == Key classes ==
#
# * HelmholtzNet: A neural network that infers the source current density for a given electric field.
#   This is useful to verify an electric field computed by various methods. This neural network only has one set of
#   parameters that can be trained, the connection weights that represent the electric field. These are accessed as:
#   ```
#     model = HelmholtzNet(permittivity, grid, k0)
#     electric_field = model.weight
#     model.weight.data = electric_field
#   ```
#   The optimizer iteratively updated the model network's weight.  The response of the model can be read out as:
#   ```
#     source_current_density = model(1)
#   ```
#   For inference, the weights can be left at the default value of 1 and the source current density can be determined as
#   ```
#     source_current_density = model(electric_field)
#   ```
#
# * PreconditioningNet: A neural network that preconditions the network it is appended to.
#
# * PreconditionedHelmholtzNet: The preconditioned HelmholtzNet infers the preconditioned current density. Unlike, the
#   original HelmholtzNet, this network can be trained in a reasonable time.
#
# * InverseHelmholtzNet: A recurrent neural network that repeatedly takes the source current density and infers the
#   electric field.
#
# The script at the bottom of this file runs the training of the PreconditionedHelmholtzNet and the inference with
#   InverseHelmholtzNet in parallel. The latter can be seen to converge in a fraction of the time.
#

import torch
from torch import nn
import numpy as np
import scipy.constants as const
from typing import Union, Optional, Callable
import matplotlib.pyplot as plt
import pathlib

from macromax.bound import LinearBound
from macromax.utils import Grid
from macromax.utils.display import complex2rgb, grid2extent

try:
    from examples import log
except ImportError:
    from macromax import log  # Fallback in case this script is not started as part of the examples package.log.setLevel(logging.INFO)

array_like = Union[np.ndarray, torch.Tensor]

display_progress = True  # Enable / disable output figures

device = 'cuda' if torch.has_cuda else 'cpu'
dtype = torch.complex128  # complex64  often sufficient

torch.manual_seed(0)  # Fix for reproducibility


#
# First, define some useful generic neural network layers
#

class DirectLayer(nn.Module):
    """A Simple layer with the same number of inputs as outputs and single connections between neurons."""
    def __init__(self, weight: Union[complex, array_like]):
        """
        The weights of the direct connections, default: all 1.

        :param weight: Unless specified as a `torch.nn.Parameter` object, these weights are not trainable.
        """
        super().__init__()
        if isinstance(weight, torch.nn.Parameter):
            self.__weight = weight
        elif isinstance(weight, torch.Tensor):
            self.__weight = weight.detach().clone()
        else:
            self.__weight = torch.from_numpy(weight)
        self.__weight = self.__weight.to(dtype=dtype, device=device)

    @property
    def weight(self) -> torch.Tensor:
        return self.__weight

    def forward(self, _):
        """Return the weighted inputs."""
        return self.weight * _


class ConvFFTLayer(DirectLayer):
    """A convolutional neural network layer based on the cyclic fast-Fourier transform."""
    def __init__(self, filter_ft):
        """
        Constructs an FFT-based convolutional neural network layer.

        :param filter_ft: The Fourier transform of the convolution kernel.
        """
        super().__init__(filter_ft)

    def forward(self, _):
        """Returns the convolved input."""
        return torch.fft.ifftn(self.weight * torch.fft.fftn(_))


class DeconvFFTLayer(ConvFFTLayer):
    """A convolutional neural network layer that inverts the action of another."""
    def __init__(self, filter_ft):
        """
        Creates a convolutional neural network layer that inverts the action of `ConvFFTLayer(filter_ft)`.

        :param filter_ft: The fourier transform of the convolution kernel that needs to be inverted.
        """
        super().__init__(1 / filter_ft)


class WaveModuleBase(nn.Module):
    """A base class to help implement neural networks that solve wave equations."""
    def __init__(self, grid: Grid, k0: float = 1.0):
        """
        A class to provide common properties for neural networks that represent wave equations.

        :param grid: The Cartesian grid at which the permittivity is defined and the fields are calculated.
        :param k0: The vacuum wavenumber k0 = 2 pi / wavelength.
        """
        nn.Module.__init__(self)
        self.__grid = grid
        self.__rel_forward_factor = -k0 / (1j * const.c * const.mu_0)  # The factor converting from the unitless normalized wave equation to Maxwell's equations that translate electric fields to current density.
        self.__k2_rel = None
        self.weight = torch.nn.Parameter(torch.ones(size=tuple(grid.shape), dtype=dtype))  # To be trained

    @property
    def grid(self) -> Grid:
        """The Cartesian grid at which the permittivity is defined and the fields are calculated."""
        return self.__grid

    @property
    def rel_forward_factor(self) -> complex:
        """The common complex factor to translate from electric fields to electric currents."""
        return self.__rel_forward_factor

    @property
    def k2_rel(self) -> torch.Tensor:
        """The squared relative wavenumber (k/k0) at every point of the spatial frequency grid."""
        return sum((torch.from_numpy(_).to(dtype=dtype, device=device) / k0) ** 2 for _ in grid.k)


class HelmholtzNet(WaveModuleBase):
    """The Helmholz equation as a neural network."""
    def __init__(self, permittivity: array_like, grid: Grid = None, k0: float = 1.0):
        if grid is None:
            grid = Grid(permittivity.shape)
        super().__init__(grid, k0)  # Defines properties to access the electric field, the Cartesian calculation grid, k2_rel, and j_from_e
        self.__laplacian_layer = ConvFFTLayer(- self.k2_rel * self.rel_forward_factor)
        self.__permittivity_layer = DirectLayer(permittivity * self.rel_forward_factor)

    def forward(self, _):  # Input should be 1
        """Returns the outputs of the neural network in its current state for an input of all 1."""
        electric_field = self.weight * _
        return self.__laplacian_layer(electric_field) + self.__permittivity_layer(electric_field)


class ModifiedWaveModuleBase(WaveModuleBase):
    def __init__(self, permittivity: array_like, grid: Grid = None, k0: float = 1.0):
        """
        A base class to help modify a Helmholtz problem. It provides the modified Green's function and the modified
        potential as neural network layers. The modification consists in a shifting and scaling so that a simple Born
        series / power / Neumann iteration converges.

        :param permittivity: The relative permittivity (n^2) distribution at the points defined by the sample grid.
        :param grid: The Cartesian grid at which the potential is sampled.
        :param k0: The vacuum wavenumber, k0 = 2 pi / wavelength.
        """
        if isinstance(permittivity, torch.Tensor):
            permittivity = permittivity.detach().clone()
        else:
            permittivity = torch.from_numpy(permittivity)
        permittivity = permittivity.to(dtype=dtype, device=device)
        if grid is None:
            grid = Grid(permittivity.shape)
        super().__init__(grid, k0)  # Defines properties to access the electric field, the Cartesian calculation grid, k2_rel, and j_from_e

        # Shift and normalize potential and Green's function
        permittivity_0 = (torch.amin(permittivity.real) + torch.amax(permittivity.real)) / 2.0  # Choose a value so that the scaling factor is small and convergence fast
        problem_scale = 1.0j * self.rel_forward_factor * torch.amax(torch.abs(permittivity - permittivity_0))  # The complex 'scale' of the problem, used by the preconditioner
        v_diag = (permittivity - permittivity_0) * (self.rel_forward_factor / problem_scale)  # must be a contraction (operation that shrinks) and with non-negative real part
        l_ft_diag = (- self.k2_rel + permittivity_0) * (self.rel_forward_factor / problem_scale)  # must have spectrum with non-negative real part

        self.__modified_greens_function = DeconvFFTLayer(l_ft_diag + 1)  # G' = (L + 1)^{-1} = (M / scaling_factor - V')^{-1}
        self.__modified_potential = DirectLayer(v_diag - 1)  # V' = V - 1 = (epsilon - epsilon_0) j_from_e / scaling_factor - 1
        self.__problem_scale = problem_scale.detach().numpy()

    @property
    def modified_greens_function_layer(self) -> torch.nn.Module:
        """The modified green's function (L + 1)^{-1} = (laplacian/k0^2 + epsilon_0) / scaling_factor + 1."""
        return self.__modified_greens_function

    @property
    def modified_potential_layer(self) -> torch.nn.Module:
        """The modified potential V - 1 = (epsilon - epsilon_0) / scaling_factor - 1."""
        return self.__modified_potential

    @property
    def problem_scale(self) -> complex:
        """
        The complex scaling factor that was used to scale down the problem so that the potential ||V|| < 1 and so that
        it is accretive. This must be used by the preconditioner.
        """
        return self.__problem_scale


class PreconditioningNet(ModifiedWaveModuleBase):
    """A neural network that can be used as preconditioner when appended to another neural network."""
    def __init__(self, permittivity: array_like, grid: Grid = None, k0: float = 1.0):
        super().__init__(permittivity, grid, k0)

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        return -1 / self.problem_scale * self.modified_potential_layer(self.modified_greens_function_layer(_))  # - 1/scaling_factor V' G'


class PreconditionedHelmholtzNet(ModifiedWaveModuleBase):
    """
    The precondioned Helmholz equation as a neural network.

    This is a more efficient implementation than the equivalent class:
    class PreconditionedHelmholtzNet(nn.Sequential):
        def __init__(self, permittivity: array_like, grid: Grid = None, k0: float = 1.0):
            super().__init__(HelmholtzNet(permittivity, grid, k0),
                             PreconditioningNet(permittivity, grid, k0))

    """
    def __init__(self, permittivity: array_like, grid: Grid = None, k0: float = 1.0):
        super().__init__(permittivity, grid, k0)

    def forward(self, _):  # Input is expected to be 1
        """Returns the outputs of the neural network in its current state for an input of all 1."""
        electric_field = self.weight * _  # E
        layer_3_output = self.modified_greens_function_layer(self.modified_potential_layer(electric_field)) + electric_field  # (G' V' + 1) E
        return -self.modified_potential_layer(layer_3_output)  # - V' (G' V' + 1) E


class InverseHelmholtzNet(ModifiedWaveModuleBase):
    """
    A recurrent neural network with memory of its output.

    Takes as input the current density source and outputs increasingly close estimates of the electric field.
    """
    def __init__(self, permittivity: array_like, grid: Grid = None, k0: float = 1.0):
        super().__init__(permittivity, grid, k0)
        self.output = torch.zeros(size=permittivity.shape, dtype=dtype)  # E (from the last step)
        self.update = None  # dE (the last update)

    def forward(self, _):  # Input is expected to be 1
        """
        Returns increasingly close approximations of the electric field for a given input current density.

            ```E_{n+1} = E_n + V' [G' V' + 1] E_n + j / problem_scale```,
            where ```V' := V - 1 = (\\epsilon - \\epsilon_0) * j_from_e / problem_scale - 1``` and
            ```G' := (L + 1)^{-1} = (M / problem_scale - V')^{-1}```.
        """
        self.update = self.modified_potential_layer(
            self.modified_greens_function_layer(
                self.modified_potential_layer(self.output) - _ / self.problem_scale
            ) + self.output
        )  # V' [G' (V' E - j/s) + E]  (no subtractions)
        self.output += 0.8 * self.update  # E += V' [G' (V' E - j/s) + E]  # todo: Include alpha = 0.80 factor here? It reduces the residue by 1/3 to 1/2 but it is more complicated.
        return self.output


if __name__ == '__main__':
    output_filepath = pathlib.PurePath(pathlib.Path('output').absolute(), 'convergence.pdf')

    log.info('Defining the optical system...')
    wavelength = 500e-9
    boundary_thickness = 1e-6
    k0 = 2 * np.pi / wavelength
    data_shape = np.full(2, 64)
    sample_pitch = wavelength / 4
    grid = Grid(data_shape, sample_pitch)
    beam_diameter = grid.extent[0] / 4

    # Define source
    source_j = np.exp(1j * k0 * grid[1])  # propagate along axis 1
    # Aperture the incoming beam
    source_j = source_j * np.exp(-0.5 * (np.abs(grid[1] - (grid[1].ravel()[0] + boundary_thickness)) / wavelength) ** 2)
    source_j = source_j * np.exp(-0.5 * ((grid[0] - grid[0].ravel()[int(len(grid[0]) * 1 / 2)]) / (beam_diameter / 2)) ** 2)

    # Define material with scatterer
    sphere_mask = np.sqrt(sum(_**2 for _ in grid)) < 0.5 * np.amin(data_shape * sample_pitch) / 2
    permittivity = np.ones(data_shape)
    permittivity[sphere_mask] = 1.5 ** 2

    # Add absorbing boundary
    bound = LinearBound(grid, thickness=boundary_thickness, max_extinction_coefficient=0.3)
    permittivity = permittivity + bound.electric_susceptibility

    # Convert the source to a PyTorch Tensor
    source_j = torch.from_numpy(source_j).to(dtype=dtype, device=device)
    source_norm = torch.linalg.norm(source_j)

    # The Helmholtz problem as a neural network can be used to check if the electric field is correct for a given current density.
    forward = HelmholtzNet(permittivity, grid, k0)

    def verify_result(computed_field: torch.Tensor) -> float:
        """A function to check how close both sides of the Helmholtz equation are in relative terms."""
        with torch.no_grad():
            if isinstance(computed_field, torch.nn.Parameter):
                computed_field = computed_field.data
            forward.weight.data = computed_field
            return torch.linalg.norm(forward(1) - source_j) / source_norm

    #
    # Solve by training a neural network
    #
    log.info('Calculating the electric field by training the preconditioned Helmholtz neural network...')
    preconditioner_inv = PreconditioningNet(permittivity, grid, k0)
    preconditioned_source = preconditioner_inv(source_j)
    preconditioned_source_ms = torch.mean(torch.abs(preconditioned_source) ** 2)
    preconditioned_forward = PreconditionedHelmholtzNet(permittivity, grid, k0)

    # training_model = HelmholtzNet(permittivity, grid, k0)
    # training_target_output = source
    training_model = preconditioned_forward
    training_target_output = preconditioned_source

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.0)
    optimizer = torch.optim.Adam(training_model.parameters(), eps=1e-6)

    loss_function = torch.nn.MSELoss()  # Can only handle real values
    # Define some lists to store the errors at each epoch-iteration
    prec_training_errors = []
    prec_inference_errors = []
    true_training_errors = []
    true_inference_errors = []
    true_input_training_errors = []
    true_input_inference_errors = []

    def calc_error_and_grad():
        training_model.zero_grad()  # Reset gradient calculations
        # Calculate the cost (and build the calculation graph used for the gradient calculation)
        output = training_model(1)
        error2 = loss_function(output.real, training_target_output.real) + loss_function(output.imag, training_target_output.imag)
        error2.backward()  # Work backwards through the graph to determine the error gradient directly as a function of the model's parameters
        # Keep intermediate results for reporting
        prec_training_errors.append((error2.detach().numpy() / preconditioned_source_ms) ** 0.5)
        return error2

    log.info('Calculating the solution once to maximum precision, so we can track progress...')
    inverse_reference = InverseHelmholtzNet(permittivity, grid, k0)
    with torch.no_grad():
        prec_inference_error = torch.inf
        while prec_inference_error > 3 * torch.finfo(dtype).eps:
            reference_solution = inverse_reference(source_j)
            prec_inference_error = torch.linalg.norm(inverse_reference.update) / torch.linalg.norm(reference_solution)

    log.info('Calculating the electric field using inference with the recurrent neural network...')
    inverse = InverseHelmholtzNet(permittivity, grid, k0)

    if display_progress:
        log.info('Displaying...')
        fig, axs = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(12, 6))

    training_model.weight.data *= 0.0  # Start with all 0 field
    for _ in range(3000):
        # One training step
        optimizer.step(calc_error_and_grad)

        display_current_iteration = display_progress or _ % 1000 == 0

        with torch.no_grad():
            # One inference step
            inferred_field = inverse(source_j)

            if display_progress or _ % 1000 == 0:
                # calculate and store the relative errors for display
                prec_inference_error = torch.linalg.norm(inverse.update) / torch.linalg.norm(inferred_field)
                true_output_training_error, true_output_inference_error = (verify_result(_) for _ in (training_model.weight.data, inferred_field))
                prec_inference_errors.append(prec_inference_error)
                true_training_errors.append(true_output_training_error)
                true_inference_errors.append(true_output_inference_error)
                true_input_training_error = torch.linalg.norm(training_model.weight - reference_solution) / torch.linalg.norm(reference_solution)
                true_input_inference_error = torch.linalg.norm(inverse.output - reference_solution) / torch.linalg.norm(reference_solution)
                true_input_training_errors.append(true_input_training_error)
                true_input_inference_errors.append(true_input_inference_error)

            # Display
            if _ % 1000 == 0:
                log.info(f'Iteration {_} errors: training {prec_training_errors[-1]:0.12f} (true: {true_input_training_error:0.12f}), inference {prec_inference_error:0.12f} (true: {true_input_inference_error:0.12f}).')
                if display_progress:
                    axs[0].imshow(complex2rgb(training_model.weight, normalization=1),
                                  extent=grid2extent(grid) * 1e6)
                    axs[0].set(title=f'{_}: training prec-$\\epsilon_j = {prec_training_errors[-1]:0.6f}$, true-$\\epsilon_E = {true_input_training_error:0.6f}$')
                    axs[1].imshow(complex2rgb(inferred_field, normalization=1), extent=grid2extent(grid) * 1e6)
                    axs[1].set(title=f'{_}: inference prec-$\\epsilon_E = {prec_inference_error:0.6f}$, true-$\\epsilon_E = {true_input_inference_error:0.6f}$')
                    plt.show(block=False)
                    plt.pause(0.001)

    with torch.no_grad():
        # Verify the result by inserting it as weights in the HelmholtzNet
        true_output_training_error, true_output_inference_error = (verify_result(_) for _ in (training_model.weight, inferred_field))
        log.info(f'Relative residue of Helmholtz equation after training: {true_output_training_error:0.12f}, and after inference: {true_output_inference_error:0.12f}')
        if display_progress:
            axs[0].set(title=f'training prec-$\\epsilon_j = {prec_training_errors[-1]:0.6f}$, true-$\\epsilon_j = {true_output_training_error:0.6f}$')
            axs[1].set(title=f'inference prec-$\\epsilon_E = {prec_inference_error:0.6f}$, true-$\\epsilon_j = {true_output_inference_error:0.6f}$')

    if display_progress:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        for _, plot in enumerate([axs[0].plot, axs[1].loglog]):
            plot(prec_training_errors, color='#ff0000', linewidth=1, linestyle='--', label='Prec. training $\epsilon_j$')
            plot(prec_inference_errors, color='#006000', linewidth=1, linestyle='--', label='Prec. inference $\epsilon_E$')
            plot(true_training_errors, color='#ff0000', linewidth=2, linestyle=':', label='True training $\epsilon_j$')
            plot(true_inference_errors, color='#006000', linewidth=2, linestyle=':', label='True inference $\epsilon_j$')
            plot(true_input_training_errors, color='#ff0000', linewidth=3, linestyle='-', label='True input training $\epsilon_E$')
            plot(true_input_inference_errors, color='#006000', linewidth=3, linestyle='-', label='True input inference $\epsilon_E$')
            axs[_].set(xlabel='iteration', ylabel='relative error', title='convergence',
                       xlim=[1, len(prec_training_errors)], ylim=[0 if _ == 0 else 1e-6, 1])
            axs[_].legend()

        plt.savefig(output_filepath.as_posix(), bbox_inches='tight', format='pdf')

        plt.show(block=True)
