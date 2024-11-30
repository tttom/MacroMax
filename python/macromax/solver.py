"""
This module calculates the solution to the wave equations. More specifically, the work is done in the iteration defined
in the :meth:`Solution.__iter__` method of the :class:`Solution` class. The convenience function :func:`solve` is
provided to construct a :class:`Solution` object and iterate it to convergence using its :meth:`Solution.solve` method.
"""
import numpy as np
import scipy.constants as const
import scipy.optimize
from typing import Union, Sequence, Callable, Optional
import logging

from . import backend
from macromax.utils.ft.grid import Grid
from macromax.bound import Bound, Electric, Magnetic, PeriodicBound

log = logging.getLogger(__name__)

array_like = Union[complex, Sequence, np.ndarray]


def solve(grid: Union[Grid, Sequence, np.ndarray], vectorial: Optional[bool] = None,
          wavenumber: Optional[float] = 1.0, angular_frequency: Optional[float] = None, vacuum_wavelength: Optional[float] = None,
          current_density: array_like = None, source_distribution: array_like = None,
          epsilon: array_like = None, xi: array_like = 0.0, zeta: array_like = 0.0, mu: array_like = 1.0,
          refractive_index: array_like = None,
          bound: Bound = None,
          initial_field: array_like = 0.0, dtype = None,
          callback: Callable = lambda s: s.iteration < 1e4 and s.residue > 1e-4):
    """
    Function to find a solution for Maxwell's equations in a media specified by the epsilon, xi,
    zeta, and mu distributions in the presence of a current source.

    :param grid: A Grid object or a Sequence of vectors with uniformly increasing values that indicate the positions
        in a plaid grid of sample points for the material and solution. In the one-dimensional case, a simple increasing
        Sequence of uniformly-spaced numbers may be provided as an alternative. The length of the ranges determines the
        data_shape, to which the source_distribution, epsilon, xi, zeta, mu, and initial_field must broadcast when
        specified as ndarrays.
    :param vectorial: a boolean indicating if the source and solution are 3-vectors-fields (True) or scalar fields (False).
    :param wavenumber: the wavenumber in vacuum = 2 pi / vacuum_wavelength.
        The wavelength in the same units as used for the other inputs/outputs.
    :param angular_frequency: alternative argument to the wavenumber = angular_frequency / c
    :param vacuum_wavelength: alternative argument to the wavenumber = 2 pi / vacuum_wavelength
    :param current_density: (optional, instead of source_distribution) An array or function that returns the free
        (vectorial) current density input distribution, J. The free current density has units of :math:`A m^-2`.
    :param source_distribution: (optional, instead of current_density) An array or function that returns
        the (vectorial) source input wave distribution. The source values relate to the current density, J,
        as  1j * angular_frequency * scipy.constants.mu_0 * J and has units of
        :math:`rad s^-1 H m^-1 A m^-2 = rad V m^-3`.
        More general, non-electro-magnetic wave problems can be solved using the source_distribution, as it does
        not rely on the vacuum permeability constant, :math:`mu_0`.
    :param epsilon: an array or function that returns the (tensor) epsilon that represents the permittivity at
        the points indicated by the grid specified as its input arguments.
    :param xi: an array or function that returns the (tensor) xi for bi-(an)isotropy at the
        points indicated by the grid specified as its input arguments.
    :param zeta: an array or function that returns the (tensor) zeta for bi-(an)isotropy at the
        points indicated by the grid specified as its input arguments.
    :param mu: an array or function that returns the (tensor) permeability at the
        points indicated by the grid specified as its input arguments.
    :param refractive_index: an array or function that returns the (complex) (tensor) refractive_index, as the square
        root of the permittivity, at the points indicated by the `grid` input argument.
    :param bound: An object representing the boundary of the calculation volume. Default: None, PeriodicBound(grid)
    :param initial_field: optional start value for the E-field distribution (default: all zero E)
    :param dtype: optional numpy datatype for the internal operations and results. This must be a complex number type
        as numpy.complex128 or np.complex64.
    :param callback: optional function that will be called with as argument this solver.
        This function can be used to check and display progress. It must return a boolean value of True to
        indicate that further iterations are required.
        
    :return: The Solution object that has the E and H fields, as well as iteration information.
    """
    return Solution(grid=grid, vectorial=vectorial,
                    wavenumber=wavenumber, angular_frequency=angular_frequency, vacuum_wavelength=vacuum_wavelength,
                    current_density=current_density, source_distribution=source_distribution,
                    epsilon=epsilon, xi=xi, zeta=zeta, mu=mu, refractive_index=refractive_index, bound=bound,
                    initial_field=initial_field, dtype=dtype).solve(callback)


class Solution(object):
    def __init__(self, grid: Union[Grid, Sequence, np.ndarray], vectorial: Optional[bool] = None,
                 wavenumber: Optional[float] = 1.0, angular_frequency: Optional[float] = None, vacuum_wavelength: Optional[float] = None,
                 current_density: array_like = None, source_distribution: array_like = None,
                 epsilon: array_like = None, xi: array_like = 0.0, zeta: array_like = 0.0, mu: array_like = 1.0,
                 refractive_index: array_like = None,
                 bound: Bound = None,
                 initial_field: array_like = 0.0, dtype=None):
        """
        Class a solution that can be further iterated towards a solution for Maxwell's equations in a media specified by
        the epsilon, xi, zeta, and mu distributions.

        :param grid: A Grid object or a Sequence of vectors with uniformly increasing values that indicate the positions
            in a plaid grid of sample points for the material and solution. In the one-dimensional case, a simple increasing
            Sequence of uniformly-spaced numbers may be provided as an alternative. The length of the ranges determines the
            data_shape, to which the source_distribution, epsilon, xi, zeta, mu, and initial_field must broadcast when
            specified as :py:class:`numpy.ndarray`.
        :param vectorial: a boolean indicating if the source and solution are 3-vectors-fields (True) or scalar fields (False).
            Default, True when vectorial nor the source is specified. False if the source is specified and scalar.
        :param wavenumber: the wavenumber in vacuum = 2pi / vacuum_wavelength.
            The wavelength in the same units as used for the other inputs/outputs.
        :param angular_frequency: alternative argument to the wavenumber = angular_frequency / c
        :param vacuum_wavelength: alternative argument to the wavenumber = 2 pi / vacuum_wavelength
        :param current_density: (optional, instead of source_distribution) An array or function that returns
            the (vectorial) current density input distribution, J. The current density has units of :math:`A m^{-2}`.
        :param source_distribution: (optional, instead of current_density) An array or function that returns
            the (vectorial) source input wave distribution. The source values relate to the current density, J,
            as  `1j * angular_frequency * scipy.constants.mu_0 * J` and has units of
            :math:`rad s^{-1} H m^{-1} A m^{-2} = rad V m^{-3}`.
            More general, non-electro-magnetic wave problems can be solved using the source_distribution, as it does
            not rely on the vacuum permeability constant, :math:`mu_0`.
        :param epsilon: an array or function that returns the (tensor) epsilon that represents the permittivity at the
            points indicated by the `grid` input argument.
        :param xi: an array or function that returns the (tensor) xi for bi-(an)isotropy at the points indicated by
            the `grid` input argument.
        :param zeta: an array or function that returns the (tensor) zeta for bi-(an)isotropy at the points indicated
            by the `grid` input argument.
        :param mu: an array or function that returns the (tensor) permeability at the points indicated by the `grid`
            input argument.
        :param refractive_index: an array or function that returns the (complex) (tensor) refractive_index, as the
            square root of the permittivity, at the points indicated by the `grid` input argument.
        :param bound: An object representing the boundary of the calculation volume. Default: None, PeriodicBound(grid)
        :param initial_field: optional start value for the E-field distribution (default: all zero E)
        :param dtype: optional numpy datatype for the internal operations and results. This must be a complex number type
            as numpy.complex128 or np.complex64.
        """
        self.__iteration = 0

        if not isinstance(grid, Grid):
            if np.isscalar(grid[0]):
                grid = [grid]  # This must be a 1D problem and the grid was specified as a single list of numbers
            grid = Grid.from_ranges(*grid)

        self.__grid = grid.immutable

        if angular_frequency is not None:
            self.__wavenumber = angular_frequency / const.c
            if vacuum_wavelength is not None:
                log.debug(f'Using specified angular_frequency = {angular_frequency}, ignoring vacuum_wavelength.')
        elif vacuum_wavelength is not None:
            self.__wavenumber = 2 * const.pi / vacuum_wavelength
            log.debug(f'Using vacuum_wavelength = {vacuum_wavelength}.')
        else:
            self.__wavenumber = wavenumber
            if angular_frequency is not None or vacuum_wavelength is not None:
                log.debug(f'Using specified wavenumber = {self.__wavenumber}, ignoring angular_frequency and vacuum_wavelength.')

        self.__previous_update_norm = np.inf

        self.__chi_op = None

        # Convert functions to arrays.
        # This is not strictly necessary for the source, though it simplifies the code.
        def func2arr(f):
            if callable(f):
                return f(*self.__grid)
            else:
                return np.asarray(f)

        # Determine the source distribution, either directly, or from current_density (assuming this is a an EM problem)
        if source_distribution is None:
            if current_density is None:
                if vectorial is None:
                    vectorial = True
                current_density = np.zeros((1 + 2 * vectorial, *self.grid.shape),
                                           dtype=dtype if dtype is not None else np.complex128)
            current_density = np.asarray(func2arr(current_density))
            source_distribution = current_density * (-1j * self.angular_frequency * const.mu_0)  # [ V m^-3 ]
        else:
            source_distribution = np.asarray(func2arr(source_distribution))

        # Decide whether this is a vectorial or a scalar calculation
        if vectorial is None:
            if source_distribution is not None:
                vectorial = source_distribution.ndim > self.grid.ndim and source_distribution.shape[0] == 3
            elif current_density is not None:
                vectorial = current_density.ndim > self.grid.ndim and current_density.shape[0] == 3
            else:
                vectorial = True
        self.__vectorial = vectorial

        # Set boundary conditions
        if bound is None:
            bound = PeriodicBound(self.grid)  # Default boundaries are periodic
        self.__bound = bound

        # Prepare a backend object to handle our parallel operations
        if dtype is None:
            dtype = np.asarray(source_distribution).dtype
        if not np.issubdtype(dtype, np.complexfloating):
            if (dtype == np.float16) or (dtype == np.float32):
                dtype = np.complex64
            else:  # np.float64, integer, bool
                dtype = np.complex128

        # Normalize the dimensions in the parallel operations to k0
        self.__BE = backend.load(1 + 2 * self.vectorial, self.grid * self.wavenumber, dtype=dtype)

        # Allocate the working memory
        self.__field_array = self.__BE.allocate_array()
        self.__d_field = self.__BE.array_ft_input

        # The following requires the self.__PO to be defined
        self.E = initial_field
        del initial_field
        self.__BE.clear_cache()

        # Adapt the material properties of the boundaries as defined by the `bound` argument.
        mu = func2arr(mu).astype(dtype)  # TODO: Is .astype(dtype) always redundant?
        mu = self.__BE.astype(mu)

        if epsilon is None:  # Calculate it as the square of the refractive index
            refractive_index = func2arr(refractive_index).astype(dtype) if refractive_index is not None else 1.0
            refractive_index = self.__BE.to_matrix_field(refractive_index)
            epsilon = self.__BE.mul(refractive_index, refractive_index)  # Square the refractive index to get permittivity
            # If negative refractive index material => invert both epsilon and mu
            if self.__BE.is_scalar(refractive_index) and self.__BE.any(self.__BE.real(self.__BE.ravel(refractive_index)) < 0):  # TODO: Can this be made to work for matrices by checking the eigenvalues? Problem: need to implement non-Hermitian eigenvalue decomposition for all backends, not just numpy and pytorch.
                mu = mu * self.__BE.astype(1 - 2 * (self.__BE.real(refractive_index) < 0))
                epsilon *= (1 - 2 * (self.__BE.real(refractive_index) < 0))
            del refractive_index
            self.__BE.clear_cache()
        else:
            epsilon = func2arr(epsilon).astype(dtype)
            epsilon = self.__BE.astype(epsilon)

        # Apply the boundary properties before the iteration
        if isinstance(self.__bound, Electric):
            bound_chi_epsilon = self.__BE.astype(self.__bound.electric_susceptibility)
            if np.any(np.asarray(epsilon.shape[:-self.grid.ndim]) > 1):
                bound_chi_epsilon = self.__BE.eye * bound_chi_epsilon
            epsilon = self.__BE.astype(epsilon + bound_chi_epsilon)
            del bound_chi_epsilon
        if isinstance(self.__bound, Magnetic):
            bound_chi_mu = self.__BE.astype(self.__bound.magnetic_susceptibility)
            if np.any(np.asarray(mu.shape[:-self.grid.ndim]) > 1):
                bound_chi_mu = self.__BE.eye * bound_chi_mu
            mu = self.__BE.astype(mu + self.__BE.eye * bound_chi_mu)
            del bound_chi_mu
        self.__BE.clear_cache()

        xi = func2arr(xi)
        zeta = func2arr(zeta)

        # Before the first iteration, the pre-conditioner must be determined and applied to the source and medium
        self.__source_normalized = None
        self.__prepare_preconditioner(
            source_distribution, epsilon, xi, zeta, mu, self.grid.step, self.wavenumber
        )
        # Now we can forget epsilon, xi, zeta, mu, and source_distribution. Their information is encapsulated
        # in the newly created operator method __chi_op
        del mu, zeta, xi, epsilon
        self.__BE.clear_cache()
        self.__residue = None  # Invalid residue

    def __prepare_preconditioner(self, source_distribution, epsilon, xi, zeta, mu, sample_pitch, wavenumber):
        """
        Sets or updates the private value for self.__source_normalized, and the methods __chi_op and __green_function_op
        This uses the values for source_distribution, epsilon, xi, zeta, mu, as well as
        the sample_pitch and wavenumber.

        :param source_distribution: A 1+N-D array representing the source vector field.
        :param epsilon: A 2+N-D array representing the permittivity distribution.
        :param xi: A 2+N-D array representing the xi bi-anistrotropy distribution.
        :param zeta: A 2+N-D array representing the zeta bi-anistrotropy distribution.
        :param mu: A 2+N-D array representing the permeability distribution.
        :param sample_pitch: A vector with numbers indicating the sample distance in each dimension.
        :param wavenumber: The wavenumber, k,  of the coherent illumination considered for this problem.

        :returns None.
            Sets the private attributes:
            - self.__magnetic
            - self.__beta
            - self.__source_normalized
            - self.__chiEH
            - self.__chiHE
            - self.__chiHH
            - self.__chiEE_base
            - self.__alpha
            - self.__chi_op
            - self.__green_function_op
        """
        log.debug('Preparing pre-conditioner: determining alpha and beta...')

        # Convert all inputs to the canonical form
        epsilon = self.__BE.to_matrix_field(epsilon)
        xi = self.__BE.to_matrix_field(xi)
        zeta = self.__BE.to_matrix_field(zeta)
        mu = self.__BE.to_matrix_field(mu)

        # Determine if the media is magnetic
        self.__magnetic = not (self.__BE.allclose(xi) and self.__BE.allclose(zeta)
                               and self.__BE.allclose(mu, self.__BE.first(mu)))
        if self.magnetic:
            log.debug('Material has magnetic properties.')
        else:
            log.debug('Material has no magnetic properties. Using faster permittivity-only solver.')

        def largest_eigenvalue(a):  # todo: relatively slow operation during startup
            return self.__BE.amax(self.__BE.abs(self.__BE.mat3_eigh(a)))

        def largest_singularvalue(a):
            return (largest_eigenvalue(self.__BE.mul(self.__BE.adjoint(a), a))) ** 0.5

        # Do a quick check to see if the media has no gain
        max_allowed_gain = np.sqrt(self.__BE.eps)

        def has_gain(a) -> bool:
            gain_condition = self.__BE.real(
                self.__BE.mat3_eigh(-0.5j * (a - self.__BE.adjoint(a)))) < - max_allowed_gain
            result = self.__BE.any(gain_condition)
            del gain_condition
            del a
            self.__BE.clear_cache()
            return result

        def has_loss(a) -> bool:
            return has_gain(-a)

        transpose = (has_gain(epsilon) or has_gain(xi) or has_gain(zeta) or has_gain(mu)
                     ) and not (has_loss(epsilon) or has_loss(xi) or has_loss(zeta) or has_loss(mu))

        if not transpose and (has_gain(epsilon) or has_gain(xi) or has_gain(zeta) or has_gain(mu)):
            def max_gain(a):
                return self.__BE.amax(-self.__BE.real(self.__BE.mat3_eigh(-0.5j * (a - self.__BE.adjoint(a)))))

            log.warning(f'Convergence not guaranteed!\n'
                        f'Permittivity has a gain as large as {max_gain(epsilon):0.3g}, xi up to {max_gain(xi):0.3g}, zeta up to {max_gain(zeta):0.3g},'
                        f' and the permeability up to {max_gain(mu):0.3g}. All are expected to be less than {max_allowed_gain:0.3g}.')
        else:
            log.debug('Material has no gain, safe to proceed.')

        # determine alpha and beta for the given media
        # Determine mu^-2
        mu_inv = self.__BE.inv(mu)

        # Determine calcChiHH, calcSigmaHH
        if self.magnetic:
            def calc_chiHH(beta_):
                return self.__BE.subtract(1.0, mu_inv / beta_)

            mu_inv_transpose = self.__BE.adjoint(mu_inv)
            mu_inv2 = self.__BE.mul(mu_inv_transpose, mu_inv)  # Positive definite

            def calc_chiHH_beta2(beta_):
                return (mu_inv2 +
                        self.__BE.subtract(np.abs(beta_) ** 2, mu_inv_transpose * beta_ + mu_inv * np.conj(beta_))
                        )

            def calc_sigmaHH(beta_):
                return largest_eigenvalue(calc_chiHH_beta2(beta_)) ** 0.5 / abs(beta_)

            if has_gain(mu) or has_gain(xi) or has_gain(zeta):
                log.warning('Permeability or bi-(an)isotropy has gain. Convergence not guaranteed!')
            else:
                log.debug('Permeability and bi-(an)isotropy have no gain, safe to proceed.')
        else:
            # non-magnetic, mu is scalar and both xi and zeta are zero
            def calc_chiHH(beta_):
                return 1.0 - mu_inv * (1 / beta_)  # always zero when beta == mu_inv

            def calc_sigmaHH(beta_):
                return abs(calc_chiHH(beta_))  # always zero when beta == mu_inv

        # Determine: calcChiEE, chiEHTheta, chiHETheta, chiHH, alpha, beta
        chiEH_beta = -1.0j * self.__BE.mul(xi, mu_inv)
        chiHE_beta = 1.0j * self.__BE.mul(mu_inv, zeta)

        # zeta needed for conversion from E to H
        xi_mu_inv_zeta = self.__BE.mul(xi, -1.0j * chiHE_beta)
        del xi
        epsilon_xi_mu_inv_zeta = self.__BE.subtract(epsilon, xi_mu_inv_zeta)
        del epsilon, xi_mu_inv_zeta
        self.__BE.clear_cache()

        # TODO: These were put in calc_deltaEE_beta2 to save memory [but slowdown startup (performed 2x in this fun)]
        # epsilon_xi_mu_inv_zeta_transpose = self.__BE.adjoint(epsilon_xi_mu_inv_zeta)
        # epsilon_xi_mu_inv_zeta2 = self.__BE.mul(epsilon_xi_mu_inv_zeta_transpose, epsilon_xi_mu_inv_zeta)

        # The above must be positive definite
        def calc_DeltaEE_beta2(alpha_, beta_):  # todo: relatively slow during startup
            alpha_beta = self.__BE.astype(self.__BE.real(alpha_) * beta_)
            result = self.__BE.adjoint(epsilon_xi_mu_inv_zeta) * alpha_beta
            result += epsilon_xi_mu_inv_zeta * self.__BE.conj(alpha_beta)
            result = self.__BE.subtract(self.__BE.abs(alpha_beta) ** 2, result)
            result += self.__BE.mul(self.__BE.adjoint(epsilon_xi_mu_inv_zeta), epsilon_xi_mu_inv_zeta)
            return result

        def calc_sigmaEE(alpha_, beta_):  # todo: relatively slow during startup
            return float(largest_eigenvalue(calc_DeltaEE_beta2(alpha_, beta_))) ** 0.5 / abs(float(beta_))

        # Determine alpha, beta and chiHH
        alpha_tolerance = 0.01
        if self.magnetic:
            # Optimize the real part of alpha and beta
            sigmaD = np.linalg.norm(2.0 * np.pi / (2.0 * sample_pitch)) / wavenumber
            sigmaHE_beta = largest_singularvalue(chiHE_beta)
            sigmaEH_beta = largest_singularvalue(chiEH_beta)

            def max_singular_value_sum(eta_, beta_):
                return sigmaD ** 2 * calc_sigmaHH(beta_) + \
                       sigmaD * (sigmaEH_beta + sigmaHE_beta) / np.abs(beta_) + calc_sigmaEE(eta_, beta_)

            def beta_from_vec(alpha_beta_vec_):
                return alpha_beta_vec_[1] ** 2  # enforce positivity

            def target_function_vec(vec):
                beta = beta_from_vec(vec)
                return max_singular_value_sum(vec[0], beta) * beta + np.maximum(0.0,
                                                                                beta - 1e-3) ** 2  # ensure that beta doesn't get too close to 0

            log.debug('Finding optimal alpha and beta...')
            try:
                alpha_beta_vec = scipy.optimize.fmin(target_function_vec, [0, 1],
                                                     initial_simplex=[[0, 1], [1, 1], [0, 0.9]],
                                                     disp=False, full_output=False,
                                                     ftol=alpha_tolerance, xtol=alpha_tolerance, maxiter=100,
                                                     maxfun=200)
            except TypeError:  # Some older scipy implementations don't seem to have the initial_simplex argument
                alpha_beta_vec = scipy.optimize.fmin(target_function_vec, [0, 1],
                                                     disp=False, full_output=False,
                                                     ftol=alpha_tolerance, xtol=alpha_tolerance, maxiter=100,
                                                     maxfun=200)
            self.__beta = beta_from_vec(alpha_beta_vec)

            alpha = alpha_beta_vec[0] + 1.0j * self.__BE.asnumpy(max_singular_value_sum(alpha_beta_vec[0], self.__beta))

            del beta_from_vec, alpha_beta_vec
        else:
            # non-magnetic
            self.__beta = self.__BE.asnumpy(self.__BE.first(mu_inv)).real  # Must be scalar and real

            def target_function_vec(alpha_):
                return calc_sigmaEE(alpha_, self.__beta)  # beta is fixed to mu_inv so that chi_HH==0

            log.debug('beta = %0.4g, finding optimal alpha...' % self.__beta)
            try:
                alpha_real, min_value = scipy.optimize.fmin(target_function_vec, 0.0, initial_simplex=[[0.0], [1.0]],
                                                            disp=False, full_output=True,
                                                            ftol=alpha_tolerance, xtol=alpha_tolerance,
                                                            maxiter=100, maxfun=100)[:2]
            except TypeError:  # Some older scipy implementations don't seem to have the initial_simplex argument
                alpha_real, min_value = scipy.optimize.fmin(target_function_vec, 0.0,
                                                            disp=False, full_output=True,
                                                            ftol=alpha_tolerance, xtol=alpha_tolerance,
                                                            maxiter=100, maxfun=100)[:2]

            alpha = alpha_real[0] + 1.0j * min_value
            if transpose:
                alpha = alpha.conj()

            del alpha_real
        self.__BE.clear_cache()

        alpha = self.__increase_bias_to_limit_kernel_width(alpha)
        log.info(f'Preconditioner constants: alpha = {alpha.real:0.4g} + {alpha.imag:0.4g}i, beta = {self.__beta:0.4g}')

        log.debug('Preparing pre-conditioned operators...')
        self.__alpha = alpha
        self.__chiEH = self.__BE.mul(chiEH_beta, 1.0j / self.__alpha.imag / self.__beta)
        del chiEH_beta
        self.__chiHE = self.__BE.mul(chiHE_beta, 1.0j / self.__alpha.imag / self.__beta)
        del chiHE_beta
        self.__chiHH = self.__BE.mul(calc_chiHH(self.__beta), 1.0j / self.__alpha.imag)
        self.__chiEE_base = self.__BE.mul(epsilon_xi_mu_inv_zeta, 1.0j / self.__alpha.imag / self.__beta)
        if self.__chiEE_base.shape[0] == 1:
            self.__chiEE_base -= self.__alpha * 1.0j / self.__alpha.imag
        else:
            self.__chiEE_base -= self.__BE.eye * (self.__alpha * 1.0j / self.__alpha.imag)

        # Store the modified source distribution
        self.source_distribution = source_distribution
        del source_distribution, epsilon_xi_mu_inv_zeta
        self.__BE.clear_cache()

        # Update the operators that are stored as private attributes
        self.__update_operators(alpha)
        self.__BE.clear_cache()



    def __update_operators(self, alpha):
        """
        Updates the value of alpha, and the operators for chi as well as the Green function.
        This is used in __prepare_preconditioner() and __iter__().

        :param alpha: The new value for alpha.

        :returns None
            Changes the private attributes:
            - self.__source_normalized
            - self.__chiEH
            - self.__chiHE
            - self.__chiHH
            - self.__chiEE_base
            - self.__alpha
            - self.__chi_op
            - self.__green_function_op
        """
        # Rescale the source
        self.__source_normalized *= self.__alpha.imag / alpha.imag
        # Correct the magnetic components for changes in alpha.imag
        self.__chiEH *= self.__alpha.imag / alpha.imag
        self.__chiHE *= self.__alpha.imag / alpha.imag
        self.__chiHH *= self.__alpha.imag / alpha.imag
        # Once the susceptibility_offset is fixed, we can also calculate chiEE
        self.__chiEE_base /= 1.0j / self.__alpha.imag
        if self.__chiEE_base.shape[0] == 1:
            self.__chiEE_base += self.__alpha - alpha  # remove the previous alpha before applying new one
        else:
            self.__chiEE_base += self.__BE.eye * (
                        self.__alpha - alpha)  # remove the previous alpha before applying new one
        self.__chiEE_base *= 1.0j / alpha.imag
        self.__alpha = alpha

        # Define the Chi operator for magnetic or non-magnetic (potentially anisotropic)
        if self.magnetic:
            def D(field_E):
                return self.__BE.curl(field_E)  # includes k0^-1 by the definition of __PO

            def chi_op(E):
                """
                Applies the magnetic :math:`\\Chi` operator to the input E-field.

                :param E: an array representing the E to apply the Chi operator to.

                :return: an array with the result E of the same size as E or of the size of its singleton expansion.
                """
                chiE = self.__BE.mul(self.__chiEE_base, E)
                chiH = self.__BE.mul(self.__chiEH, E)
                ED = D(E)
                chiE += self.__BE.mul(self.__chiHE, ED)
                chiH += self.__BE.mul(self.__chiHH, ED, out=ED)

                result = chiE
                result += D(chiH)

                return result  # chiE + D(chiH)
        else:
            def chi_op(E):
                """
                Applies the non-magnetic Chi operator to the input E-field.

                :param E: an array representing the E to apply the Chi operator to.

                :return: an array with the result E of the same size as E or of the size of its singleton expansion.
                """
                return self.__BE.mul(self.__chiEE_base, E, out=E)

        # Now create the Green function operator
        # Pre-calculate the convolution filter
        g_scalar_ft = self.__BE.k2 - self.__BE.astype(self.__alpha)
        self.__BE.clear_cache()
        g_scalar_ft = self.__BE.astype(-1.0j) * self.__BE.astype(self.__alpha.imag) / g_scalar_ft

        if self.__BE.vectorial:
            def g_ft_op(FFt):  # Overwrites input argument! No need to represent the full matrix in memory
                PiL_FFt = self.__BE.longitudinal_projection_ft(FFt)  # relatively memory intensive
                FFt -= PiL_FFt
                FFt *= g_scalar_ft
                PiL_FFt *= 1.0j * self.__alpha.imag / self.__alpha
                FFt += PiL_FFt
                return FFt  # g_scalar_ft * self.__PO.subtract(FFt, PiL_FFt) - PiL_FFt * 1.0j * self.__alpha.imag / self.__alpha
        else:
            def g_ft_op(FFt):  # Overwrites input argument!
                FFt *= g_scalar_ft
                return FFt  # g_scalar_ft * FFt

        def dyadic_green_function_op(F):  # overwrites input argument!
            return self.__BE.convolve(g_ft_op, F)  # g_ft_op is memory intensive

        # Update the methods for the operator Chi:
        self.__chi_op = chi_op
        # Set the Green function
        self.__green_function_op = dyadic_green_function_op

    def __increase_bias_to_limit_kernel_width(self, alpha, max_kernel_field_residue=0.001):
        """
        Used in __prepare_preconditioner().

        Limit the kernel size by increasing alpha so that the 1D kernel field decreases to `max_kernel_field_residue`
        at the other side of the boundary.

        :param alpha: the complex constant in the pre-conditioner to ensure convergence
        :param max_kernel_field_residue: The maximum amplitude that a 1D kernel should have after traversing the boundary.
            Note that if waves pass both ways through the boundary, the amplitude of the interference may be twice this.

        :return: the adapted alpha value (real and imaginary parts)
        """
        susceptibility_offset = alpha.imag + 0.05  # todo: How much margin should we add to avoid numerical issues?

        # Ignore boundary thicknesses that are set to 0, these are meant to be periodic boundaries
        thicknesses = self.__bound.thickness[self.__bound.thickness != 0]
        if thicknesses.size > 0:
            bound_thickness = np.amin(thicknesses)

            min_kappa = np.log(max_kernel_field_residue) / (
                        - self.wavenumber * bound_thickness)  # extinction coefficient
            corresponding_n_real = (np.maximum(0, float(
                alpha.real) + min_kappa ** 2) ** 0.5)  # Calculate the real part of the refractive index from alpha.real and min_kappa
            min_susceptibility_offset = 2 * corresponding_n_real * min_kappa  # because alpha = (n + i*kappa)^2, so alpha.imag = 2 * n * kappa

            if susceptibility_offset < min_susceptibility_offset:
                log.info(
                    f'Increasing susceptibility offset to {min_susceptibility_offset} in order to avoid passing through the boundary of thickness {bound_thickness}.')
                susceptibility_offset = min_susceptibility_offset
            else:
                log.debug(
                    f'Minimum susceptibility offset {min_susceptibility_offset} is lower than that required for the permittivity variation: {susceptibility_offset}.')

        alpha = alpha.real + 1j * susceptibility_offset

        return alpha

    @property
    def grid(self) -> Grid:
        """
        The sample positions of the plaid sampling grid.
        This may be useful for displaying result axes.

        :return: A Grid object representing the sample points of the fields and material.
        """
        return self.__grid

    @property
    def vectorial(self) -> bool:
        """Boolean to indicates whether calculations happen on vectorial (True) or scalar (False) fields."""
        return self.__vectorial

    @property
    def dtype(self):
        """The numpy equivalent data type used in the calculation. This is either np.complex64 or np.complex128."""
        return self.__BE.numpy_dtype

    @property
    def wavenumber(self) -> float:
        """
        The vacuum wavenumber, :math:`k_0`, used in the calculation.

        :return: A scalar indicating the wavenumber used in the calculation.
        """
        return self.__wavenumber

    @property
    def angular_frequency(self) -> float:
        r"""
        The angular frequency, :math:`\omega`, used in the calculation.

        :return: A scalar indicating the angular frequency used in the calculation.
        """
        return self.__wavenumber * const.c

    @property
    def wavelength(self) -> float:
        r"""
        The vacuum wavelength, :math:`\lambda_0`, used in the calculation.

        :return: A scalar indicating the vacuum wavelength used in the calculation.
        """
        return 2.0 * const.pi / self.__wavenumber

    @property
    def magnetic(self) -> bool:
        """
        Indicates if this media is considered magnetic.

        :return: A boolean, True when magnetic, False otherwise.
        """
        return self.__magnetic

    @property
    def bound(self) -> Bound:
        """The Bound object that defines the calculation boundaries."""
        return self.__bound

    @property
    def source_distribution(self) -> np.ndarray:
        """
        The source distribution, i k0 mu_0 times the current density j.

        :return: A complex array indicating the amplitude and phase of the source vector field.
            The dimensions of the array are [1|3, self.grid.shape], where the first dimension is 1 in case of a scalar
            field, and 3 in case of a vector field.
        """
        return self.__source_normalized[:, 0, ...] * (self.__beta / 1.0j * self.__alpha.imag)

    @source_distribution.setter
    def source_distribution(self, new_source_dist: array_like):
        """
        Set the source distribution, i k0 mu_0 times the current density j.

        :param new_source_dist: A complex array indicating the amplitude and phase of the source vector field.
            The dimensions of the array are [1|3, self.grid.shape], where the first dimension is 1 in case of a scalar
            field, and 3 in case of a vector field.
        """
        new_source_dist = self.__BE.to_matrix_field(new_source_dist)

        if self.__source_normalized is None:
            self.__source_normalized = self.__BE.mul(new_source_dist, (1.0j / self.__alpha.imag / self.__beta))
        else:
            self.__source_normalized = self.__BE.assign(
                new_source_dist * (1.0j / self.__alpha.imag / self.__beta), self.__source_normalized)  # Adjust for bias
        del new_source_dist
        self.__BE.clear_cache()
        self.__previous_update_norm = np.inf

    @property
    def j(self) -> np.ndarray:
        """
        The free current density, j, of the source vector field.

        :return: A complex array indicating the amplitude and phase of the current density vector field [A m^-2].
            The dimensions of the array are [1|3, self.grid.shape], where the first dimension is 1 in case of a scalar field,
            and 3 in case of a vector field.
        """
        return self.__BE.asnumpy(self.source_distribution / (-1.0j * self.angular_frequency * const.mu_0))

    @j.setter
    def j(self, new_j: array_like):
        """
        Set the free current density, j, of the source vector field.

        :param new_j: A complex array indicating the amplitude and phase of the current density vector field [A m^-2].
            The dimensions of the array are [1|3, self.grid.shape], where the first dimension is 1 in case of a scalar field,
            and 3 in case of a vector field.
        """
        self.source_distribution = self.__BE.astype(new_j) * (-1.0j * self.angular_frequency * const.mu_0)

    @property
    def E(self) -> np.ndarray:
        """
        The electric field for every point in the sample space (SI units).

        :return: A vector array with the first dimension containing Ex, Ey, and Ez,
            while the following dimensions are the spatial dimensions.
        """
        result = self.__field_array[:, 0, ...] / (self.wavenumber ** 2)
        result = self.__BE.asnumpy(result)
        return result

    @E.setter
    def E(self, E):
        """
        The electric field for every point in the sample space (SI units).

        :param E: The new field. A vector array with the first dimension containing :math:`E_x, E_y, and E_z`,
            while the following dimensions are the spatial dimensions.
        """
        self.__field_array = self.__BE.assign(E * (self.wavenumber ** 2), self.__field_array)
        self.__previous_update_norm = np.inf

    @property
    def B(self) -> np.ndarray:
        """
        The magnetic field for every point in the sample space (SI units).
        This is calculated from H and E.

        :return: A vector array with the first dimension containing :math:`B_x, B_y, and B_z`,
            while the following dimensions are the spatial dimensions.
        """
        B = self.__BE.curl(self.E[:, np.newaxis, ...]) / (1.0j * const.c)  # curl includes k0 by definition of __PO

        return self.__BE.asnumpy(B)[:, 0, ...]

    @property
    def D(self) -> np.ndarray:
        """
        The displacement field for every point in the sample space (SI units).
        This is calculated from E and H.

        :return: A vector array with the first dimension containing :math:`D_x, D_y, and D_z`,
            while the following dimensions are the spatial dimensions.
        """
        # D = (J - self.__PO.curl(self.H[:, np.newaxis, ...]) * self.wavenumber) / (1.0j * self.angular_frequency)  # curl includes k0 by definition of __PO
        D = self.__BE.curl(self.H[:, np.newaxis, ...])
        D *= self.wavenumber
        D -= (self.__beta / (1.0j * self.angular_frequency * const.mu_0)) * (
                    self.__source_normalized * (self.__alpha.imag / 1.0j))
        D *= 1j / self.angular_frequency

        return self.__BE.asnumpy(D)[:, 0, ...]

    @property
    def H(self) -> np.ndarray:
        """
        The magnetizing field for every point in the sample space (SI units).
        This is calculated from E.

        :return: A vector array with the first dimension containing :math:`H_x, H_y, and H_z`,
            while the following dimensions are the spatial dimensions.
        """
        if self.magnetic:
            # Use stored matrices to safe the space
            # Use stored matrices to safe the space
            mu_inv = self.__BE.subtract(1.0, self.__chiHH * (-1.0j * self.__alpha.imag)) * self.__beta

            # the curl in the following includes factor k0^-1 by the definition of __PO above
            H = self.__BE.astype(1.0j / (const.mu_0 * const.c) * (
                - self.__BE.mul(mu_inv, self.__BE.curl(self.__BE.astype(self.E[:, np.newaxis, ...])))
                + self.__beta * self.__BE.mul(self.__chiHE * (-1.0j * self.__alpha.imag),
                                              self.E[:, np.newaxis, ...])
            )
                                 )
        else:
            mu_inv = (1.0 - self.__BE.first(self.__chiHH) * (-1.0j * self.__alpha.imag)) * self.__BE.astype(self.__beta)
            mu_H = (-1.0j / (const.mu_0 * const.c)) * self.__BE.curl(
                self.__BE.astype(self.E[:, np.newaxis, ...]))  # includes k0^-1 by the definition of __PO
            H = mu_inv * mu_H

        return self.__BE.asnumpy(H)[:, 0, ...]

    @property
    def S(self) -> np.ndarray:
        """
        The time-averaged Poynting vector for every point in space.
        :return: A vector array with the first dimension containing :math:`S_x, S_y, and S_z`,
        while the following dimensions are the spatial dimensions.
        """
        E = self.E[:, np.newaxis, ...]

        H = self.H[:, np.newaxis, ...]
        poynting_vector = 0.5 * self.__BE.asnumpy(self.__BE.cross(self.__BE.astype(E), self.__BE.conj(H))).real

        return poynting_vector[:, 0, ...]

    @property
    def energy_density(self) -> np.ndarray:
        """
        Returns the energy density, u.

        :return: A real array indicating the energy density in space.
        """
        E = self.E
        B = self.B  # Can be calculated more efficiently from E, though avoiding code-replication for now.
        H = self.H  # Can be calculated more efficiently from B and E, though avoiding code-replication for now.
        D = self.D  # Can be calculated more efficiently from H, though avoiding code-replication for now.

        u = np.sum(self.__BE.asnumpy((self.__BE.astype(E) * self.__BE.conj(D)).real), axis=0)
        u += np.sum(self.__BE.asnumpy((self.__BE.astype(B) * self.__BE.conj(H)).real), axis=0)
        u *= 0.5 * 0.5  # 0.5 * (E.D' + B.H'), the other 0.5 is because we time-average the products of real functions

        return u

    @property
    def stress_tensor(self) -> np.ndarray:
        """
        Maxwell's stress tensor for every point in space.

        :return: A real and symmetric matrix-array with the stress tensor for every point in space.
            The units are :math:`N / m^2`.
        """
        E = self.E[:, np.newaxis, ...]
        E2 = np.sum(np.abs(E) ** 2, axis=0)
        result = const.epsilon_0 * self.__BE.outer(E, E)

        H = self.H[:, np.newaxis, ...]
        H2 = np.sum(np.abs(H) ** 2, axis=0)
        result += self.__BE.outer(H, H) * const.mu_0

        result -= (0.5 * self.__BE.eye) * self.__BE.astype((const.epsilon_0 * E2 + H2 * const.mu_0))
        result = 0.5 * result  # TODO: Do we want the Abraham or Minkowski form?
        result = self.__BE.asnumpy(result)

        return result.real

    @property
    def f(self) -> np.ndarray:
        """
        The electromagnetic force density (force per SI unit volume, not per voxel).

        :return: A vector array representing the electro-magnetic force exerted per unit volume.
            The first dimension contains :math:`f_x, f_y, and f_z`, while the following dimensions are the spatial dimensions.
            The units are :math:`N / m^3`.
        """
        # Could be written more efficiently by either caching H or explicitly calculating the stress tensor
        # in this method. Leaving this for now, so to avoid code replication.
        force = self.__BE.asnumpy(self.__BE.real(self.__BE.div(self.stress_tensor)) * self.wavenumber)  # The parallel operations is pre-scaled by k0
        # The time derivative of the Poynting vector averages to zero.
        # Make sure to remove imaginary part which must be due to rounding errors

        return force[:, 0]

    @property
    def torque(self) -> np.ndarray:
        """
        The electromagnetic force density (force per SI unit volume, not per voxel).

        :return: A vector array representing the electro-magnetic torque exerted per unit volume.
            The first dimension contains torque_x, torque_y, and torque_z, while the following dimensions are the spatial
            dimensions. The units are :math:`N m / m^3 = N m^{-2}`.
        """
        # Could be written more efficiently by either caching H or explicitly calculating the stress tensor
        # in this method. Leaving this for now, so to avoid code replication.
        sigma = self.stress_tensor  # Units N m^-2
        torque = np.r_[sigma[1, 2, np.newaxis], sigma[0, 2, np.newaxis], sigma[0, 1, np.newaxis]]
        # torque = self.__PO.curl(sigma).real * self.wavenumber  # The parallel operations is pre-scaled by k0
        # # Make sure to remove imaginary part which must be due to rounding errors
        # torque *= vector_to_axis(self.grid.step, n=self.__PO.nb_dims, axis=1)
        # torque = np.sum(torque, axis=1)

        return torque

    @property
    def iteration(self) -> int:
        """
        The current iteration number.

        :return: An integer indicating how many iterations have been done.
        """
        return self.__iteration

    @iteration.setter
    def iteration(self, it: int = 0):
        """
        The current iteration number.
        Resets the iteration count or sets it to a specified integer.
        This does not affect the calculation, only (potentially) the stop criterion.

        :param it: (optional) the new iteration number
        """
        self.__iteration = it

    @property
    def previous_update_norm(self) -> float:
        """
        The L2-norm of the last update, the difference between current and previous E-field.

        :return: A positive scalar indicating the norm of the last update.
        """
        return float(self.__previous_update_norm / (self.wavenumber ** 2))

    @property
    def residue(self) -> float:
        """
        Returns the current relative residue of the inverse problem :math:`E = H^{-1}S`.
        The relative residue is return as the l2-norm fraction :math:`||E - H^{-1}S|| / ||E||`, where H represents the
        vectorial Helmholtz equation following Maxwell's equations and S the current density source. The solver
        searches for the electric field, E, that minimizes the preconditioned inverse problem.

        :return: A non-negative real scalar that indicates the change in E with the previous iteration
        	normalized to the norm of the current E.
        """
        if self.__residue is None:
            self.__residue = self.__previous_update_norm / self.__BE.norm(self.__field_array) \
                if self.__previous_update_norm > 0 else 0

        return float(self.__residue)

    def __iter__(self):
        """
        Returns an iterator that on __next__() yields this Solution after updating it with one cycle of the algorithm.
        Obtaining this iterator resets the iteration counter.

        Usage:

        .. code:: python

            for solution in Solution(...):
                if solution.iteration > 100:
                    break
            print(solution.residue)
        """
        self.iteration = 0  # reset iteration counter
        while True:
            self.iteration += 1
            self.__residue = None  # Invalidate residue
            log.debug(f'Starting iteration {self.iteration}...')

            # Calculate update to the field (self.__field_array, d_field, and self.__source are scaled by k0^2)
            self.__d_field = self.__BE.assign_exact(self.__field_array, self.__d_field)   #      E
            d_field = self.__chi_op(self.__d_field)                                       #     XE
            d_field += self.__source_normalized                                           #     XE + s
            d_field = self.__green_function_op(d_field)                                   #   G(XE + s)
            d_field -= self.__field_array                                                 #   G(XE + s) - E
            d_field = self.__chi_op(d_field)                                              # X[G(XE + s) - E]

            # Determine convergence rate
            current_update_norm = self.__BE.norm(d_field)  # ||d||
            relative_update_norm = current_update_norm / self.__previous_update_norm
            # log.debug(f'The norm of the field update has changed by a factor {relative_update_norm:0.3f}.')

            # Check if the iteration is diverging
            if relative_update_norm < 1:
                # Update solution
                self.__field_array += d_field              # X[G(XE + s) - E] + E
                # Keep update norm for next iteration's convergence check
                self.__previous_update_norm = current_update_norm
                # log.debug(f'Updated field in iteration {self.iteration}.')
            else:
                log.warning(f'The field update is scaled by {relative_update_norm:0.3f} >= 1 by {relative_update_norm - 1.0:0.3e} in iteration {self.iteration}.\nThis may be due to numerical accuracy or the presence of gain in the material.')
                alpha_imag_current = self.__alpha.imag
                alpha_new = self.__alpha.real + 1.50j * self.__alpha.imag
                log.info(f'Increasing the imaginary part of alpha from {alpha_imag_current:0.3g} to {alpha_new.imag:0.3g}.')
                self.__update_operators(alpha_new)

                log.debug(f'Aborting field update in iteration {self.iteration}.')

            yield self

    def solve(self, callback: Callable = lambda _: _.iteration < 1e4 and _.residue > 1e-4):
        """
        Runs the algorithm until the convergence criterion is met or until the maximum number of iterations is reached.

        :param callback: optional callback function that overrides the one set for the solver.
            E.g. callback=lambda s: s.iteration < 100

        :return: This Solution object, which can be used to query e.g. the final field E using Solution.E.
        """
        for sol in self:
            # sol is the current iteration result
            # now execute the user-specified callback function
            if not callback(sol):  # function may have side effects
                log.debug('Convergence target met, stopping iteration.')
                break  # stop the iteration if it returned False

        return self
