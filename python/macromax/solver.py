import numpy as np
import scipy.constants as const
import scipy.optimize

from .parallel_ops_column import ParallelOperations
from . import utils

from . import log
from .utils.array import Grid


def solve(ranges=None, sample_pitch=None,
          wavenumber=None, angular_frequency=None, vacuum_wavelength=None,
          source_distribution=None, epsilon=None, xi=0.0, zeta=0.0, mu=1.0,
          initial_field=0.0, callback=lambda s: s.iteration < 1e4 and s.residue > 1e-4):
    """
        Function to find a solution for Maxwell's equations in a media specified by the epsilon, xi,
        zeta, and mu distributions in the presence of a current source.

        :param ranges: an array of vectors with uniformly increasing values that indicate the positions in a
            plaid grid at which to calculate the solution at. In the case of 1D, a simple vector may be provided.
            The length of the ranges determines the dataSize, which must match the dimensions of (the output of)
            source_distribution, epsilon, xi, zeta, mu, and the optional start E, unless these are singletons.
            This argument may be replaced by sample_pitch.
        :param sample_pitch: Optional, replaces ranges by zero-centered ranges with the given sample pitch.
        :param data_shape: Only required if ranges is not specified and all distributions are specified as functions.
        :param wavenumber: the wavenumber in vacuum = 2 pi / vacuum_wavelength.
            The wavelength in the same units as used for the other inputs/outputs.
        :param angular_frequency: alternative argument to the wavenumber = angular_frequency / c
        :param vacuum_wavelength: alternative argument to the wavenumber = 2 pi / vacuum_wavelength
        :param source_distribution: an array or function that returns the (vectorial) source input wave distribution.
            The source values relate to the current density, J, as  1j*angularFrequency*scipy.constants.mu_0*J
            and has units of :math:`rad s^-1 H m^-1 A m^-2 = rad V m^-3`.
        :param epsilon: an array or function that returns the (tensor) epsilon that represents the permittivity at
            the points indicated by the ranges specified as its input arguments.
        :param xi: an array or function that returns the (tensor) xi for bi-(an)isotropy at the
            points indicated by the ranges specified as its input arguments.
        :param zeta: an array or function that returns the (tensor) zeta for bi-(an)isotropy at the
            points indicated by the ranges specified as its input arguments.
        :param mu: an array or function that returns the (tensor) permeability at the
            points indicated by the ranges specified as its input arguments.
        :param initial_field: optional start value for the E-field distribution (default: all zero E)
        :param callback: optional function that will be called with as argument this solver.
            This function can be used to check and display progress. It must return a boolean value of True to
            indicate that further iterations are required.
        :return: The Solution object that has the E and H fields, as well as iteration information.
        """
    return Solution(ranges=ranges, sample_pitch=sample_pitch, data_shape=None,
                    wavenumber=wavenumber, angular_frequency=angular_frequency,
                    vacuum_wavelength=vacuum_wavelength,
                    source_distribution=source_distribution, epsilon=epsilon, xi=xi, zeta=zeta, mu=mu,
                    initial_field=initial_field).solve(callback)


class Solution(object):
    def __init__(self, ranges=None, sample_pitch=None, data_shape=None,
                 wavenumber=None, angular_frequency=None, vacuum_wavelength=None,
                 source_distribution=None, epsilon=None, xi=0.0, zeta=0.0, mu=1.0, initial_field=0.0):
        """
        Class a solution that can be further iterated towards a solution for Maxwell's equations in a media specified by
        the epsilon, xi, zeta, and mu distributions.

        :param ranges: an array of vectors with uniformly increasing values that indicate the positions in a
            plaid grid at which to calculate the solution at. In the case of 1D, a simple vector may be provided.
            The length of the ranges determines the dataSize, which must match the dimensions of (the output of)
            source_distribution, epsilon, xi, zeta, mu, and the optional start E, unless these are singletons.
            This argument may be replaced by sample_pitch.
        :param sample_pitch: Optional, replaces ranges by zero-centered ranges with the given sample pitch.
        :param data_shape: Only required if ranges is not specified and all distributions are specified as functions.
        :param wavenumber: the wavenumber in vacuum = 2pi / vacuum_wavelength.
            The wavelength in the same units as used for the other inputs/outputs.
        :param angular_frequency: alternative argument to the wavenumber = angular_frequency / c
        :param vacuum_wavelength: alternative argument to the wavenumber = 2 pi / vacuum_wavelength
        :param source_distribution: an array or function that returns the (vectorial) source input wave distribution.
            The source values relate to the current density, J, as  1j*angularFrequency*scipy.constants.mu_0*J
            and has units of :math:`rad s^-1 H m^-1 A m^-2 = rad V m^{-3}`.
        :param epsilon: an array or function that returns the (tensor) epsilon that represents the permittivity at
            the points indicated by the ranges specified as its input arguments.
        :param xi: an array or function that returns the (tensor) xi for bi-(an)isotropy at the
            points indicated by the ranges specified as its input arguments.
        :param zeta: an array or function that returns the (tensor) zeta for bi-(an)isotropy at the
            points indicated by the ranges specified as its input arguments.
        :param mu: an array or function that returns the (tensor) permeability at the
            points indicated by the ranges specified as its input arguments.
        :param initial_field: optional start value for the E-field distribution (default: all zero E)
        """
        self.__iteration = 0

        if ranges is None:
            if data_shape is None:
                # Try to guess the data_shape from other input arrays
                if not callable(epsilon):
                    data_shape = epsilon.shape[2:]
                elif not callable(source_distribution):
                    data_shape = source_distribution.shape[2:]
                elif not callable(mu):
                    data_shape = mu.shape[2:]
                elif not callable(xi):
                    data_shape = xi.shape[2:]
                elif not callable(zeta):
                    data_shape = zeta.shape[2:]
                else:
                    log.error('If ranges and data_shape are not specified, the medium must be an ndarray, not a function.')

            ranges = utils.calc_ranges(data_shape, sample_pitch)
        else:
            if np.isscalar(ranges[0]):
                ranges = [ranges]  # This must be a one dimensional problem, the user forgot to specify the range in a list

        self.__grid = Grid.from_ranges(*ranges)
        self.__ranges = ranges
        self.__shape = np.array([len(r) for r in ranges])
        self.__sample_pitch = np.array([r[int(len(r) > 1)]-r[0] for r in ranges])

        if wavenumber is not None:
            self.__wavenumber = wavenumber
            if angular_frequency is not None or vacuum_wavelength is not None:
                log.warning('Using specified wavenumber = %d, ignoring angular_frequency and vacuum_wavelength.' %
                            self.__wavenumber)
        elif angular_frequency is not None:
            self.__wavenumber = angular_frequency / const.c
            if vacuum_wavelength is not None:
                log.warning('Using specified angular_frequency = %d, ignoring vacuum_wavelength.' %
                            angular_frequency)
        elif vacuum_wavelength is not None:
            self.__wavenumber = 2 * const.pi / vacuum_wavelength
        else:
            message = 'Error: no wavenumber, angular frequency, nor wavelength specified.'
            log.critical(message)
            raise Exception(message)

        self.__last_update_norm = np.inf

        self.__field_mat = None
        self.__chi_op = None
        self.__gamma_op = None

        # Convert functions to arrays.
        # This is not strictly necessary for the source, though it simplifies the code.
        def func2arr(f):
            if callable(f):
                return f(*self.__ranges)
            else:
                return f  # already an array

        source_distribution = func2arr(source_distribution)

        # Create an object to handle our parallel operations
        if source_distribution.shape[0] == 1:
            nb_pol_dims = 1
        else:
            nb_pol_dims = 3

        # Normalize the dimensions in the parallel operations to k0
        self.__PO = ParallelOperations(nb_pol_dims, self.shape, self.sample_pitch * self.wavenumber)

        # The following requires the self.__PO to be defined
        self.E = initial_field

        # Before the first iteration, the pre-conditioner must be determined and applied to the source and medium
        self.__prepare_preconditioner(
            source_distribution, func2arr(epsilon), func2arr(xi), func2arr(zeta), func2arr(mu),
            self.sample_pitch, self.wavenumber
        )
        # Now we can forget epsilon, xi, zeta, mu, and source_distribution. Their information is encapsulated
        # in the newly created operator methods __chi_op, and __gamma_op
        self.__residue = None  # Invalidate residue

    def __prepare_preconditioner(self, source_distribution, epsilon, xi, zeta, mu, sample_pitch, wavenumber):
        """
        Sets or updates the private value for self.__source, and the methods __chi_op, __gamma_op, __green_function_op
        This uses the values for source_distribution, epsilon, xi, zeta, mu, as well as
        the sample_pitch and wavenumber.

        :param source_distribution: A 1+N-D array representing the source vector field.
        :param epsilon: A 2+N-D array representing the permittivity distribution.
        :param xi: A 2+N-D array representing the xi bi-anistrotropy distribution.
        :param zeta: A 2+N-D array representing the zeta bi-anistrotropy distribution.
        :param mu: A 2+N-D array representing the permeability distribution.
        :param sample_pitch: A vector with numbers indicating the sample distance in each dimension.
        :param wavenumber: The wavenumber, k,  of the coherent illumination considered for this problem.
        """
        log.debug('Preparing pre-conditioner: determining alpha and beta...')

        # Convert all inputs to a canonical form
        epsilon = self.__PO.to_simple_matrix(epsilon)
        xi = self.__PO.to_simple_matrix(xi)
        zeta = self.__PO.to_simple_matrix(zeta)
        mu = self.__PO.to_simple_matrix(mu)

        # Determine if the media is magnetic
        self.__magnetic = np.any(xi.ravel() != 0.0) or np.any(zeta.ravel() != 0.0) or np.any(mu.ravel() != mu.ravel()[0])
        if self.magnetic:
            log.debug('Medium has magnetic properties.')
        else:
            log.debug('Medium has no magnetic properties. Using faster permittivity-only solver.')

        def largest_eigenvalue(a):
            return np.max(np.abs(self.__PO.mat3_eig(a)))

        def largest_singularvalue(a):
            return np.sqrt(largest_eigenvalue(self.__PO.mul(self.__PO.conjugate_transpose(a), a)))

        # Do a quick check to see if the the media has no gain
        def has_gain(a):
            return np.any(
                self.__PO.mat3_eig(-0.5j * (a - self.__PO.conjugate_transpose(a))).real < - 2*np.finfo(a.dtype).eps)

        if has_gain(epsilon) or has_gain(xi) or has_gain(zeta) or has_gain(mu):
            def max_gain(a):
                return np.max(-self.__PO.mat3_eig(-0.5j * (a - self.__PO.conjugate_transpose(a))).real)
            log.warning("Convergence not guaranteed!\n"
                        "Permittivity has a gain as large as %0.3g, xi up to %0.3g, zeta up to %0.3g,"
                        " and the permeability up to %0.3g." %
                        (max_gain(epsilon), max_gain(xi), max_gain(zeta), max_gain(mu))
                        )
        else:
            log.debug('Media has no gain, safe to proceed.')

        # determine alpha and beta for the given media
        # Determine mu^-2
        mu_inv = self.__PO.inv(mu)

        # Determine calcChiHH, calcSigmaHH
        if self.magnetic:
            def calc_chiHH(beta_): return self.__PO.subtract(1.0, mu_inv / beta_)

            mu_inv_transpose = self.__PO.conjugate_transpose(mu_inv)
            mu_inv2 = self.__PO.mul(mu_inv_transpose, mu_inv)  # Positive definite

            def calc_chiHH_beta2(beta_):
                return (mu_inv2 +
                        self.__PO.subtract(np.abs(beta_) ** 2, mu_inv_transpose * beta_ + mu_inv * np.conj(beta_))
                        )

            def calc_sigmaHH(beta_): return np.sqrt(largest_eigenvalue(calc_chiHH_beta2(beta_))) / abs(beta_)

            if has_gain(mu) or has_gain(xi) or has_gain(zeta):
                log.warning('Permeability or bi-(an)isotropy has gain. Convergence not guaranteed!')
            else:
                log.debug('Permeability and bi-(an)isotropy have no gain, safe to proceed.')
        else:
            # non-magnetic, mu is scalar and both xi and zeta are zero
            def calc_chiHH(beta_): return 1.0 - mu_inv / beta_  # always zero when beta == mu_inv

            def calc_sigmaHH(beta_): return abs(calc_chiHH(beta_))  # always zero when beta == mu_inv

        # Determine: calcChiEE, chiEHTheta, chiHETheta, chiHH, alpha, beta
        chiEH_beta = -1.0j * self.__PO.mul(xi, mu_inv)
        chiHE_beta = 1.0j * self.__PO.mul(mu_inv, zeta)

        # del zeta # Needed for conversion from E to H
        xi_mu_inv_zeta = self.__PO.mul(xi, -1.0j * chiHE_beta)
        del xi
        epsilon_xi_mu_inv_zeta = self.__PO.subtract(epsilon, xi_mu_inv_zeta)
        del epsilon, xi_mu_inv_zeta

        epsilon_xi_mu_inv_zeta_transpose = self.__PO.conjugate_transpose(epsilon_xi_mu_inv_zeta)
        epsilon_xi_mu_inv_zeta2 = self.__PO.mul(epsilon_xi_mu_inv_zeta_transpose, epsilon_xi_mu_inv_zeta)
        # The above must be positive definite

        def calc_DeltaEE_beta2(alpha_, beta_):
            return epsilon_xi_mu_inv_zeta2 + self.__PO.subtract(
                np.abs(alpha_.real * beta_) ** 2,
                epsilon_xi_mu_inv_zeta_transpose * (alpha_.real * beta_) + epsilon_xi_mu_inv_zeta * np.conj(alpha_.real * beta_)
                )

        def calc_sigmaEE(alpha_, beta_):
            return np.sqrt(largest_eigenvalue(calc_DeltaEE_beta2(alpha_, beta_))) / np.abs(beta_)

        # Determine alpha, beta and chiHH
        alpha_tolerance = 1e-4
        if self.magnetic:
            # Optimize the real part of alpha and beta
            sigmaD = np.linalg.norm(2.0*np.pi / (2.0*sample_pitch)) / wavenumber
            sigmaHE_beta = largest_singularvalue(chiHE_beta)
            sigmaEH_beta = largest_singularvalue(chiEH_beta)

            def max_singular_value_sum(eta_, beta_):
                return sigmaD**2 * calc_sigmaHH(beta_) +\
                       sigmaD * (sigmaEH_beta + sigmaHE_beta) / np.abs(beta_) + calc_sigmaEE(eta_, beta_)

            def beta_from_vec(alpha_beta_vec_): return alpha_beta_vec_[1] ** 2  # enforce positivity

            def target_function_vec(vec):
                beta = beta_from_vec(vec)
                return max_singular_value_sum(vec[0], beta) * beta

            log.debug('Finding optimal alpha and beta...')
            alpha_beta_vec = scipy.optimize.fmin(target_function_vec, [0, 1], initial_simplex=[[0, 1], [1, 1], [0, 0.9]],
                                                disp=False, full_output=False,
                                                ftol=alpha_tolerance, xtol=alpha_tolerance, maxiter=100, maxfun=200)
            self.__beta = beta_from_vec(alpha_beta_vec)
            alpha = alpha_beta_vec[0] + 1.0j * max_singular_value_sum(alpha_beta_vec[0], self.__beta)

            del beta_from_vec, alpha_beta_vec
        else:
            # non-magnetic
            self.__beta = mu_inv.flatten()[0].real  # Must be scalar and real

            def target_function_vec(alpha_):
                return calc_sigmaEE(alpha_, self.__beta)  # beta is fixed to mu_inv so that chi_HH==0

            log.debug('beta = %0.4g, finding optimal alpha...' % self.__beta)
            alpha_real, min_value = scipy.optimize.fmin(target_function_vec, 0.0, initial_simplex=[[0.0], [1.0]],
                                                      disp=False, full_output=True,
                                                      ftol=alpha_tolerance, xtol=alpha_tolerance, maxiter=100, maxfun=100)[:2]

            alpha = alpha_real + 1.0j * min_value
            log.debug('alpha = %0.4g + %0.4gi, or larger in the imaginary part' % (alpha.real, alpha.imag))

            del alpha_real

        # Store the modified source distribution
        self.__source = self.__PO.to_simple_matrix(source_distribution) / self.__beta  # Adjust for magnetic bias
        del source_distribution

        alpha = self.__increase_bias_to_limit_kernel_width(alpha)
        log.info('alpha = %0.4g + %0.4gi, beta = %0.4g' % (alpha.real, alpha.imag, self.__beta))

        log.info('Preparing pre-conditioned operators...')
        self.__chiEH = chiEH_beta / self.__beta
        del chiEH_beta
        self.__chiHE = chiHE_beta / self.__beta
        del chiHE_beta
        self.__chiHH = calc_chiHH(self.__beta)
        self.__chiEE_base = epsilon_xi_mu_inv_zeta / self.__beta

        # Update the operators stored as private attributes
        self.__update_operators(alpha)

    def __update_operators(self, alpha):
        """
        Updates the value of alpha, and the operators for chi, gamma, as well as the Green function

        :param alpha: The new value for alpha.

        returns Nothing. Side effect is setting of properties __alpha, __chi_dot, __gamma_dot, and __green_function_op
        """

        # Once the susceptibility_offset is fixed, we can also calculate chiEE
        self.__alpha = alpha
        # chiEE = self.__PO.subtract(self.__chiEE_base, self.__alpha)  # Safe some memory by not replicating this

        # Pick the right Chi operator
        if self.magnetic:
            def chi_op(E):
                """
                Applies the magnetic :math:`\Chi` operator to the input E-field.

                :param E: an array representing the E to apply the Chi operator to.
                :return: an array with the result E of the same size as E or of the size of its singleton expansion.
                """
                def D(field_E): return self.__PO.curl(field_E)  # includes k0^-1 by the definition of __PO

                # chiE = self.__PO.mul(self.__chiEE_base, E) - self.__alpha * E + self.__PO.mul(self.__chiHE, ED)
                # chiH = self.__PO.mul(self.__chiEH, E) + self.__PO.mul(self.__chiHH, ED)
                chiE = self.__PO.mul(self.__chiEE_base, E)
                chiE -= self.__alpha * E
                chiH = self.__PO.mul(self.__chiEH, E)
                ED = D(E)
                chiE += self.__PO.mul(self.__chiHE, ED)
                chiH += self.__PO.mul(self.__chiHH, ED)

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
                result = self.__PO.mul(self.__chiEE_base, E)
                result -= self.__alpha * E

                return result  # self.__PO.mul(self.__chiEE_base, E) - self.__alpha * E

        # Now create the Green function operator
        # Calculate the convolution filter just once
        g_scalar_ft = 1.0 / (self.__PO.calc_K2() - self.__alpha)
        if self.__PO.vectorial:
            def g_ft_op(FFt):  # No need to represent the full matrix in memory
                PiL_FFt = self.__PO.longitudinal_projection_ft(FFt)  # Creates K^2 on-the-fly and still memory intensive
                result = self.__PO.subtract(FFt, PiL_FFt)
                result *= g_scalar_ft
                result -= PiL_FFt / self.__alpha
                return result  # g_scalar_ft * self.__PO.subtract(FFt, PiL_FFt) - PiL_FFt / self.__alpha
        else:
            def g_ft_op(FFt):
                result = FFt
                result *= g_scalar_ft
                return result  # g_scalar_ft * FFt

        def dyadic_green_function_op(F):
            FFt = self.__PO.ft(F)  # Convert each component separately to frequency coordinates
            FFt = g_ft_op(FFt)  # Memory intensive
            return self.__PO.ift(FFt)  # Back to spatial coordinates

        def gamma_op(E):
            result = self.__chi_op(E)
            result *= 1.0j / self.__alpha.imag
            # Long description:
            # B = self.__PO.curl(result) / (1j * self.wavenumber)
            # H = mu_inv = B / const.mu_0
            # D = (J - self.__PO.curl(H)) / (1j * self.wavenumber)
            # result = self.__PO.inv(epsilon, D) / const.epsilon_0
            return result  # (1.0j / self.__alpha.imag) * self.__chi_op(E)

        # Update the methods for the operators Chi and Gamma:
        self.__chi_op = chi_op
        self.__gamma_op = gamma_op
        # Set the Green function
        self.__green_function_op = dyadic_green_function_op

    def __increase_bias_to_limit_kernel_width(self, alpha, max_kernel_width_in_px=None, max_kernel_residue=1.0 / 100):
        """
        Limit the kernel size by increasing alpha so that:
        -log(max_kernel_residue)/(imag(sqrt(central_permittivity + 1i*alpha))) == max_kernel_radius_in_rad

            -log(max_kernel_residue) / max_kernel_radius_in_rad == imag(sqrt(central_permittivity + 1i*alpha)) == B
            B^4 + central_permittivity*B^2 - alpha^2/4 == 0
            -central_permittivity +- sqrt(central_permittivity^2 + alpha^2) == 2 * B^2
            -central_permittivity +- sqrt(central_permittivity^2 + alpha^2) == 2*(-log(max_kernel_residue) /
                max_kernel_radius_in_rad)^2
        Increase offset if needed to restrict kernel size.

        :param alpha: the complex constant in the pre-conditioner to ensure convergence
        :param max_kernel_width_in_px: the target kernel width
        :param max_kernel_residue: the estimated error outside the target kernel box
        :return: the adapted alpha value
        """
        if max_kernel_width_in_px is None:
            max_kernel_width_in_px = self.shape / 4.0

        max_kernel_radius_in_rad = np.min(max_kernel_width_in_px / self.sample_pitch) / 2.0 / self.wavenumber

        central_permittivity = alpha.real
        susceptibility_offset = alpha.imag
        min_susceptibility_offset = np.sqrt(
            np.maximum(0.0,
                       (2.0*(-np.log(max_kernel_residue) / max_kernel_radius_in_rad)**2 + central_permittivity)**2 -
                       central_permittivity**2)
        )
        susceptibility_offset = np.maximum(min_susceptibility_offset, susceptibility_offset)

        return alpha.real + 1.0j * susceptibility_offset

    @property
    def ranges(self):
        """
        The sample positions of the plaid sampling grid.
        This may be useful for displaying result axes.

        :return: A vector of monotonously increasing vectors, one per dimension,
            indicating the sampling points for each dimension.
        """
        return self.__ranges

    @property
    def grid(self):
        """
        The sample positions of the plaid sampling grid.
        This may be useful for displaying result axes.

        :return: A Grid object representing the sample points of the fields and material.
        """
        return self.__grid

    @property
    def dtype(self):
        return self.__field_mat.dtype

    @property
    def shape(self):
        """
        The shape of the sample volume in pixels.

        :return: A vector with 1, 2, or 3 elements, representing the number of samples per dimension.
        """
        return self.__shape

    @property
    def sample_pitch(self):
        """
        The distance between voxels per dimension (in m x m x m).

        :return: A vector with 1, 2, or 3 elements, representing the physical distance between samples.
        """
        return self.__sample_pitch

    @property
    def volume(self):
        """
        The physical dimensions of the sample volume (in m x m x m).

        :return: A vector with 1, 2, or 3 elements, representing the physical size of the sample volume.
        """
        return self.sample_pitch * self.shape

    @property
    def wavenumber(self):
        """
        The vacuum wavenumber, :math:`k_0`, used in the calculation.

        :return: A scalar indicating the wavenumber used in the calculation.
        """
        return self.__wavenumber

    @property
    def angular_frequency(self):
        """
        The angular frequency, :math:`\omega`, used in the calculation.

        :return: A scalar indicating the angular frequency used in the calculation.
        """
        return self.__wavenumber * const.c

    @property
    def wavelength(self):
        """
        The vacuum wavelength, :math:`\lambda_0`, used in the calculation.

        :return: A scalar indicating the vacuum wavelength used in the calculation.
        """
        return 2.0 * const.pi / self.__wavenumber

    @property
    def magnetic(self):
        """
        Indicates if this media is considered magnetic.

        :return: A boolean, True when magnetic, False otherwise.
        """
        return self.__magnetic

    @property
    def j(self):
        """
        The current density, j, of the source vector field.

        :return: A complex array indicating the amplitude and phase of the source vector field [A m^-2].
            The dimensions of the array are [1|3, self.shape], where the first dimension is 1 in case of a scalar field.
        """
        source_distribution = self.__source * self.__beta
        current_density = source_distribution / (1.0j * self.angular_frequency * const.mu_0)

        return current_density[:, 0, ...]

    @property
    def E(self):
        """
        The electric field for every point in the sample space (SI units).

        :return: A vector array with the first dimension containing Ex, Ey, and Ez,
            while the following dimensions are the spatial dimensions.
        """
        return self.__field_mat[:, 0, ...] / (self.wavenumber**2)

    @E.setter
    def E(self, E):
        """
        The electric field for every point in the sample space (SI units).

        :param E: The new field. A vector array with the first dimension containing :math:`E_x, E_y, and E_z`,
            while the following dimensions are the spatial dimensions.
        """
        self.__field_mat = self.__PO.to_simple_vector(E+0.0j) * (self.wavenumber**2)

    @property
    def B(self):
        """
        The magnetic field for every point in the sample space (SI units).
        This is calculated from H and E.

        :return: A vector array with the first dimension containing :math:`B_x, B_y, and B_z`,
            while the following dimensions are the spatial dimensions.
        """
        B = self.__PO.curl(self.E[:, np.newaxis, ...]) / (1.0j * const.c)  # curl includes k0 by definition of __PO

        return B[:, 0, ...]

    @property
    def D(self):
        """
        The displacement field for every point in the sample space (SI units).
        This is calculated from E and H.

        :return: A vector array with the first dimension containing :math:`D_x, D_y, and D_z`,
            while the following dimensions are the spatial dimensions.
        """
        J = (self.__beta / (1.0j * self.angular_frequency * const.mu_0)) * self.__source
        D = (J - self.__PO.curl(self.H[:, np.newaxis, ...]) * self.wavenumber) / (1.0j * self.angular_frequency)  # curl includes k0 by definition of __PO
        # epsilon = self.__chiEE ...
        # D = const.epsilon_0 * self.__PO.mul(epsilon, self.E[:, np.newaxis, ...])\
        #     + (self.__PO.mul(xi, self.H[:, np.newaxis, ...]) / const.c)

        return D[:, 0, ...]

    @property
    def H(self):
        """
        The magnetizing field for every point in the sample space (SI units).
        This is calculated from E.

        :return: A vector array with the first dimension containing :math:`H_x, H_y, and H_z`,
            while the following dimensions are the spatial dimensions.
        """
        if self.magnetic:
            # Use stored matrices to safe the space
            mu_inv = self.__PO.subtract(1.0, self.__chiHH) * self.__beta

            # the curl in the following includes factor k0^-1 by the definition of __PO above
            H = (1.0j / (const.mu_0 * const.c)) * (
                - self.__PO.mul(mu_inv, self.__PO.curl(self.E[:, np.newaxis, ...]))
                + self.__beta * self.__PO.mul(self.__chiHE, self.E[:, np.newaxis, ...])
            )
        else:
            mu_inv = (1.0 - self.__chiHH.flatten()[0]) * self.__beta
            mu_H = (-1.0j / (const.mu_0 * const.c)) * self.__PO.curl(self.E[:, np.newaxis, ...])  # includes k0^-1 by the definition of __PO
            H = mu_inv * mu_H

        return H[:, 0, ...]

    @property
    def S(self):
        """
        The time-averaged Poynting vector for every point in space.

        :return: A vector array with the first dimension containing :math:`S_x, S_y, and S_z`,
            while the following dimensions are the spatial dimensions.
        """
        E = self.E[:, np.newaxis, ...]
        H = self.H[:, np.newaxis, ...]
        poynting_vector = 0.5 * self.__PO.cross(E, np.conj(H)).real

        return poynting_vector[:, 0, ...]

    @property
    def energy_density(self):
        """
        R the energy density u

        :return: A real array indicating the energy density in space.
        """
        E = self.E
        B = self.B  # Can be calculated more efficiently from E, though avoiding code-replication for now.
        H = self.H  # Can be calculated more efficiently from B and E, though avoiding code-replication for now.
        D = self.D  # Can be calculated more efficiently from H, though avoiding code-replication for now.

        u = np.sum((E * np.conj(D)).real, axis=0)
        u += np.sum((B * np.conj(H)).real, axis=0)
        u *= 0.5 * 0.5  # 0.5 * (E.D' + B.H'), the other 0.5 is because we time-average the products of real functions

        return u

    @property
    def stress_tensor(self):
        """
        Maxwell's stress tensor for every point in space.

        :return: A real and symmetric matrix-array with the stress tensor for every point in space.
            The units are :math:`N / m^2`.
        """
        E = self.E[:, np.newaxis, ...]
        E2 = np.sum(np.abs(E) ** 2, axis=0)
        result = const.epsilon_0 * self.__PO.outer(E, E)

        H = self.H[:, np.newaxis, ...]
        H2 = np.sum(np.abs(H) ** 2, axis=0)
        result += self.__PO.outer(H, H) * const.mu_0

        result -= (0.5 * self.__PO.eye) * (const.epsilon_0 * E2 + H2 * const.mu_0)
        result = 0.5 * result  # TODO: Do we want the Abraham or Minkowski form?

        return result.real

    @property
    def f(self):
        """
        The electromagnetic force density (force per SI unit volume, not per voxel).

        :return: A vector array representing the electro-magnetic force exerted per unit volume.
            The first dimension contains :math:`f_x, f_y, and f_z`, while the following dimensions are the spatial dimensions.
            The units are :math:`N / m^3`.
        """
        # Could be written more efficiently by either caching H or explicitly calculating the stress tensor
        # in this method. Leaving this for now, so to avoid code replication.
        force = self.__PO.div(self.stress_tensor).real * self.wavenumber  # The parallel operations is pre-scaled by k0
        # The time derivative of the Poynting vector averages to zero.
        # Make sure to remove imaginary part which must be due to rounding errors

        return force[:, 0, ...]

    @property
    def torque(self):
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
        # torque *= utils.vector_to_axis(self.sample_pitch, n=self.__PO.nb_dims, axis=1)
        # torque = np.sum(torque, axis=1)

        return torque

    @property
    def iteration(self):
        """
        The current iteration number.

        :return: An integer indicating the how many iterations have been done.
        """
        return self.__iteration

    @iteration.setter
    def iteration(self, it=0):
        """
        The current iteration number.
        Resets the iteration count or sets it to a specified integer.
        This does not affect the calculation, only (potentially) the stop criterion.

        :param it: (optional) the new iteration number
        """
        self.__iteration = it

    @property
    def last_update_norm(self):
        """
        The L2-norm of the last update, the difference between current and previous E-field.

        :return: A positive scalar indicating the norm of the last update.
        """
        return self.__last_update_norm / (self.wavenumber**2)

    @property
    def residue(self):
        """
        Returns the current residue.

        :return: A non-negative real scalar that indicates the change in E with the previous iteration
            normalized to the norm of the current E.
        """
        if self.__residue is None:
            self.__residue = self.__last_update_norm / np.linalg.norm(self.__field_mat)

        return self.__residue

    def __iter__(self):
        """
        Performs one iteration and returns a reference to this object representing the next state.

        :return: a reference it itself after doing one iteration
        """
        while True:
            self.iteration += 1
            log.debug('Starting iteration %d...' % self.iteration)
            self.__residue = None  # Invalidate residue

            # Calculate update
            #d_field = self.__gamma_op(self.__green_function_op(self.__chi_op(self.__field_mat) + self.__source) - self.__field_mat)
            d_field = self.__chi_op(self.__field_mat)
            d_field = self.__PO.add(d_field, self.__source)
            d_field = self.__green_function_op(d_field)
            d_field -= self.__field_mat
            d_field = self.__gamma_op(d_field)

            # Check if the iteration is diverging
            previous_update = self.__last_update_norm
            current_update_norm = np.linalg.norm(d_field)
            relative_update_norm = current_update_norm / previous_update
            if np.isfinite(relative_update_norm) and relative_update_norm < 1.0:
                log.debug('The field update is scaled by %0.3f < 1.' % relative_update_norm)
                # Update solution
                if np.all(self.__field_mat.shape == d_field.shape):
                    self.__field_mat += d_field
                else:
                    # The initial field can be a scalar constant such as 0.0
                    d_field += self.__field_mat
                    self.__field_mat = d_field
                    del d_field  # corrupted by previous operations
                # Keep update norm for next iteration's convergence check
                self.__last_update_norm = current_update_norm

                log.debug('Updated field in iteration %d.' % self.iteration)
            else:
                log.warning('The field update is scaled by %0.3f >= 1, so the maximum singular value of the update matrix',
                         ' is larger than one. Convergence issues may occur.' % relative_update_norm)
                log.info('Increasing the imaginary part of alpha from %0.3g...' % self.__alpha.imag)
                self.__update_operators(self.__alpha.real + 1.10j * self.__alpha.imag)
                log.info('to %0.3g.' % self.__alpha.imag)

                log.debug('Aborting field update in iteration %d.' % self.iteration)

            yield self

    def solve(self, callback=None):
        """
        Runs the algorithm until the convergence criterion is met or until the maximum number of iterations is reached.

        :param callback: optional callback function that overrides the one set for the solver.
            E.g. callback=lambda s: s.iteration < 100
        :return: an array containing the final field E
        """
        if callback is None:
            def callback(s):
                return s.iteration < 1e4 and s.residue > 1e-4

        for sol in self:
            # sol is the current iteration result
            # now execute the user-specified callback function
            if not callback(sol):  # function may have side effects
                log.debug('Convergence target met, stopping iteration.')
                break  # stop the iteration if it returned False

        return self
