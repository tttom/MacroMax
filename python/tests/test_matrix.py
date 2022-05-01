import unittest
import numpy.testing as npt

import numpy as np

from macromax.utils.array import Grid
from macromax.utils import ft
from macromax.bound import LinearBound
from macromax import matrix


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.wavelength = 500e-9
        self.k0 = 2 * np.pi / self.wavelength
        self.grids = (
                      # Grid([256, 128], self.wavelength/4),
                      Grid([256, 7], self.wavelength/4),  # Avoid standing waves that travel parallel to the slab by making wavelength not fit an integer number of times.
                      Grid([256, 5, 3], self.wavelength/4),
                      Grid([256, 8], [self.wavelength/4, self.wavelength/5]),
                      Grid([256], self.wavelength/4),
                      )

        # Define bounds for each grid
        self.bounds = []
        for grid in self.grids:
            thickness = np.zeros([grid.ndim, 1])
            thickness[0, 0] = 20 * self.wavelength  # Periodic boundaries in transverse
            bound = LinearBound(grid, thickness=thickness, max_extinction_coefficient=0.25)
            # bound = InfiniBound(grid, thickness=thickness)  # TODO: try with a grid of shape [256, 7]
            self.bounds.append(bound)

    def test_srcvec2freespace(self):
        for vectorial in (False, True):
            desc = 'vectorial' if vectorial else 'scalar'
            for grid, bound in zip(self.grids, self.bounds):
                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, refractive_index=1, bound=bound)
                for _ in range(m.shape[1]):
                    input_vector = np.zeros(m.shape[1], dtype=np.complex128)
                    input_vector[_] = 1 + 1j

                    # Input basis vector properties
                    amplitude = input_vector[_]
                    side_idx = _ // (input_vector.size // 2)
                    angle_idx = _ % (input_vector.size // 2)

                    if vectorial:
                        pol_idx = _ % 2
                        angle_idx = angle_idx // 2
                        unit_mode_pol = np.zeros(2, dtype=np.complex)
                        unit_mode_pol[pol_idx] = 1

                    # Determine k-vector direction. The angle_idx is an index in all the propagating modes of one polarization
                    # First, calculate all k-vectors, then select the one we are interested in
                    k_rel_transverse = np.zeros([2, *grid.shape[1:]])
                    for transverse_pol_idx in range(1, grid.ndim):
                        k_rel_transverse[transverse_pol_idx-1] = grid.k.as_origin_at_center[transverse_pol_idx] / self.k0
                    propagating = np.sqrt(np.sum(k_rel_transverse**2, axis=0)) + np.sqrt(np.finfo(m.dtype).eps) < 1
                    k_rel_transverse = k_rel_transverse[:, propagating]  # keep only the propagating modes
                    k_rel_transverse = k_rel_transverse[:, angle_idx]  # Select the one indicated by angle_idx
                    k_vector_dir = np.array([np.sqrt(1 - np.sum(k_rel_transverse**2)), *k_rel_transverse])  # complete the k-vector direction arrow with the axial component

                    fld = m.srcvec2freespace(input_vector)

                    npt.assert_equal(fld.shape, (1 + 2 * vectorial, *grid.shape), err_msg=f'Field shape not correct at mode {_} for {desc} {grid}.')
                    field_at_origin = fld.reshape(fld.shape[0], -1)[:,
                                      np.ravel_multi_index(np.array(fld.shape[1:]) // 2, fld.shape[1:])]
                    expected_at_origin = np.linalg.norm(input_vector) / np.sqrt(k_vector_dir[0])  # larger k-vectors correspond to larger fields to compensate for the non-normal transmission through a fixed transverse voxel
                    npt.assert_almost_equal(np.linalg.norm(field_at_origin), expected_at_origin,
                                            err_msg=f'Field amplitude incorrect at origin for mode {_} for {desc} {grid}.')
                    # Compensate for angle of incidence. The field vector corresponds to the longitudinal propagation,
                    # if the field propagates transversally it must have a larger amplitude to compensate.

                    if vectorial:
                        radial_mode_norm = np.sqrt(np.sum(k_rel_transverse**2, axis=0))  # in 2D (transverse only)
                        radial_mode_dir = k_rel_transverse / (radial_mode_norm + (radial_mode_norm == 0))
                        radial_mode_pol_norm = np.dot(radial_mode_dir, unit_mode_pol)
                        radial_mode_pol = radial_mode_dir * radial_mode_pol_norm
                        azimuthal_pol = unit_mode_pol - radial_mode_pol
                        azimuthal_pol = np.array([0, *azimuthal_pol])  # convert from mode to 3D polarization
                        radial_pol = np.array([-np.sqrt(1.0 - k_vector_dir[0]**2) * radial_mode_pol_norm,
                                               *(radial_mode_pol * k_vector_dir[0])])
                        expected_at_origin = amplitude * (radial_pol + azimuthal_pol)
                    else:
                        expected_at_origin = amplitude
                    # Compensate for angle of incidence. The field vector corresponds to the longitudinal propagation,
                    # if the field propagates transversally it must have a larger amplitude to compensate.
                    expected_at_origin /= np.sqrt(k_vector_dir[0])

                    # Check center of field
                    npt.assert_almost_equal(field_at_origin, expected_at_origin, err_msg=f'Field at mode {_} incorrect at origin for {desc} {grid}.')

                    # Check complete field
                    expected_at_origin = np.atleast_1d(expected_at_origin)
                    while expected_at_origin.ndim < grid.ndim + 1:
                        expected_at_origin = expected_at_origin[:, np.newaxis]
                    expected = expected_at_origin * np.exp(
                        (1 - side_idx*2) * 1j * self.k0 * sum(grid[_] * k_vector_dir[_] for _ in range(grid.ndim)))
                    npt.assert_array_almost_equal(np.abs(fld), np.abs(expected), err_msg=f'Absolute value of field at mode {_} incorrect for {desc} {grid}.')
                    npt.assert_array_almost_equal(fld, expected, err_msg=f'Field at mode {_} incorrect for {desc} {grid}.')

    def test_srcvec2source(self):
        for vectorial in (False, True):
            desc = 'vectorial' if vectorial else 'scalar'
            for grid, bound in zip(self.grids, self.bounds):
                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, refractive_index=1, bound=bound)
                for dir_idx in range(m.shape[1]):
                    input_vector = np.zeros(m.shape[1], dtype=np.complex128)
                    input_vector[dir_idx] = 1 + 1j
                    freespace_fld = m.srcvec2freespace(input_vector)
                    src = m.srcvec2source(input_vector)

                    intensity = np.sum(np.sum(np.abs(src)**2, axis=0).reshape(src.shape[1], -1), -1)
                    source_indices = np.where(np.logical_not(np.isclose(intensity, 0.0)))[0]
                    npt.assert_equal(source_indices.size, 1, err_msg=f'Found {source_indices.size} source planes, expected 1 for {desc} {grid}.')

                    # Correct for tilted wavefront with respect to source plane
                    forward_fraction = np.sqrt(np.maximum(1.0 - sum((grid.k[_]/self.k0)**2 for _ in range(1, grid.ndim)), 0.0))
                    source2field = 1j * grid.step[0] / (2 * self.k0 * forward_fraction + np.isclose(forward_fraction, 0.0))
                    field_from_source = ft.ifftn(ft.fftn(src, axes=1+np.arange(grid.ndim)) * source2field, axes=1+np.arange(grid.ndim))

                    npt.assert_array_almost_equal(field_from_source[:, source_indices], freespace_fld[:, source_indices],
                                                  err_msg=f'Plane wave source at angle {dir_idx} different from free space plane wave distribution for {desc} {grid}.')

    def test_srcvec2det_field(self):
        for vectorial in (False, True):
            desc = ('vectorial' if vectorial else 'scalar') + ' free space plane wave distribution'
            for grid, bound in zip(self.grids, self.bounds):
                freespace_indices = np.where(np.logical_and(
                    grid.first[0] + bound.thickness[0, 0] + 4*self.wavelength < grid[0].ravel(),
                    grid[0].ravel() < grid[0].flatten()[-1] - bound.thickness[0, -1] - 4*self.wavelength)
                )[0]
                npt.assert_equal(len(freespace_indices) > 2, True, err_msg='Insufficient free space slices for {grid}.')

                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, refractive_index=1, bound=bound)
                for _ in range(m.shape[1]):
                    input_vector = np.zeros(m.shape[1], dtype=np.complex128)
                    input_vector[_] = 1 + 1j
                    freespace_fld = m.srcvec2freespace(input_vector)
                    fld = m.srcvec2det_field(input_vector)

                    if vectorial:
                        for pol_axis in range(3):
                            npt.assert_array_almost_equal(np.abs(fld[pol_axis, freespace_indices]), np.abs(freespace_fld[pol_axis, freespace_indices]), decimal=2,
                                                          err_msg=f'Absolute value of calculated plane wave field at mode {_}, polarization {pol_axis} different from {desc} for {grid}.')
                    else:
                        npt.assert_array_almost_equal(np.abs(fld[:, freespace_indices]), np.abs(freespace_fld[:, freespace_indices]), decimal=2,
                                                      err_msg=f'Calculated plane wave field absolute value at angle {_} different source field amplitude for {desc} {grid}.')
                    npt.assert_array_almost_equal(np.abs(fld[:, freespace_indices]), np.abs(freespace_fld[:, freespace_indices]), decimal=2,
                                                  err_msg=f'Calculated plane wave absolute value at angle {_} different from {desc} for {grid}.')
                    npt.assert_array_almost_equal(fld[:, freespace_indices], freespace_fld[:, freespace_indices], decimal=2,
                                                  err_msg=f'Calculated plane wave field at angle {_} different from {desc} for {grid}.')

    def test_det_field2detvec(self):
        for vectorial in (False, True):
            desc = 'vectorial' if vectorial else 'scalar'
            for grid, bound in zip(self.grids, self.bounds):
                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, refractive_index=1, bound=bound)
                for angle_idx in range(m.shape[1]):
                    input_vector = np.zeros(m.shape[1], dtype=np.complex128)
                    input_vector[angle_idx] = 1 + 1j
                    outward_fld = 0.0
                    for side in range(2):
                        input_mask = np.zeros_like(input_vector)  # one side at a time so we can separate inward/outward
                        if side == 0:
                            input_mask[:input_mask.shape[0]//2] = 1
                        else:
                            input_mask[input_mask.shape[0]//2:] = 1
                        freespace_fld = m.srcvec2freespace(input_vector * input_mask)
                        output_mask = np.zeros(freespace_fld.shape[1:])  # mask the outward fields
                        if side == 0:
                            output_mask[output_mask.shape[0]//2:, ...] = 1
                        else:
                            output_mask[:output_mask.shape[0]//2, ...] = 1
                        outward_fld = outward_fld + freespace_fld * output_mask

                    # Determine the vector for the output field
                    output_vector = m.det_field2detvec(outward_fld)
                    npt.assert_array_almost_equal(np.linalg.norm(output_vector), np.linalg.norm(input_vector), decimal=8,
                                                  err_msg=f'Norm of output vector not as expected for input vector at angle {angle_idx} for {desc} {grid}.')
                    npt.assert_array_almost_equal(np.abs(output_vector), np.abs(input_vector), decimal=8,
                                                  err_msg=f'Absolute values of output vector not as expected for input vector at angle {angle_idx} for {desc} {grid}.')
                    npt.assert_array_almost_equal(output_vector, input_vector, decimal=8,
                                                  err_msg=f'Output vector {output_vector} not as expected for input vector at angle {angle_idx} for {desc} {grid}.')

    def test_detvec2srcvec(self):
        for vectorial in (False, True):
            desc = 'vectorial' if vectorial else 'scalar'
            for grid, bound in zip(self.grids, self.bounds):
                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, refractive_index=1, bound=bound)
                # Create a field
                k_vector = np.array([1, 0, 0]) * self.k0
                incident_fld = np.exp(1j * sum(-np.abs(rng * k) for rng, k in zip(grid, k_vector)))[np.newaxis]
                emanating_fld = incident_fld.conj()
                top_mask = grid[0] < 0
                bottom_mask = grid[0] > 0
                for side_mask in top_mask, bottom_mask:
                    polarization = np.array([0, 1, 0]) if vectorial else np.array([1])
                    for _ in grid:
                        polarization = polarization[..., np.newaxis]
                    emanating_fld = polarization * emanating_fld
                    detvec = m.det_field2detvec(emanating_fld * side_mask)  # Describe as detection vector
                    srcvec = m.detvec2srcvec(detvec)  # Convert to source vector
                    reversed_fld = m.srcvec2freespace(srcvec)  # Converted field
                    # Compare
                    npt.assert_array_almost_equal(reversed_fld[:, side_mask.ravel()].conj(), emanating_fld[:, side_mask.ravel()],
                                                  err_msg=f'{desc} calculation not the same for light at the transverse plane at the origin.')

    def test_matmul(self):
        for vectorial in (False, True):
            desc = ('vectorial' if vectorial else 'scalar') + ' multiplication'
            for grid, bound in zip(self.grids, self.bounds):
                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, refractive_index=1, bound=bound)
                input_vec = np.zeros(m.shape[1], dtype=m.dtype)
                npt.assert_array_equal(m @ input_vec, input_vec, err_msg=f'{desc} with all zeros is not 0 for {grid}.')
                for _ in range(m.shape[1]):
                    input_vec *= 0
                    input_vec[_] = 1 + 1j
                    product = m @ input_vec

                    npt.assert_almost_equal(np.linalg.norm(product), np.linalg.norm(input_vec), decimal=2, err_msg=f'Norm of {desc} with 1 at index {_} not correct for {grid}.')
                    npt.assert_array_almost_equal(np.abs(product), np.abs(input_vec), decimal=2, err_msg=f'Absolute {desc} with 1 at index {_} not correct for {grid}.')
                    npt.assert_array_almost_equal(product, input_vec, decimal=2, err_msg=f'{desc} with 1 at index {_} not correct for {grid}.')

    def test_identity(self):
        for vectorial in (False, True):
            desc = 'vectorial' if vectorial else 'scalar'
            for grid, bound in zip(self.grids, self.bounds):
                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, bound=bound)
                expected = np.eye(*m.shape)
                vec = np.zeros(m.shape[1])

                det_field = m.srcvec2det_field(vec)
                detvec = m.det_field2detvec(det_field)
                npt.assert_array_almost_equal(np.abs(m), expected, decimal=2,
                                              err_msg=f'Absolute value of empty space {desc} scattering matrix not correct for {grid}.')
                npt.assert_array_almost_equal(m, expected, decimal=2,
                                              err_msg=f'Empty space {desc} scattering matrix not correct for {grid}.')

    def test_unitarity(self):
        def lowpass(grid: Grid, arr: np.ndarray, min_size: float) -> np.ndarray:
            lowpass_filter = sum(grid.f[_]**2 for _ in range(grid.ndim)) < min_size ** (-2)
            return ft.ifftn(ft.fftn(arr) * lowpass_filter)

        for vectorial in (False, True):
            desc = 'vectorial' if vectorial else 'scalar'
            for grid, bound in zip(self.grids, self.bounds):
                # Test if a thin film with multiple layers acts as a unitary scatterer
                n = np.ones(grid.shape)
                n[grid.shape[0]//2 - 11] = 1.5
                n[grid.shape[0]//2 - 10] = 1.5
                n[grid.shape[0]//2 - 9] = 1.5
                n[grid.shape[0]//2 - 8] = 1.5
                n[grid.shape[0]//2 - 1] = 1.5
                n[grid.shape[0]//2] = 1.5
                n[grid.shape[0]//2 + 2] = 1.5
                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, refractive_index=n, bound=bound)
                base_vector_lengths = np.sqrt(np.sum(np.abs(m)**2, axis=0))
                npt.assert_almost_equal(base_vector_lengths, 1.0, decimal=2,
                                        err_msg=f'Thin film scattering matrix does not have unit columns {desc} {grid}.')
                u, singular_values, vt = np.linalg.svd(m)
                npt.assert_almost_equal(singular_values, 1.0, decimal=2,
                                        err_msg=f'Thin film scattering matrix is not unitary for {desc} {grid}.')

                # Test if a random sample without absorption acts as a unitary scatterer
                # Generate a random scattering refractive index
                rng = np.random.RandomState(seed=1)
                layer_shape = grid.shape
                layer_shape[0] = 8*self.wavelength // grid.step[0]
                # # Random noise
                # material_n = np.ones(grid.shape)
                # material_n[grid.shape[0]//2 + np.arange(layer_shape[0]) - layer_shape[0]//2] = rng.uniform(1.0, 1.5, layer_shape)
                # Random spheres
                material_n = np.zeros(grid.shape, dtype=bool)
                while np.mean(material_n) < 0.75:
                    center = rng.rand(grid.ndim) * grid.extent + grid.first
                    radius = rng.uniform(0.75, 1.5) * self.wavelength / 2
                    sphere = sum((x - c)**2 for x, c in zip(grid, center)) <= radius ** 2
                    material_n = np.logical_or(material_n, sphere)
                material_n = 1 + 0.5 * material_n
                # material_n = lowpass(grid, material_n, self.wavelength/2).real
                n = np.ones(grid.shape)  # set background refractive index to 1
                n[grid.shape[0]//2 + np.arange(layer_shape[0]) - layer_shape[0]//2] = material_n[grid.shape[0]//2 + np.arange(layer_shape[0]) - layer_shape[0]//2]
                m = matrix.ScatteringMatrix(grid, vectorial=vectorial, vacuum_wavelength=self.wavelength, refractive_index=n, bound=bound)
                base_vector_lengths = np.sqrt(np.sum(np.abs(m)**2, axis=0))
                npt.assert_almost_equal(base_vector_lengths, 1.0, decimal=2,
                                        err_msg=f'Random scattering matrix does not have unit columns {desc} {grid}.')
                u, singular_values, vt = np.linalg.svd(m)
                npt.assert_almost_equal(singular_values, 1.0, decimal=2,
                                        err_msg=f'Random scattering matrix is not unitary for {desc} {grid}.')

                # # check if open transmission channels
                # transmission_matrix = m.forward_transmission
                # _, transmission_singular_values, vt = np.linalg.svd(transmission_matrix)
                # closed_channel_idx = np.argmin(transmission_singular_values)
                # closed_channel_in = np.concatenate([vt[closed_channel_idx].conj(), np.zeros(vt.shape[0])])
                # closed_channel_out = m @ closed_channel_in
                # closed_channel_out[closed_channel_out.size//2:] = 0  # ignore back reflected
                # open_channel_idx = np.argmax(transmission_singular_values)
                # open_channel_in = np.concatenate([vt[open_channel_idx].conj(), np.zeros(vt.shape[0])])
                # open_channel_out = m @ open_channel_in
                # open_channel_out[open_channel_out.size//2:] = 0  # ignore back reflected
                # print(f'closed channel theory {transmission_singular_values[closed_channel_idx]:0.6f}, actual {np.linalg.norm(closed_channel_out):0.6f}  for {desc} on {grid.shape} (n_avg {np.mean(n.ravel()):0.6f}).')
                # print(f'  open channel theory {transmission_singular_values[open_channel_idx]:0.6f}, actual {np.linalg.norm(open_channel_out):0.6f}  for {desc} on {grid.shape} (n_avg {np.mean(n.ravel()):0.6f}).')

    #
    # Tests of scattering <-> transfer matrix conversion functionality
    #

    def test_convert_scalar(self):
        rr_lr = np.atleast_2d(0.9)
        ll_lr = np.atleast_2d(0.1)
        rr_rl = np.atleast_2d(0.2)
        ll_rl = np.atleast_2d(0.8)
        s = np.array([[rr_lr, rr_rl], [ll_lr, ll_rl]]).transpose((0, 2, 1, 3)).reshape(2 * np.asarray(rr_lr.shape))
        rl_ll = np.linalg.inv(ll_rl)
        rl_lr = - rl_ll @ ll_lr
        rr_ll = rr_rl @ rl_ll
        rr_lr = rr_lr - rr_rl @ rl_ll @ ll_lr
        t_ref = np.array([[rr_lr, rr_ll], [rl_lr, rl_ll]]).transpose((0, 2, 1, 3)).reshape(s.shape)
        t = matrix.convert(s)
        npt.assert_array_almost_equal(t, t_ref)
        t_approx = matrix.convert(s, noise_level=1e-3)
        npt.assert_array_almost_equal(t_approx, t_ref, decimal=5)
        s2 = matrix.convert(matrix.convert(s))
        npt.assert_array_almost_equal(s2, s, err_msg='Repeated conversion did not work.')
        s3 = matrix.convert(matrix.convert(s, noise_level=1e-3), noise_level=1e-3)
        npt.assert_array_almost_equal(s3, s, err_msg='Regularization did not work for repeated conversion.')

    def test_convert_matrix(self):
        rr_lr = np.array([[0.9, 0.3], [-0.2, 0.7]])
        ll_lr = np.array([[0.1, 0.2], [0.0, 0.1]])
        rr_rl = np.array([[0.0, -0.2], [0.1, 0.1]])
        ll_rl = np.array([[0.8, 0.2], [-0.2, 0.8]])
        s = np.array([[rr_lr, rr_rl], [ll_lr, ll_rl]]).transpose((0, 2, 1, 3)).reshape(2 * np.asarray(rr_lr.shape))
        rl_ll = np.linalg.inv(ll_rl)
        rl_lr = - rl_ll @ ll_lr
        rr_ll = rr_rl @ rl_ll
        rr_lr = rr_lr - rr_rl @ rl_ll @ ll_lr
        t_ref = np.array([[rr_lr, rr_ll], [rl_lr, rl_ll]]).transpose((0, 2, 1, 3)).reshape(s.shape)
        t = matrix.convert(s)
        npt.assert_array_almost_equal(t, t_ref)
        t_approx = matrix.convert(s, noise_level=1e-3)
        npt.assert_array_almost_equal(t_approx, t_ref, decimal=5)
        s2 = matrix.convert(matrix.convert(s))
        npt.assert_array_almost_equal(s2, s, err_msg='Repeated conversion did not work.')
        s3 = matrix.convert(matrix.convert(s, noise_level=1e-3), noise_level=1e-3)
        npt.assert_array_almost_equal(s3, s, err_msg='Regularization did not work for repeated conversion.')

    def test_convert_matrix_complex(self):
        rr_lr = np.array([[0.9, 0.3], [-0.2, 0.7 + 0.2j]])
        ll_lr = np.array([[0.1j, 0.2], [0.0, 0.1]])
        rr_rl = np.array([[0.0, -0.2], [0.2 + 0.1j, 0.1]])
        ll_rl = np.array([[0.1 + 0.8j, 0.2], [-0.2, 0.8]])
        s = np.array([[rr_lr, rr_rl], [ll_lr, ll_rl]]).transpose((0, 2, 1, 3)).reshape(2 * np.asarray(rr_lr.shape))
        rl_ll = np.linalg.inv(ll_rl)
        rl_lr = - rl_ll @ ll_lr
        rr_ll = rr_rl @ rl_ll
        rr_lr = rr_lr - rr_rl @ rl_ll @ ll_lr
        t_ref = np.array([[rr_lr, rr_ll], [rl_lr, rl_ll]]).transpose((0, 2, 1, 3)).reshape(s.shape)
        t = matrix.convert(s)
        npt.assert_array_almost_equal(t, t_ref)
        t_approx = matrix.convert(s, noise_level=1e-3)
        npt.assert_array_almost_equal(t_approx, t_ref, decimal=5)
        s2 = matrix.convert(matrix.convert(s))
        npt.assert_array_almost_equal(s2, s, err_msg='Repeated conversion did not work.')
        s3 = matrix.convert(matrix.convert(s, noise_level=1e-3), noise_level=1e-3)
        npt.assert_array_almost_equal(s3, s, err_msg='Regularization did not work for repeated conversion.')


