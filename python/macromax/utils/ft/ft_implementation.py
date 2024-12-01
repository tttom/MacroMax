"""
The `ft` module provides direct access to `numpy.fft`, `pyfftw`, or `mkl_fft`, depending on availability.
The back-end that is deemed most efficient is automatically imported. Independently of the back-end, the `ft` module provides at least the following functions:

- `ft.fft(a: array_like, axis: int)`: The 1-dimensional fast Fourier transform.

- `ft.ifft(a: array_like, axis: int)`: The 1-dimensional fast Fourier transform.

- `ft.fftn(a: array_like, axes: Sequence)`: The n-dimensional fast Fourier transform.

- `ft.ifftn(a: array_like, axes: Sequence)`: The n-dimensional inverse fast Fourier transform.

- `ft.fftshift(a: array_like, axes: Sequence)`: This fftshifts the input array.

- `ft.ifftshift(a: array_like, axes: Sequence)`: This ifftshifts the input array (only different from `ft.fftshift` for odd-shapes).

All functions return a numpy.ndarray of the same shape as the input array or array_like, `a`.
With the exception of `ft.fft()` and `ft.ifft()`, all functions take the `axes` argument to limit the action to specific axes of the numpy.ndarray.

Note that axis indices should be unique and non-negative. **Negative or repeated axis indices are not compatible with all back-end implementations!**
"""
import logging
import os

from numpy.fft import fftshift, ifftshift

from macromax.utils.ft import log

__all__ = ['fftshift', 'ifftshift', 'fft', 'ifft', 'fftn', 'ifftn']

log = log.getChild(__name__)

if (nb_threads := os.cpu_count()) is not None:
    for _ in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_] = str(nb_threads)
    try:
        import mkl
        mkl.set_num_threads(nb_threads)
    except (ImportError, TypeError):
        pass
    log.info(f'Set maximum number of threads to {nb_threads}.')

try:
    from mkl_fft import *
    log.info('Using mkl_fft for Fast Fourier transforms.')
except (ModuleNotFoundError, TypeError):
    try:
        import pyfftw
        from pyfftw.interfaces.scipy_fft import *
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(60.0)
        if nb_threads is not None:
            pyfftw.config.NUM_THREADS = nb_threads
        log.info('Using FFTW for Fast Fourier transforms instead of mkl_fft.')
    except (ModuleNotFoundError, TypeError):
        from numpy.fft import *
        log.info('Using numpy.fft Fast Fourier transforms instead of mkl_fft or pyfftw.')
