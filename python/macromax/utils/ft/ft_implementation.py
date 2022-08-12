"""
The `ft` module provides direct access to `numpy.fft`, `scipy.fftpack`, `pyfftw`, or `mkl_fft`, depending on availability.
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

__all__ = ['fftshift', 'ifftshift', 'fft', 'ifft', 'fftn', 'ifftn']

log = logging.getLogger(__name__)

from numpy.fft import fftshift, ifftshift

try:
    from mkl_fft import *
    log.debug("Using mkl_fft for Fast Fourier transforms.")
except (ModuleNotFoundError, TypeError):
    try:
        import pyfftw
        from pyfftw.interfaces.scipy_fft import *
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(60.0)
        try:
            import multiprocessing
            nb_threads = multiprocessing.cpu_count()
            log.debug(f'Using pyfftw with {nb_threads} threads for Fast Fourier transform.')
            pyfftw.config.NUM_THREADS = nb_threads
        except ModuleNotFoundError:
            log.info('Python multiprocessing module not found, using default number of threads')
    except ModuleNotFoundError:
        try:
            from scipy.fftpack import *
            log.info("Using scipy.fftpack Fast Fourier transforms instead of mkl_fft or pyfftw.")
        except ModuleNotFoundError:
            from numpy.fft import *
            log.info("Using numpy.fft Fast Fourier transforms instead of scipy.fftpack, mkl_fft, or pyfftw.")
