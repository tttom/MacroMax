"""
The `ft` module links through to `numpy.fft`, `scipy.fftpack`, `pyfftw`, or `mkl_fft`, depending on availability.
It provides the functions:
* `ft.fft()`: The 1-dimensional fast Fourier transform.
* `ft.ifft()`: The 1-dimensional fast Fourier transform.
* `ft.fft2()`: The 2D specific case for `ft.fftn()`.
* `ft.ifft2()`: The 2D specific case for `ft.ifftn()`.
* `ft.fftn()`: The n-dimensional fast Fourier transform.
* `ft.ifftn()`: The n-dimensional inverse fast Fourier transform.
* `ft.fftshift()`: This fftshifts the input array.
* `ft.ifftshift()`: This ifftshifts the input array (only different from `ft.fftshift` for odd-shapes).

With the exception of `ft.fft()` and `ft.ifft()`, all functions take the `axes` argument to limit the action to specific axes of the numpy.ndarray.

**Note that axis indices should be unique and non-negative. Negative or repeated axis indices are not compatible with all back-end implementations!**
"""
from macromax import log

try:
    from mkl_fft import *
    from numpy.fft import fftshift, ifftshift
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
