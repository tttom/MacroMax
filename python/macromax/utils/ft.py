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
