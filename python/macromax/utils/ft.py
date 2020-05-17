from macromax import log


try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import *
    pyfftw.interfaces.cache.enable()
except ModuleNotFoundError:
    log.info("Module pyfftw for FFTW not found, trying alternative...")
    try:
        from numpy.fft import *
        log.info("Using numpy.fft Fast Fourier transform instead.")
    except ModuleNotFoundError:
        log.info("Module pyfftw nor numpy.fft found, using scipy.fftpack Fast Fourier transform instead.")
        from scipy.fftpack import *
