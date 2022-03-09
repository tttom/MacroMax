"""
The `solver` module calculates the solution to the wave equations. More specifically, the work is done in the iteration defined
in the :meth:`Solution.__iter__` method of the :class:`Solution` class. The convenience function :func:`solve` is
provided to construct a :class:`Solution` object and iterate it to convergence using its :meth:`Solution.solve` method.

Public attributes:

- :data:`__version__`: The MacroMax version number as a `str`.

- :func:`solve`: The function to solve the wave problem.

- :class:`Solution`: The class that is used by the :func:`solve` function, which can be used for fine-control of the iteration or re-use.

- :class:`Grid`: A class representing uniformly spaced Cartesian grids and their Fourier Transforms.

- :attr:`log`: The :mod:`logging` object of the :mod:`macromax` library. This can be used to make the messages more or less verbose.

- :mod:`backend <macromax.backend>`: The sub-package with the back-end specifications.
"""
import logging
try:
    import coloredlogs
    formatter_class = coloredlogs.ColoredFormatter
except ImportError:
    formatter_class = logging.Formatter

# create logger
log = logging.getLogger('MacroMax')
log.setLevel(logging.WARNING)

# create formatter and add it to the handlers
log_format = '%(asctime)s|%(name)s-%(levelname)s: %(message)s'

# Clear all previously added handlers
for h in log.handlers:
    log.removeHandler(h)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter_class(log_format))
# add the handler to the logger
log.addHandler(ch)

# create file handler which logs debug messages
try:
    fh = logging.FileHandler(log.name + '.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format))  # Use plain formatting (no color) for the file
    # add the handler to the logger
    log.addHandler(fh)
except IOError:
    ch.setLevel(logging.DEBUG)
    log.warning("Could not create log file. Redirecting messages to console output.")


# Import in main name space
from .solver import solve, Solution
from .utils.array import Grid
import macromax.backend

__version__ = '0.1.5'

__all__ = ['__version__', 'solve', 'Solution', 'Grid', 'log', 'backend']
