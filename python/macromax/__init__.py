"""
The `solver` module calculates the solution to the wave equations. More specifically, the work is done in the iteration
defined in the :meth:`Solution.__iter__` method of the :class:`Solution` class. The convenience function :func:`solve`
is provided to construct a :class:`Solution` object and iterate it to convergence using its :meth:`Solution.solve` method.

Public attributes:

    :data:`__version__`: The MacroMax version number as a `str`.

    :func:`solve`: The function to solve the wave problem.

    :class:`Solution`: The class that is used by the :func:`solve` function, which can be used for fine-control of the iteration or re-use.

    :class:`Grid`: A class representing uniformly spaced Cartesian grids and their Fourier Transforms.

    :attr:`log`: The :mod:`logging` object of the :mod:`macromax` library. This can be used to make the messages more or less verbose.

    :mod:`backend <macromax.backend>`: The sub-package with the back-end specifications.

"""
import sys
import pathlib
import logging
import coloredlogs
coloredlogs.enable_ansi_support()
__field_styles = coloredlogs.DEFAULT_FIELD_STYLES
__field_styles['msecs'] = __field_styles['asctime']
__field_styles['levelname'] = dict(color='green')
__level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.update(
    spam=dict(color='blue', faint=True),
    debug=dict(color='blue'),
    verbose=dict(color='blue', bold=True),
    info=dict(),
    warning=dict(color=(255, 64, 0)),
    error=dict(color=(255, 0, 0)),
    fatal=dict(color=(255, 0, 0), bold=True, background=(255, 255, 0)),
    critical=dict(color=(0, 0, 0), bold=True, background=(255, 255, 0))
)

__formatter = coloredlogs.ColoredFormatter(f'%(asctime)s|%(name)s-%(levelname)s: %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S.%f',
                                           field_styles=__field_styles, level_styles=__level_styles)

# create logger
log = logging.getLogger(__name__)
log.level = logging.WARNING

# Don't use colored logs for the file logs
__file_formatter = logging.Formatter('%(asctime)s.%(msecs).03d|%(name)s-%(levelname)5s %(threadName)s:%(filename)s:%(lineno)s:%(funcName)s| %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# create console handler
console_log_handler = logging.StreamHandler(sys.stdout)
console_log_handler.level = logging.INFO
console_log_handler.formatter = __formatter
log.addHandler(console_log_handler)  # add the handler to the logger

# create file handler which logs debug messages
try:
    __log_file_path = pathlib.Path(__file__).resolve().parent.parent / f'{log.name}.log'
    __fh = logging.FileHandler(__log_file_path, encoding='utf-8')
    __fh.level = -1
    __fh.formatter = __file_formatter
    # add the handler to the logger
    log.addHandler(__fh)
except IOError:
    console_log_handler.level = logging.DEBUG
    log.warning('Could not create log file. Redirecting messages to console output.')


# Import in main name space
from .solver import solve, Solution
from .matrix import ScatteringMatrix
from .utils.ft.grid import Grid
import macromax.backend

__version__ = '0.2.2'

__all__ = ['__version__', 'solve', 'Solution', 'ScatteringMatrix', 'Grid', 'log', 'backend']
