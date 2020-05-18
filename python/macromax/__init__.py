import logging

# create logger
log = logging.getLogger('MacroMax')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s|%(name)s-%(levelname)s: %(message)s')

# Clear all previously added handlers
for h in log.handlers:
    log.removeHandler(h)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
# add the handler to the logger
log.addHandler(ch)

# create file handler which logs debug messages
try:
    fh = logging.FileHandler(log.name + '.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # add the handler to the logger
    log.addHandler(fh)
except IOError:
    ch.setLevel(logging.DEBUG)
    log.warning("Could not create log file. Redirecting messages to console output.")


# Import in main name space
from .solver import solve, Solution
from .utils.array import Grid


__version__ = '0.1.2'

__all__ = ['__version__', 'solve', 'Solution', 'Grid', 'log']
