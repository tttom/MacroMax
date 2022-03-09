import logging
try:
    import coloredlogs
    formatter_class = coloredlogs.ColoredFormatter
except ImportError:
    formatter_class = logging.Formatter

# create logger
log = logging.getLogger('MacroMaxExamples')
log.setLevel(logging.INFO)

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
