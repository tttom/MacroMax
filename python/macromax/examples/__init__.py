import logging

# create logger
log = logging.getLogger('MacroMaxExamples')
log.setLevel(logging.INFO)

# Clear all previously added handlers
for h in log.handlers:
    log.removeHandler(h)

# create file handler which logs even debug messages
fh = logging.FileHandler('MacroMaxExamples.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s|%(name)s-%(levelname)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
log.addHandler(fh)
log.addHandler(ch)

