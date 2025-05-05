import sys
import logging

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def bind_std_2_logging():
    # stout and stderr to logging https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)