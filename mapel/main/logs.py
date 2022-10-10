import logging

log_format = "%(levelname)s :: %(asctime)s :: %(name)s :: %(funcName)s :: %(message)s"
library_wide_level = logging.INFO

logging.basicConfig(format = log_format, level = library_wide_level)


def set_up_logging_level(level):
  global library_wide_level
  num_level = getattr(logging, level.upper(), None)
  if not isinstance(num_level, int):
    raise ValueError(f"Invalid log level: {loglevel}")
  library_wide_level = num_level

def get_logger(name):
  logger = logging.getLogger(name)
  logger.setLevel(library_wide_level)
  return logger

  

