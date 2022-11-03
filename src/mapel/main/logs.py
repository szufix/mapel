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
  # A bit hacky as the loggers get the new level only
  # when this method is run but the loggers are created earlier
  # (for example many modules are loaded in __init__.py
  # of the mapel module). As a result, there might be time
  # (before calling this function) that the loggers have
  # other than the requested level
  # This cannot be fixed unless we use proper file-based
  # logging configuration.
  _update_all_mapel_loggers(num_level)

def get_logger(name):
  global library_wide_level
  logger = logging.getLogger(name)
  logger.setLevel(library_wide_level)
  return logger

  
def _update_all_mapel_loggers(new_level):
  for logname in logging.Logger.manager.loggerDict.keys():
    if logname.startswith("mapel."):
      logging.getLogger(logname).setLevel(library_wide_level)

