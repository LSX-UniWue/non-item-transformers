import os
import logging
import sys
import threading
from typing import Optional

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None
_root_logger: Optional[logging.Logger] = None


def config_logging():

    global _default_handler
    global _root_logger

    # FIXME for now we use an environment variable to configure log level, later we can switch to a cli flag?
    if "LOG_LEVEL" in os.environ:
        level = int(os.environ["LOG_LEVEL"])
    else:
        level = logging.INFO

    with _lock:
        if _default_handler:
            # logger already configured
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_asme_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(level)
        library_root_logger.propagate = False

        _root_logger = library_root_logger


def _get_asme_root_logger() -> logging.Logger:
    return logging.getLogger(_get_lib_name())


def _get_lib_name() -> str:
    return __name__.split(".")[0]


def get_root_logger() -> logging.Logger:
    """
    Returns the root logger.

    :return: the root logger.
    """
    return get_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger derived from the root logger with the specified name.
    If no name is supplied the root logger is returned.
    """
    if _root_logger is None:
        config_logging()

    if name is None:
        return _root_logger

    return _root_logger.getChild(name)
