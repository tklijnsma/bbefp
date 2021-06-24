import logging

COLORS = {
    "yellow": "\033[33m",
    "red": "\033[31m",
    "green": "\033[32m",
}
RESET = "\033[0m"

def colored(text, color=None):
    if color is not None:
        text = COLORS[color] + text + RESET
    return text

def setup_logger(name="bbefp", fmt=None):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info("Logger %s is already defined", name)
    else:
        fmt = logging.Formatter(
            fmt=(
                colored(
                    "[{0}|%(levelname)8s|%(asctime)s|%(module)s]:".format(name),
                    "yellow",
                )
                + " %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

from . import utils
from . import dataset
from . import networks
from . import particlenet