#!/usr/bin/env python3
# timing.py
# source: http://stackoverflow.com/a/1557906/6009280

# ----------------------------------------------
# import from standard lib
import atexit
from datetime import timedelta
from time import localtime, strftime, time

# import from other lib
from loguru import logger

# import from my project


def _secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def log(s, elapsed=None):
    line = "-" * 40
    logger.info(line)
    logger.info(f"{_secondsToStr()} - {s}")
    if elapsed:
        logger.info(f"Elapsed time: {elapsed}")
    logger.info(line)


def endlog():
    end = time()
    elapsed = end - start
    log("End Program", _secondsToStr(elapsed))


# ----------------------------------------------
start = time()
atexit.register(endlog)
log("Start Program")
