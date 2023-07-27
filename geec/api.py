"""
Application Program Interface

Main functions to interact with the application
"""

# --- import -----------------------------------
# import from standard lib
import sys
from pathlib import Path

# import from other lib
import pandas as pd
from loguru import logger

# import from my project


def setup_logdir() -> Path:
    """setup log directory"""
    _ = Path(__file__)
    logdir = _.parent.parent / "log"
    logdir.mkdir(parents=True, exist_ok=True)
    return logdir


def setup_logger():
    logdir = setup_logdir()
    """setup logger"""
    # removes all handlers
    # Note: remove(0) to remove the default (0ᵗʰ) handler
    logger.remove()
    # add handler to stderr
    logger.add(sys.stderr, level="SUCCESS", format="{message}")
    # add handler to file
    logger.add(logdir / "geec_{time}.log", level="DEBUG")


def write_file(df: pd.DataFrame, output: str = "") -> None:
    """Write output file."""
    if output:
        # Save result in csv file
        file_path = Path(output).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path.with_suffix(".csv"), index=False)
        logger.success(
            f"\nResults are saved in csv file {file_path.with_suffix('.csv')}"
        )
