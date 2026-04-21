"""Persistent file logging configuration for experiment runs."""

import logging
from pathlib import Path


def setup_file_logging(
    output_dir: Path,
    *,
    level: int = logging.DEBUG,
    verbose: bool = False,
) -> None:
    """Configure the ``auto_scientist`` logger with a file handler.

    Writes structured log entries to ``{output_dir}/debug.log`` at *level*.
    When *verbose* is True, also attaches a console handler at INFO level.
    """
    logger = logging.getLogger("auto_scientist")
    logger.setLevel(level)

    # Avoid duplicate handlers on resume (run() called twice in same process)
    if any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("debug.log")
        for h in logger.handlers
    ):
        return

    fh = logging.FileHandler(output_dir / "debug.log")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(ch)
