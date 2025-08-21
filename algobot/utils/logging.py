"""Central logging setup."""
from __future__ import annotations
import logging
from logging import Logger
from pathlib import Path

_DEF_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "algobot", level: int = logging.INFO, log_dir: str = "logs") -> Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        fmt = logging.Formatter(_DEF_FORMAT)
        fh = logging.FileHandler(Path(log_dir)/f"{name}.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
