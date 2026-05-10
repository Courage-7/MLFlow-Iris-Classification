# logging.py — Centralized logging setup

import logging


def setup_logging(name: str, level=logging.INFO) -> logging.Logger:
    """Setup and return a logger with consistent formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
