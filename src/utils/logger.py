"""Configuração de logging estruturado."""

import logging
import sys


def setup_logger(name: str = "churn", level: int = logging.INFO) -> logging.Logger:
    """Configura e retorna um logger estruturado."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
