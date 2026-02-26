"""
Logging setup cho HolmHz.

Tại sao không dùng print()?
→ Logger có levels (DEBUG, INFO, WARNING, ERROR)
→ Tự thêm timestamp
→ Dễ redirect output vào file
→ Professional code practice
"""

import logging
import sys


def get_logger(
    name: str = "holmhz",
    level: str = "INFO",
) -> logging.Logger:
    """Tạo logger với format sạch.

    Args:
        name: tên logger (thường dùng __name__)
        level: log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        configured logging.Logger

    Example:
        >>> logger = get_logger("training")
        >>> logger.info("Epoch 1: loss=0.5")
        2026-02-26 10:00:00 | INFO     | Epoch 1: loss=0.5
    """
    logger = logging.getLogger(name)

    # Tránh thêm handler trùng nếu gọi nhiều lần
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper()))
    return logger
