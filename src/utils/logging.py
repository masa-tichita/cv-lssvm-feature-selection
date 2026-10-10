import sys

from loguru import logger


def setup_logger() -> None:
    """Initialize logging (call once at CLI startup)."""
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.info("Logger setup")
