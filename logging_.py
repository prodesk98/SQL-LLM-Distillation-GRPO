from loguru import logger
import sys
from pathlib import Path


def setup_logger(log_dir="logs", log_level="INFO"):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        str(log_path / "run.log"),
        level=log_level,
        rotation="5 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logger initialized")

