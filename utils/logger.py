"""
Logging utility for the driver fatigue detection system.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class SystemLogger:
    """Centralized logging system without duplicate handlers."""

    LOGGER_NAME = "FatigueDetection"

    def __init__(self, log_level: int = logging.INFO, log_dir: Optional[str] = None):
        self.logger = logging.getLogger(self.LOGGER_NAME)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
            self._configure_handlers(log_dir)

        for handler in self.logger.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(log_level)

    def _configure_handlers(self, log_dir: Optional[str]) -> None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        )
        self.logger.addHandler(console_handler)

        try:
            log_path = Path(log_dir or ".")
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_path / f"fatigue_detection_{datetime.now().strftime('%Y%m%d')}.log",
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
            )
            self.logger.addHandler(file_handler)
        except OSError as exc:
            self.logger.warning("File logging disabled: %s", exc)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)
