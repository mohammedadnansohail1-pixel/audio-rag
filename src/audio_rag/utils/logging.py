"""Structured logging configuration."""

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_configured = False


def setup_logging(level: LogLevel = "INFO", format_style: Literal["simple", "detailed"] = "simple") -> None:
    """Configure logging for the application.
    
    Args:
        level: Log level
        format_style: 'simple' for development, 'detailed' for production
    """
    global _configured
    
    if _configured:
        return
    
    if format_style == "simple":
        fmt = "%(levelname)s | %(name)s | %(message)s"
    else:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger
    """
    return logging.getLogger(name)
