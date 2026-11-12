"""Utility functions."""

import logging
import sys
import time
import pyperclip

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging.

    Without *verbose*: WARNING and above, minimal format.
    With *verbose*: DEBUG and above with timestamps, module names,
    and thread names -- suitable for tracing scanner events.
    """
    if verbose:
        level = logging.DEBUG
        fmt = (
            "%(asctime)s.%(msecs)03d "
            "[%(threadName)s] %(levelname)s %(name)s: %(message)s"
        )
        datefmt = "%Y-%m-%dT%H:%M:%S"
    else:
        level = logging.WARNING
        fmt = "%(levelname)s: %(message)s"
        datefmt = None

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)


def log_event(message: str, *args) -> None:
    """Log an INFO-level scanner event (visible with --verbose)."""
    logger.info(message, *args)


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard. Returns True on success."""
    try:
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def format_relative_time(timestamp: float) -> str:
    """Format a Unix timestamp as a relative time string."""
    elapsed = time.time() - timestamp

    if elapsed < 1:
        return "just now"
    elif elapsed < 60:
        secs = int(elapsed)
        return f"{secs}s ago" if secs > 1 else "1s ago"
    elif elapsed < 3600:
        mins = int(elapsed / 60)
        return f"{mins}m ago" if mins > 1 else "1m ago"
    else:
        hours = int(elapsed / 3600)
        return f"{hours}h ago" if hours > 1 else "1h ago"


def truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."