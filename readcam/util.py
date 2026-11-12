"""Utility functions."""

import time
import pyperclip


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