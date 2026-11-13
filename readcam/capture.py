"""Capture saving: timestamped PNG output and clipboard image support."""

import ctypes
import logging
import struct
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Windows clipboard constants
CF_DIB = 8
GMEM_MOVEABLE = 0x0002


def save_frame(frame: np.ndarray, directory: Path, timestamp: datetime | None = None) -> Path:
    """Save a BGR frame as a PNG in *directory*.

    The filename is ``ReadCam_YYYY-MM-DD_HH-MM-SS.png``; when a second
    capture lands in the same second, an ``_2``-style suffix is appended.
    Raises :class:`OSError` if the write fails.

    Returns the written path.
    """
    ts = timestamp or datetime.now()
    stem = ts.strftime("ReadCam_%Y-%m-%d_%H-%M-%S")
    directory = Path(directory).expanduser()
    path = directory / f"{stem}.png"
    n = 2
    while path.exists():
        path = directory / f"{stem}_{n}.png"
        n += 1

    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Failed to write {path}")
    logger.info("Capture saved to %s", path)
    return path


def copy_image_to_clipboard(frame: np.ndarray) -> bool:
    """Copy a BGR frame to the clipboard as a bitmap image.

    Windows only (CF_DIB via the Win32 clipboard API); other platforms
    return False. Returns True on success.
    """
    if sys.platform != "win32":
        logger.warning("Clipboard image capture not supported on this platform")
        return False
    try:
        dib = _frame_to_dib(frame)
        return _set_clipboard_dib(dib)
    except Exception:
        logger.exception("Failed to copy image to clipboard")
        return False


def _frame_to_dib(frame: np.ndarray) -> bytes:
    """Encode a BGR frame as a packed DIB (BITMAPINFOHEADER + pixels)."""
    h, w = frame.shape[:2]
    row_size = w * 3
    padding = (4 - (row_size % 4)) % 4

    # BGR byte order is already what DIB expects; rows are bottom-up
    pixels = np.ascontiguousarray(frame, dtype=np.uint8)
    if padding:
        pad = np.zeros((h, padding, 3), dtype=np.uint8)
        pixels = np.concatenate([pixels, pad], axis=1)
    data = pixels[::-1].tobytes()

    header = struct.pack(
        "<IiiHHIIiiII",  # BITMAPINFOHEADER fields
        40,              # biSize
        w,               # biWidth
        h,               # biHeight (positive = bottom-up)
        1,               # biPlanes
        24,              # biBitCount
        0,               # biCompression (BI_RGB)
        row_size + padding,  # biSizeImage
        0, 0, 0, 0,      # resolution and colour counts
    )
    return header + data


def _set_clipboard_dib(dib: bytes) -> bool:
    """Place a DIB on the Windows clipboard (CF_DIB)."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # 64-bit safety: handles are pointer-sized
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]

    if not user32.OpenClipboard(0):
        return False
    try:
        user32.EmptyClipboard()
        mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib))
        if not mem:
            return False
        ptr = kernel32.GlobalLock(mem)
        if not ptr:
            kernel32.GlobalFree(mem)
            return False
        ctypes.memmove(ptr, dib, len(dib))
        kernel32.GlobalUnlock(mem)
        if not user32.SetClipboardData(CF_DIB, mem):
            # The system owns the memory only on success
            kernel32.GlobalFree(mem)
            return False
        return True
    finally:
        user32.CloseClipboard()
