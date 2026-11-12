"""Command-line interface for ReadCam."""

import argparse
import sys

from .camera import CameraDiscovery
from .config import DEFAULT_CAMERA_INDEX, DEFAULT_OCR_INTERVAL, DEFAULT_FRAME_INTERVAL


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="readcam",
        description="Live camera scanner: detect QR codes, barcodes, and text in real time.",
    )
    parser.add_argument(
        "--list-cameras", "-l",
        action="store_true",
        help="List available camera devices and exit.",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Camera index (default: {DEFAULT_CAMERA_INDEX}).",
    )
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=DEFAULT_OCR_INTERVAL,
        help=f"OCR processing interval in ms (default: {DEFAULT_OCR_INTERVAL}).",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=DEFAULT_FRAME_INTERVAL,
        help=f"Frame capture interval in ms (default: {DEFAULT_FRAME_INTERVAL}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser


def entry_point() -> int:
    """Application entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args()

    if args.list_cameras:
        cameras = CameraDiscovery.list_cameras()
        if not cameras:
            print("No cameras detected.")
            return 1
        print("Available cameras:")
        for cam in cameras:
            print(f"  [{cam.index}] {cam.name}")
        return 0

    # Import and launch UI
    try:
        from .ui.app import ReadCamApp
        app = ReadCamApp(
            camera_index=args.camera,
            ocr_interval_ms=args.ocr_interval,
            frame_interval_ms=args.frame_interval,
            debug=args.debug,
        )
        app.run()
    except KeyboardInterrupt:
        pass

    return 0


def main():
    """Script entry point."""
    raise SystemExit(entry_point())


if __name__ == "__main__":
    main()