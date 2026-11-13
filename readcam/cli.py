"""Command-line interface for ReadCam."""

import argparse
import sys

from .camera import CameraDiscovery
from .config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FLIP,
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_OCR_INTERVAL,
    DEFAULT_ROTATION,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="readcam",
        description="Live camera scanner: detect QR codes, barcodes, and text in real time.",
    )
    sub = parser.add_subparsers(dest="command", metavar="")

    # --- scan (default) ---
    scan = sub.add_parser(
        "scan", aliases=["run"],
        help="Start scanning with a camera (default command).",
    )
    scan.add_argument(
        "--camera", "-c",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Camera index (default: {DEFAULT_CAMERA_INDEX}).",
    )
    scan.add_argument(
        "--ocr-interval",
        type=int,
        default=DEFAULT_OCR_INTERVAL,
        help=f"OCR processing interval in ms (default: {DEFAULT_OCR_INTERVAL}).",
    )
    scan.add_argument(
        "--frame-interval",
        type=int,
        default=DEFAULT_FRAME_INTERVAL,
        help=f"Frame capture interval in ms (default: {DEFAULT_FRAME_INTERVAL}).",
    )
    scan.add_argument(
        "--no-auto",
        action="store_true",
        help=(
            "Disable automatic scanning; only OCR/decode the manually "
            "selected region (drag on the preview)."
        ),
    )
    scan.add_argument(
        "--no-ocr",
        action="store_true",
        help=(
            "Bypass text OCR entirely (barcodes-only mode); QR/barcode "
            "decoding keeps running, saving CPU."
        ),
    )
    scan.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=DEFAULT_ROTATION,
        metavar="DEG",
        help=f"Rotate the camera view (default: {DEFAULT_ROTATION}).",
    )
    scan.add_argument(
        "--flip",
        choices=["none", "h", "v", "hv"],
        default=DEFAULT_FLIP,
        help=(
            "Flip the camera view: h (horizontal), v (vertical), "
            "hv (both), or none (default: %s)." % DEFAULT_FLIP
        ),
    )
    scan.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    scan.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Log scanner events (devices, detections, clipboard) to the console.",
    )

    # --- list-cameras ---
    sub.add_parser(
        "list-cameras", aliases=["ls"],
        help="List available camera devices and exit.",
    )

    # --- install-zbar ---
    sub.add_parser(
        "install-zbar",
        help="Download and install zbar system DLL for QR/barcode decoding.",
    )

    # --- install-tesseract ---
    sub.add_parser(
        "install-tesseract",
        help="Show Tesseract installation instructions for OCR.",
    )

    return parser


def entry_point() -> int:
    """Application entry point. Returns exit code."""
    parser = build_parser()
    args, extra = parser.parse_known_args()
    command = args.command

    # Default: if no subcommand given and no --list-cameras, treat as "scan"
    if command is None:
        # Check for legacy --list-cameras / -l flag
        if "-l" in extra or "--list-cameras" in extra:
            cameras = CameraDiscovery.list_cameras()
            if not cameras:
                print("No cameras detected.")
                return 1
            print("Available cameras:")
            for cam in cameras:
                print(f"  [{cam.index}] {cam.name}")
            return 0

        # Re-parse as "scan"
        args = parser.parse_args(["scan"] + extra)
        command = "scan"

    # Logging setup -- applies to every subcommand
    from .util import setup_logging
    verbose = getattr(args, "verbose", False) or getattr(args, "debug", False)
    setup_logging(verbose=verbose)

    if command == "list-cameras" or command == "ls":
        cameras = CameraDiscovery.list_cameras()
        if not cameras:
            print("No cameras detected.")
            return 1
        print("Available cameras:")
        for cam in cameras:
            print(f"  [{cam.index}] {cam.name}")
        return 0

    if command == "install-zbar":
        from .install import cmd_install_zbar
        return cmd_install_zbar()

    if command == "install-tesseract":
        from .install import cmd_install_tesseract
        return cmd_install_tesseract()

    # "scan" / "run" -- launch the UI
    try:
        from .ui.app import ReadCamApp
        app = ReadCamApp(
            camera_index=args.camera,
            ocr_interval_ms=args.ocr_interval,
            frame_interval_ms=args.frame_interval,
            no_auto=args.no_auto,
            no_ocr=args.no_ocr,
            rotation=args.rotate,
            flip=args.flip,
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