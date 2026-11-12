"""Auto-download and install system dependencies (zbar, Tesseract).

ReadCam depends on `pyzbar` (QR/barcode) and optionally `pytesseract`
(OCR). Both require system-native libraries that cannot be installed
via ``pip`` alone.

This module can download the latest zbar Windows binary release and
install it into the ``pyzbar`` package directory so it loads correctly.
"""

import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


_REPO = "mchehab/zbar"
_GITHUB_API = f"https://api.github.com/repos/{_REPO}/releases/latest"
_WIN_ASSET = "zbar-win_x86_64-DShow.zip"


def _pyzbar_dir() -> Path:
    """Return the pyzbar package directory, or None if not installed."""
    try:
        import pyzbar  # noqa: F811
        return Path(pyzbar.__file__).resolve().parent
    except (ImportError, OSError):
        return None


def _readcam_data_dir() -> Path:
    """Return the path to our local data/zbar directory."""
    return Path(__file__).resolve().parent / "data" / "zbar" / "bin"


def latest_zbar_asset_url() -> str:
    """Fetch the download URL for the latest Windows x86_64 zbar release."""
    req = urllib.request.Request(
        _GITHUB_API, headers={"Accept": "application/vnd.github.v3+json"},
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    for asset in data["assets"]:
        if asset["name"] == _WIN_ASSET:
            return asset["browser_download_url"]
    raise RuntimeError(
        f"Could not find a {_WIN_ASSET} asset "
        f"in the latest {_REPO} release",
    )


def download_zip(url: str) -> zipfile.ZipFile:
    """Download a zip archive and return an in-memory ZipFile object."""
    sys.stdout.write(f"Downloading {url} ...\n")
    sys.stdout.flush()
    req = urllib.request.Request(
        url, headers={"User-Agent": "readcam/0.1.0"},
    )
    resp = urllib.request.urlopen(req)
    return zipfile.ZipFile(io.BytesIO(resp.read()))


def extract_zbar_dlls(zf: zipfile.ZipFile, dest: Path) -> list[Path]:
    """Extract zbar DLLs from the zip to *dest*.

    Copies *libzbar-0.dll* (renamed to *libzbar-64.dll*) and
    *libwinpthread-1.dll*.
    """
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    wanted = {"libzbar-0.dll", "libwinpthread-1.dll"}
    for info in zf.infolist():
        name = Path(info.filename).name
        if name in wanted:
            zf.extract(info, dest)
            src = dest / info.filename
            # Flatten: move from bin/ subdirectory if needed
            if info.filename != name:
                shutil.move(str(src), str(dest / name))
                # Clean up empty bin directory
                parent_dir = dest / Path(info.filename).parent
                if parent_dir.exists():
                    try:
                        parent_dir.rmdir()
                    except OSError:
                        pass
            extracted.append(dest / name)

    # Rename libzbar-0.dll -> libzbar-64.dll
    lib0 = dest / "libzbar-0.dll"
    lib64 = dest / "libzbar-64.dll"
    if lib0.exists():
        shutil.copy2(str(lib0), str(lib64))

    return extracted


def install_into_pyzbar(src_dir: Path) -> bool:
    """Copy DLLs from *src_dir* into the pyzbar package directory.

    Creates a ``.bak`` copy of any existing DLLs first.
    Returns ``True`` on success.
    """
    target = _pyzbar_dir()
    if target is None:
        return False

    for dll in ("libzbar-64.dll", "libwinpthread-1.dll", "libiconv.dll"):
        src = src_dir / dll
        dst = target / dll
        if src.exists():
            # Back up existing DLL
            bak = dst.with_suffix(dst.suffix + ".bak")
            if dst.exists() and not bak.exists():
                shutil.copy2(str(dst), str(bak))
            shutil.copy2(str(src), str(dst))

    return True


def verify_zbar_works() -> bool:
    """Test whether pyzbar can actually load its zbar DLL."""
    try:
        from pyzbar.pyzbar import decode  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def cmd_install_zbar() -> int:
    """CLI handler for ``readcam install-zbar``."""
    if verify_zbar_works():
        sys.stdout.write("zbar is already working. Nothing to do.\n")
        return 0

    pyzbar_path = _pyzbar_dir()
    if pyzbar_path is None:
        sys.stdout.write("pyzbar is not installed. Run: pip install pyzbar\n")
        return 1

    sys.stdout.write("zbar system library not found.\n")
    sys.stdout.write(
        f"  pyzbar is installed at: {pyzbar_path}\n",
    )
    sys.stdout.write(
        "  The bundled libzbar-64.dll requires MSVCR120.dll which is\n"
        "  missing from this system.\n\n",
    )

    try:
        url = latest_zbar_asset_url()
    except Exception as exc:
        sys.stderr.write(f"Could not find the latest zbar release: {exc}\n")
        sys.stderr.write(
            "Download manually from:\n"
            f"  https://github.com/{_REPO}/releases/latest\n"
            "and extract the DLLs into:\n"
            f"  {pyzbar_path}\n",
        )
        return 1

    try:
        zf = download_zip(url)
    except Exception as exc:
        sys.stderr.write(f"Download failed: {exc}\n")
        return 1

    dest = _readcam_data_dir()
    extracted = extract_zbar_dlls(zf, dest)
    sys.stdout.write(
        f"Extracted {len(extracted)} file(s) to {dest}\n",
    )

    if install_into_pyzbar(dest):
        sys.stdout.write(f"Installed DLLs into {pyzbar_path}\n")

    if verify_zbar_works():
        sys.stdout.write("zbar is now working.\n")
        return 0

    sys.stderr.write(
        "zbar still does not load. Try installing the MSVC 2012 runtime\n"
        "or see: https://github.com/mchehab/zbar/releases\n",
    )
    return 1


def cmd_install_tesseract() -> int:
    """CLI handler for ``readcam install-tesseract``.

    Tesseract cannot be auto-installed -- we can only offer guidance.
    """
    try:
        import pytesseract  # noqa: F401
        pytesseract.get_tesseract_version()
        sys.stdout.write("Tesseract is already available.\n")
        return 0
    except (ImportError, OSError):
        pass

    sys.stdout.write(
        "Tesseract OCR is not available.\n\n"
        "Install it manually:\n"
        "  Windows:  winget install tesseract\n"
        "            or https://github.com/UB-Mannheim/tesseract/wiki\n"
        "  macOS:    brew install tesseract\n"
        "  Linux:    apt install tesseract-ocr\n"
    )
    return 1