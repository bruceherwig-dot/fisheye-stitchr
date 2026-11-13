"""Fisheye StitchR."""
import pathlib
import sys


def _read_version() -> str:
    """The version lives in version.txt (next to the code, or inside the built app)."""
    for base in (getattr(sys, "_MEIPASS", None), pathlib.Path(__file__).resolve().parent.parent):
        if base:
            f = pathlib.Path(base) / "version.txt"
            if f.exists():
                return f.read_text().strip()
    return "unknown"


__version__ = _read_version()
