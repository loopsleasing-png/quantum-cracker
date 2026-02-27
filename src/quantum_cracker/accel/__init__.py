"""C accelerators for quantum-cracker hot paths.

Loads libqc_accel.so via ctypes when available.
All modules in this package provide Python fallbacks when the
shared library is not found.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_lib: ctypes.CDLL | None = None


def _find_lib() -> ctypes.CDLL | None:
    """Try to load libqc_accel.so from known locations."""
    candidates = [
        # Relative to project root (development)
        Path(__file__).resolve().parents[3] / "lib" / "libqc_accel.so",
        # Relative to csrc build
        Path(__file__).resolve().parents[3] / "csrc" / ".." / "lib" / "libqc_accel.so",
        # System lib paths
        Path("/usr/local/lib/libqc_accel.so"),
        # Current directory
        Path("lib/libqc_accel.so"),
    ]

    # Also check QC_ACCEL_LIB environment variable
    env_path = os.environ.get("QC_ACCEL_LIB")
    if env_path:
        candidates.insert(0, Path(env_path))

    for path in candidates:
        try:
            resolved = path.resolve()
            if resolved.exists():
                return ctypes.CDLL(str(resolved))
        except OSError:
            continue
    return None


def get_lib() -> ctypes.CDLL | None:
    """Return the loaded C library, or None if unavailable."""
    global _lib
    if _lib is None:
        _lib = _find_lib()
    return _lib


def is_available() -> bool:
    """True if the C accelerator library is loaded."""
    return get_lib() is not None
