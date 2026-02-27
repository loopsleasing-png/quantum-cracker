#!/usr/bin/env python3
"""Build the C accelerator library for quantum-cracker.

Usage:
    python build_c.py          # Build the shared library
    python build_c.py clean    # Remove build artifacts
    python build_c.py test     # Build and run C unit tests
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
LIB = ROOT / "lib"


def build() -> None:
    """Build the shared library."""
    LIB.mkdir(exist_ok=True)
    cmd = [
        "gcc", "-O3", "-march=native", "-fPIC", "-Wall", "-Wextra",
        "-std=c11", "-shared", "-lm",
        "-o", str(LIB / "libqc_accel.so"),
        str(CSRC / "ec_arith.c"),
        str(CSRC / "ising_core.c"),
    ]
    print(f"Building: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"Built: {LIB / 'libqc_accel.so'}")


def clean() -> None:
    """Remove build artifacts."""
    so = LIB / "libqc_accel.so"
    if so.exists():
        so.unlink()
        print(f"Removed: {so}")
    test_bin = CSRC / "test_ec_arith"
    if test_bin.exists():
        test_bin.unlink()
        print(f"Removed: {test_bin}")


def test() -> None:
    """Build and run C unit tests."""
    build()
    test_bin = CSRC / "test_ec_arith"
    cmd = [
        "gcc", "-O2", "-Wall", "-std=c11",
        "-o", str(test_bin),
        str(CSRC / "test_ec_arith.c"),
        str(CSRC / "ec_arith.c"),
        "-lm",
    ]
    print(f"Building test: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"Running: {test_bin}")
    subprocess.check_call([str(test_bin)])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "clean":
            clean()
        elif action == "test":
            test()
        else:
            print(f"Unknown action: {action}")
            sys.exit(1)
    else:
        build()
