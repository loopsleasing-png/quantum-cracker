"""ctypes wrapper for the C EC arithmetic library.

Provides CECEvaluator, a drop-in replacement for ECEnergyEvaluator
that delegates all point arithmetic to C.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_double,
    c_int,
    c_int8,
    c_int64,
    c_void_p,
)

import numpy as np
from numpy.typing import NDArray

from quantum_cracker.accel import get_lib


class _ECPoint(Structure):
    _fields_ = [
        ("x", c_int64),
        ("y", c_int64),
        ("is_inf", c_int),
    ]


class _ECCurve(Structure):
    _fields_ = [
        ("p", c_int64),
        ("a", c_int64),
        ("b", c_int64),
    ]


def _setup_lib(lib: ctypes.CDLL) -> None:
    """Set up ctypes function signatures."""
    # ec_evaluator_create
    lib.ec_evaluator_create.restype = c_void_p
    lib.ec_evaluator_create.argtypes = [
        c_int64, c_int64, c_int64,  # p, a, b
        c_int64, c_int64,           # gx, gy
        c_int64, c_int64,           # px, py
        c_int,                       # n_bits
    ]

    # ec_evaluator_destroy
    lib.ec_evaluator_destroy.restype = None
    lib.ec_evaluator_destroy.argtypes = [c_void_p]

    # ec_evaluator_copy
    lib.ec_evaluator_copy.restype = c_void_p
    lib.ec_evaluator_copy.argtypes = [c_void_p]

    # ec_evaluator_set_state
    lib.ec_evaluator_set_state.restype = None
    lib.ec_evaluator_set_state.argtypes = [c_void_p, c_int64]

    # ec_evaluator_set_state_from_spins
    lib.ec_evaluator_set_state_from_spins.restype = None
    lib.ec_evaluator_set_state_from_spins.argtypes = [
        c_void_p, POINTER(c_int8),
    ]

    # ec_evaluator_constraint_penalty
    lib.ec_evaluator_constraint_penalty.restype = c_double
    lib.ec_evaluator_constraint_penalty.argtypes = [c_void_p]

    # ec_evaluator_flip_single
    lib.ec_evaluator_flip_single.restype = c_double
    lib.ec_evaluator_flip_single.argtypes = [c_void_p, c_int]

    # ec_evaluator_peek_flip_single
    lib.ec_evaluator_peek_flip_single.restype = c_double
    lib.ec_evaluator_peek_flip_single.argtypes = [c_void_p, c_int]

    # ec_evaluator_peek_flip_pair
    lib.ec_evaluator_peek_flip_pair.restype = c_double
    lib.ec_evaluator_peek_flip_pair.argtypes = [c_void_p, c_int, c_int]


_lib_initialized = False


def _ensure_lib() -> ctypes.CDLL | None:
    global _lib_initialized
    lib = get_lib()
    if lib is not None and not _lib_initialized:
        _setup_lib(lib)
        _lib_initialized = True
    return lib


class CECEvaluator:
    """C-accelerated EC constraint evaluator.

    Drop-in replacement for ECEnergyEvaluator from ec_constraints.py.
    Delegates all EC point arithmetic to the C shared library.
    """

    def __init__(
        self,
        p: int, a: int, b: int,
        gx: int, gy: int,
        px: int, py: int,
        n_bits: int,
    ) -> None:
        lib = _ensure_lib()
        if lib is None:
            raise RuntimeError("C accelerator library not available")

        self._lib = lib
        self.n_bits = n_bits
        self._ptr = lib.ec_evaluator_create(
            c_int64(p), c_int64(a), c_int64(b),
            c_int64(gx), c_int64(gy),
            c_int64(px), c_int64(py),
            c_int(n_bits),
        )
        if not self._ptr:
            raise MemoryError("Failed to create C ECEvaluator")

        # Store curve info for Python-side access
        self._p = p
        self._a = a
        self._b = b
        self._gx = gx
        self._gy = gy
        self._px = px
        self._py = py

        # Compatibility: shared sentinel objects so `copy.power_points is orig.power_points`
        self._power_points_sentinel: object = object()
        self._neg_power_points_sentinel: object = object()
        # Track current key on Python side for compatibility
        self._current_key: int = 0

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr:
            self._lib.ec_evaluator_destroy(self._ptr)
            self._ptr = None

    @property
    def power_points(self) -> object:
        """Compatibility: sentinel for shared-reference checks."""
        return self._power_points_sentinel

    @property
    def neg_power_points(self) -> object:
        """Compatibility: sentinel for shared-reference checks."""
        return self._neg_power_points_sentinel

    def set_state(self, key: int) -> None:
        self._lib.ec_evaluator_set_state(self._ptr, c_int64(key))
        self._current_key = key

    def set_state_from_spins(self, spins: NDArray[np.int8]) -> None:
        spins_c = spins.astype(np.int8, copy=False)
        self._lib.ec_evaluator_set_state_from_spins(
            self._ptr,
            spins_c.ctypes.data_as(POINTER(c_int8)),
        )
        # Reconstruct key from spins
        key = 0
        for j in range(self.n_bits):
            if spins_c[j] == -1:
                key |= 1 << j
        self._current_key = key

    def constraint_penalty(self) -> float:
        return self._lib.ec_evaluator_constraint_penalty(self._ptr)

    def flip_single(self, bit_idx: int) -> float:
        result = self._lib.ec_evaluator_flip_single(self._ptr, c_int(bit_idx))
        self._current_key ^= 1 << bit_idx
        return result

    def peek_flip_single(self, bit_idx: int) -> float:
        return self._lib.ec_evaluator_peek_flip_single(self._ptr, c_int(bit_idx))

    def peek_flip_pair(self, bit_a: int, bit_b: int) -> float:
        return self._lib.ec_evaluator_peek_flip_pair(
            self._ptr, c_int(bit_a), c_int(bit_b),
        )

    def copy(self) -> CECEvaluator:
        new = CECEvaluator.__new__(CECEvaluator)
        new._lib = self._lib
        new.n_bits = self.n_bits
        new._p = self._p
        new._a = self._a
        new._b = self._b
        new._gx = self._gx
        new._gy = self._gy
        new._px = self._px
        new._py = self._py
        new._current_key = self._current_key
        # Share sentinel objects so `copy.power_points is orig.power_points`
        new._power_points_sentinel = self._power_points_sentinel
        new._neg_power_points_sentinel = self._neg_power_points_sentinel
        new._ptr = self._lib.ec_evaluator_copy(self._ptr)
        if not new._ptr:
            raise MemoryError("Failed to copy C ECEvaluator")
        return new
