"""ctypes wrapper for the C Ising core library.

Provides CIsingModel for fast energy evaluation and delta-E
computation, plus a sqa_sweep function that runs the entire
SQA MCMC step in C.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    c_double,
    c_int,
    c_int8,
    c_int64,
    c_uint64,
    c_void_p,
)

import numpy as np
from numpy.typing import NDArray

from quantum_cracker.accel import get_lib
from quantum_cracker.accel._ec_arith import CECEvaluator


def _setup_lib(lib: ctypes.CDLL) -> None:
    """Set up Ising function signatures."""
    # ising_create
    lib.ising_create.restype = c_void_p
    lib.ising_create.argtypes = [
        c_int,                       # n_spins
        POINTER(c_double),           # h_fields (or NULL)
        c_int,                       # n_couplings
        POINTER(c_int),              # coup_i
        POINTER(c_int),              # coup_j
        POINTER(c_double),           # coup_val
        c_double,                    # constraint_weight
        POINTER(c_double),           # constraint_diagonal (or NULL)
        c_int,                       # constraint_diag_size
    ]

    # ising_destroy
    lib.ising_destroy.restype = None
    lib.ising_destroy.argtypes = [c_void_p]

    # ising_energy
    lib.ising_energy.restype = c_double
    lib.ising_energy.argtypes = [c_void_p, POINTER(c_int8), c_void_p]

    # ising_delta_e_single
    lib.ising_delta_e_single.restype = c_double
    lib.ising_delta_e_single.argtypes = [
        c_void_p, POINTER(c_int8), c_int, c_void_p,
    ]

    # ising_delta_e_pair
    lib.ising_delta_e_pair.restype = c_double
    lib.ising_delta_e_pair.argtypes = [
        c_void_p, POINTER(c_int8), c_int, c_int, c_void_p,
    ]

    # sqa_sweep
    lib.sqa_sweep.restype = c_int
    lib.sqa_sweep.argtypes = [
        c_void_p,                    # model
        c_int,                       # n_replicas
        POINTER(POINTER(c_int8)),    # replica_spins
        POINTER(c_void_p),           # replica_ec_evs
        c_double,                    # j_perp_base
        c_double,                    # parity_suppression
        c_double,                    # beta
        c_double,                    # delta_e
        c_double,                    # temperature
        c_int,                       # parity_weighted
        POINTER(c_uint64),           # rng_state (4 uint64s)
    ]


_lib_initialized = False


def _ensure_lib() -> ctypes.CDLL | None:
    global _lib_initialized
    lib = get_lib()
    if lib is not None and not _lib_initialized:
        _setup_lib(lib)
        _lib_initialized = True
    return lib


class CIsingModel:
    """C-accelerated Ising model for fast energy evaluation."""

    def __init__(
        self,
        n_spins: int,
        h_fields: NDArray[np.float64] | None,
        j_couplings: dict[tuple[int, int], float],
        constraint_weight: float,
        constraint_diagonal: NDArray[np.float64] | None = None,
    ) -> None:
        lib = _ensure_lib()
        if lib is None:
            raise RuntimeError("C accelerator library not available")

        self._lib = lib
        self.n_spins = n_spins

        # Convert h_fields
        h_ptr = None
        if h_fields is not None:
            h_arr = np.ascontiguousarray(h_fields, dtype=np.float64)
            h_ptr = h_arr.ctypes.data_as(POINTER(c_double))
            self._h_ref = h_arr  # prevent GC

        # Convert couplings to parallel arrays
        n_couplings = len(j_couplings)
        if n_couplings > 0:
            coup_i_arr = np.zeros(n_couplings, dtype=np.int32)
            coup_j_arr = np.zeros(n_couplings, dtype=np.int32)
            coup_val_arr = np.zeros(n_couplings, dtype=np.float64)
            for idx, ((i, j), val) in enumerate(j_couplings.items()):
                coup_i_arr[idx] = i
                coup_j_arr[idx] = j
                coup_val_arr[idx] = val
            ci_ptr = coup_i_arr.ctypes.data_as(POINTER(c_int))
            cj_ptr = coup_j_arr.ctypes.data_as(POINTER(c_int))
            cv_ptr = coup_val_arr.ctypes.data_as(POINTER(c_double))
            self._coup_refs = (coup_i_arr, coup_j_arr, coup_val_arr)
        else:
            ci_ptr = None
            cj_ptr = None
            cv_ptr = None

        # Convert constraint diagonal
        cd_ptr = None
        cd_size = 0
        if constraint_diagonal is not None:
            cd_arr = np.ascontiguousarray(constraint_diagonal, dtype=np.float64)
            cd_ptr = cd_arr.ctypes.data_as(POINTER(c_double))
            cd_size = len(cd_arr)
            self._cd_ref = cd_arr

        self._ptr = lib.ising_create(
            c_int(n_spins),
            h_ptr,
            c_int(n_couplings),
            ci_ptr, cj_ptr, cv_ptr,
            c_double(constraint_weight),
            cd_ptr,
            c_int(cd_size),
        )
        if not self._ptr:
            raise MemoryError("Failed to create C IsingModel")

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr:
            self._lib.ising_destroy(self._ptr)
            self._ptr = None

    def energy(
        self, spins: NDArray[np.int8], ec_ev: CECEvaluator | None = None,
    ) -> float:
        spins_c = np.ascontiguousarray(spins, dtype=np.int8)
        ev_ptr = ec_ev._ptr if ec_ev else None
        return self._lib.ising_energy(
            self._ptr,
            spins_c.ctypes.data_as(POINTER(c_int8)),
            ev_ptr,
        )

    def delta_e_single(
        self, spins: NDArray[np.int8], flip_idx: int,
        ec_ev: CECEvaluator | None = None,
    ) -> float:
        spins_c = np.ascontiguousarray(spins, dtype=np.int8)
        ev_ptr = ec_ev._ptr if ec_ev else None
        return self._lib.ising_delta_e_single(
            self._ptr,
            spins_c.ctypes.data_as(POINTER(c_int8)),
            c_int(flip_idx),
            ev_ptr,
        )

    def delta_e_pair(
        self, spins: NDArray[np.int8], idx_a: int, idx_b: int,
        ec_ev: CECEvaluator | None = None,
    ) -> float:
        spins_c = np.ascontiguousarray(spins, dtype=np.int8)
        ev_ptr = ec_ev._ptr if ec_ev else None
        return self._lib.ising_delta_e_pair(
            self._ptr,
            spins_c.ctypes.data_as(POINTER(c_int8)),
            c_int(idx_a), c_int(idx_b),
            ev_ptr,
        )


def sqa_sweep_c(
    model: CIsingModel,
    replica_spins: list[NDArray[np.int8]],
    replica_ec_evs: list[CECEvaluator | None],
    j_perp_base: float,
    parity_suppression: float,
    beta: float,
    delta_e: float,
    temperature: float,
    parity_weighted: bool,
    rng: np.random.Generator,
) -> int:
    """Run one full SQA sweep in C.

    Args:
        model: C Ising model
        replica_spins: list of spin arrays (modified in-place)
        replica_ec_evs: list of C EC evaluators (modified in-place)
        j_perp_base: base inter-replica coupling
        parity_suppression: t1/t2 suppression factor
        beta: inverse temperature
        delta_e: parity energy gap
        temperature: temperature
        parity_weighted: whether to use parity-dependent J_perp
        rng: numpy random generator (used to seed C RNG)

    Returns:
        Number of accepted moves.
    """
    lib = _ensure_lib()
    if lib is None:
        raise RuntimeError("C accelerator library not available")

    n_replicas = len(replica_spins)

    # Build array of pointers to spin arrays
    SpinPtrArray = POINTER(c_int8) * n_replicas
    spin_ptrs = SpinPtrArray()
    for r in range(n_replicas):
        replica_spins[r] = np.ascontiguousarray(replica_spins[r], dtype=np.int8)
        spin_ptrs[r] = replica_spins[r].ctypes.data_as(POINTER(c_int8))

    # Build array of EC evaluator pointers
    EvPtrArray = c_void_p * n_replicas
    ev_ptrs = EvPtrArray()
    for r in range(n_replicas):
        ev_ptrs[r] = replica_ec_evs[r]._ptr if replica_ec_evs[r] else None

    # Seed C RNG from numpy RNG
    rng_state = (c_uint64 * 4)()
    seeds = rng.integers(0, 2**64, size=4, dtype=np.uint64)
    for i in range(4):
        rng_state[i] = int(seeds[i])

    accepted = lib.sqa_sweep(
        model._ptr,
        c_int(n_replicas),
        ctypes.cast(spin_ptrs, POINTER(POINTER(c_int8))),
        ctypes.cast(ev_ptrs, POINTER(c_void_p)),
        c_double(j_perp_base),
        c_double(parity_suppression),
        c_double(beta),
        c_double(delta_e),
        c_double(temperature),
        c_int(1 if parity_weighted else 0),
        ctypes.cast(rng_state, POINTER(c_uint64)),
    )

    return accepted
