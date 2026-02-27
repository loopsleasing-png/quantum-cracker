/*
 * ising_core.h - Ising energy evaluation and MCMC sweep kernels.
 *
 * Stores couplings in flat arrays for cache-friendly iteration.
 * Provides delta-E computation for single and pair spin flips,
 * and a full SQA sweep kernel that runs the entire replica MCMC
 * step in C.
 */

#ifndef ISING_CORE_H
#define ISING_CORE_H

#include <stdint.h>
#include "ec_arith.h"

/*
 * IsingModel: stores the Ising Hamiltonian in flat arrays.
 *
 * H = sum_i h_i * sigma_i + sum_{<i,j>} J_{ij} * sigma_i * sigma_j
 *     + W * constraint(sigma)
 *
 * Couplings stored as parallel arrays: coup_i[], coup_j[], coup_val[]
 * of length n_couplings.
 */
typedef struct {
    int      n_spins;
    double  *h_fields;       /* local fields, length n_spins (or NULL) */
    int      n_couplings;
    int     *coup_i;         /* coupling source indices */
    int     *coup_j;         /* coupling target indices */
    double  *coup_val;       /* coupling values J_{ij} */

    /* Per-spin coupling lists for O(degree) energy delta */
    int     *spin_coup_offset;  /* offset into spin_coup_list, length n_spins+1 */
    int     *spin_coup_list;    /* coupling indices for each spin, length 2*n_couplings */
    int     *spin_coup_which;   /* 0 if spin is coup_i, 1 if coup_j */

    double   constraint_weight;
    /* Constraint diagonal (small N) or NULL */
    double  *constraint_diagonal;
    int      constraint_diag_size;
} IsingModel;

/* Create/destroy */
IsingModel *ising_create(
    int n_spins,
    const double *h_fields,     /* can be NULL */
    int n_couplings,
    const int *coup_i,
    const int *coup_j,
    const double *coup_val,
    double constraint_weight,
    const double *constraint_diagonal,  /* can be NULL */
    int constraint_diag_size
);
void ising_destroy(IsingModel *model);

/* Energy evaluation */
double ising_energy(const IsingModel *model, const int8_t *spins,
                    ECEvaluator *ec_ev);

double ising_delta_e_single(const IsingModel *model, const int8_t *spins,
                            int flip_idx, ECEvaluator *ec_ev);

double ising_delta_e_pair(const IsingModel *model, const int8_t *spins,
                          int idx_a, int idx_b, ECEvaluator *ec_ev);

/* Spins <-> index conversion */
int64_t spins_to_index(const int8_t *spins, int n);
int     spins_to_key(const int8_t *spins, int n);

/*
 * SQA sweep: run one complete annealing step over all replicas.
 *
 * For each replica: propose n single-flip and n pair-flip moves
 * with Metropolis acceptance including inter-replica coupling.
 *
 * Returns the number of accepted moves.
 */
int sqa_sweep(
    const IsingModel *model,
    int n_replicas,
    int8_t **replica_spins,        /* array of n_replicas spin arrays */
    ECEvaluator **replica_ec_evs,  /* array of n_replicas evaluators (or NULLs) */
    double j_perp_base,
    double parity_suppression,
    double beta,
    double delta_e,
    double temperature,
    int parity_weighted,
    /* RNG state: xoshiro256** */
    uint64_t *rng_state
);

#endif /* ISING_CORE_H */
