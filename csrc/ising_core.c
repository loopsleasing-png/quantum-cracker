/*
 * ising_core.c - Ising energy evaluation and SQA sweep kernels.
 *
 * The critical inner loop of simulated quantum annealing runs entirely
 * in C: spin flip proposals, energy deltas, Metropolis acceptance,
 * and EC constraint evaluation via the ECEvaluator.
 */

#include "ising_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- xoshiro256** PRNG (public domain, Blackman & Vigna) --- */

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro_next(uint64_t *s) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

static inline double rng_double(uint64_t *s) {
    return (double)(xoshiro_next(s) >> 11) * 0x1.0p-53;
}

static inline int rng_int(uint64_t *s, int n) {
    return (int)(xoshiro_next(s) % (uint64_t)n);
}

/* --- IsingModel lifecycle --- */

IsingModel *ising_create(
    int n_spins,
    const double *h_fields,
    int n_couplings,
    const int *coup_i,
    const int *coup_j,
    const double *coup_val,
    double constraint_weight,
    const double *constraint_diagonal,
    int constraint_diag_size
) {
    IsingModel *m = (IsingModel *)calloc(1, sizeof(IsingModel));
    if (!m) return NULL;

    m->n_spins = n_spins;
    m->n_couplings = n_couplings;
    m->constraint_weight = constraint_weight;

    /* Copy h_fields */
    if (h_fields) {
        m->h_fields = (double *)malloc(n_spins * sizeof(double));
        memcpy(m->h_fields, h_fields, n_spins * sizeof(double));
    }

    /* Copy couplings */
    if (n_couplings > 0) {
        m->coup_i = (int *)malloc(n_couplings * sizeof(int));
        m->coup_j = (int *)malloc(n_couplings * sizeof(int));
        m->coup_val = (double *)malloc(n_couplings * sizeof(double));
        memcpy(m->coup_i, coup_i, n_couplings * sizeof(int));
        memcpy(m->coup_j, coup_j, n_couplings * sizeof(int));
        memcpy(m->coup_val, coup_val, n_couplings * sizeof(double));

        /* Build per-spin coupling adjacency lists */
        m->spin_coup_offset = (int *)calloc(n_spins + 1, sizeof(int));

        /* Count couplings per spin */
        for (int c = 0; c < n_couplings; c++) {
            m->spin_coup_offset[coup_i[c] + 1]++;
            m->spin_coup_offset[coup_j[c] + 1]++;
        }
        /* Prefix sum */
        for (int s = 1; s <= n_spins; s++) {
            m->spin_coup_offset[s] += m->spin_coup_offset[s - 1];
        }

        int total = m->spin_coup_offset[n_spins];
        m->spin_coup_list = (int *)malloc(total * sizeof(int));
        m->spin_coup_which = (int *)malloc(total * sizeof(int));

        /* Fill adjacency lists */
        int *pos = (int *)calloc(n_spins, sizeof(int));
        for (int c = 0; c < n_couplings; c++) {
            int si = coup_i[c];
            int sj = coup_j[c];
            int off_i = m->spin_coup_offset[si] + pos[si];
            m->spin_coup_list[off_i] = c;
            m->spin_coup_which[off_i] = 0;  /* this spin is coup_i */
            pos[si]++;

            int off_j = m->spin_coup_offset[sj] + pos[sj];
            m->spin_coup_list[off_j] = c;
            m->spin_coup_which[off_j] = 1;  /* this spin is coup_j */
            pos[sj]++;
        }
        free(pos);
    }

    /* Copy constraint diagonal */
    if (constraint_diagonal && constraint_diag_size > 0) {
        m->constraint_diag_size = constraint_diag_size;
        m->constraint_diagonal = (double *)malloc(constraint_diag_size * sizeof(double));
        memcpy(m->constraint_diagonal, constraint_diagonal,
               constraint_diag_size * sizeof(double));
    }

    return m;
}

void ising_destroy(IsingModel *model) {
    if (!model) return;
    free(model->h_fields);
    free(model->coup_i);
    free(model->coup_j);
    free(model->coup_val);
    free(model->spin_coup_offset);
    free(model->spin_coup_list);
    free(model->spin_coup_which);
    free(model->constraint_diagonal);
    free(model);
}

/* --- Index conversions --- */

int64_t spins_to_index(const int8_t *spins, int n) {
    int64_t idx = 0;
    for (int j = 0; j < n; j++) {
        if (spins[j] == -1) {
            idx |= ((int64_t)1 << j);
        }
    }
    return idx;
}

int spins_to_key(const int8_t *spins, int n) {
    return (int)spins_to_index(spins, n);
}

/* --- Energy evaluation --- */

double ising_energy(const IsingModel *model, const int8_t *spins,
                    ECEvaluator *ec_ev) {
    double e = 0.0;
    int n = model->n_spins;

    /* Local fields */
    if (model->h_fields) {
        for (int i = 0; i < n; i++) {
            e += model->h_fields[i] * spins[i];
        }
    }

    /* Couplings */
    for (int c = 0; c < model->n_couplings; c++) {
        e += model->coup_val[c] * spins[model->coup_i[c]] * spins[model->coup_j[c]];
    }

    /* Constraint */
    if (model->constraint_diagonal) {
        int64_t idx = spins_to_index(spins, n);
        if (idx < model->constraint_diag_size) {
            e += model->constraint_weight * model->constraint_diagonal[idx];
        }
    } else if (ec_ev) {
        e += model->constraint_weight * ec_evaluator_constraint_penalty(ec_ev);
    }

    return e;
}

/*
 * Delta-E for single spin flip using per-spin adjacency lists.
 * O(degree) instead of O(n_couplings).
 */
double ising_delta_e_single(const IsingModel *model, const int8_t *spins,
                            int flip_idx, ECEvaluator *ec_ev) {
    double dE = 0.0;
    int8_t s_i = spins[flip_idx];

    /* Local field contribution */
    if (model->h_fields) {
        dE += -2.0 * model->h_fields[flip_idx] * s_i;
    }

    /* Coupling contributions (only neighbors of flip_idx) */
    if (model->spin_coup_offset) {
        int start = model->spin_coup_offset[flip_idx];
        int end = model->spin_coup_offset[flip_idx + 1];
        for (int k = start; k < end; k++) {
            int c = model->spin_coup_list[k];
            int other;
            if (model->spin_coup_which[k] == 0) {
                /* flip_idx is coup_i[c] */
                other = model->coup_j[c];
            } else {
                /* flip_idx is coup_j[c] */
                other = model->coup_i[c];
            }
            dE += -2.0 * model->coup_val[c] * s_i * spins[other];
        }
    }

    /* Constraint term */
    if (model->constraint_diagonal) {
        int n = model->n_spins;
        int64_t idx_before = spins_to_index(spins, n);
        int64_t idx_after = idx_before ^ ((int64_t)1 << flip_idx);
        if (idx_before < model->constraint_diag_size &&
            idx_after < model->constraint_diag_size) {
            dE += model->constraint_weight * (
                model->constraint_diagonal[idx_after] -
                model->constraint_diagonal[idx_before]
            );
        }
    } else if (ec_ev) {
        double old_pen = ec_evaluator_constraint_penalty(ec_ev);
        double new_pen = ec_evaluator_peek_flip_single(ec_ev, flip_idx);
        dE += model->constraint_weight * (new_pen - old_pen);
    }

    return dE;
}

double ising_delta_e_pair(const IsingModel *model, const int8_t *spins,
                          int idx_a, int idx_b, ECEvaluator *ec_ev) {
    double dE = 0.0;
    int8_t s_a = spins[idx_a];
    int8_t s_b = spins[idx_b];

    /* Local field contributions */
    if (model->h_fields) {
        dE += -2.0 * model->h_fields[idx_a] * s_a;
        dE += -2.0 * model->h_fields[idx_b] * s_b;
    }

    /* Coupling contributions for idx_a */
    if (model->spin_coup_offset) {
        int start_a = model->spin_coup_offset[idx_a];
        int end_a = model->spin_coup_offset[idx_a + 1];
        for (int k = start_a; k < end_a; k++) {
            int c = model->spin_coup_list[k];
            int other;
            if (model->spin_coup_which[k] == 0) {
                other = model->coup_j[c];
            } else {
                other = model->coup_i[c];
            }
            if (other == idx_b) continue; /* both flip: product unchanged */
            dE += -2.0 * model->coup_val[c] * s_a * spins[other];
        }

        /* Coupling contributions for idx_b */
        int start_b = model->spin_coup_offset[idx_b];
        int end_b = model->spin_coup_offset[idx_b + 1];
        for (int k = start_b; k < end_b; k++) {
            int c = model->spin_coup_list[k];
            int other;
            if (model->spin_coup_which[k] == 0) {
                other = model->coup_j[c];
            } else {
                other = model->coup_i[c];
            }
            if (other == idx_a) continue; /* both flip: product unchanged */
            dE += -2.0 * model->coup_val[c] * s_b * spins[other];
        }
    }

    /* Constraint term */
    if (model->constraint_diagonal) {
        int n = model->n_spins;
        int64_t idx_before = spins_to_index(spins, n);
        int64_t idx_after = idx_before ^ ((int64_t)1 << idx_a) ^ ((int64_t)1 << idx_b);
        if (idx_before < model->constraint_diag_size &&
            idx_after < model->constraint_diag_size) {
            dE += model->constraint_weight * (
                model->constraint_diagonal[idx_after] -
                model->constraint_diagonal[idx_before]
            );
        }
    } else if (ec_ev) {
        double old_pen = ec_evaluator_constraint_penalty(ec_ev);
        double new_pen = ec_evaluator_peek_flip_pair(ec_ev, idx_a, idx_b);
        dE += model->constraint_weight * (new_pen - old_pen);
    }

    return dE;
}

/* --- Parity computation --- */

static int compute_parity(const int8_t *spins, int n) {
    int n_minus = 0;
    for (int i = 0; i < n; i++) {
        if (spins[i] == -1) n_minus++;
    }
    return (n_minus % 2 == 0) ? 1 : -1;
}

/* --- J_perp computation --- */

static double j_perp_base_compute(double gamma, double temperature, int n_replicas) {
    if (temperature <= 0.0 || gamma <= 0.0) return 1e10;
    double pt = n_replicas * temperature;
    double arg = gamma / pt;
    if (arg > 15.0) {
        return pt * exp(-2.0 * arg);
    }
    double tv = tanh(arg);
    if (tv <= 0.0) return 1e10;
    return -(pt / 2.0) * log(tv);
}

static double j_perp_parity_compute(double jp_base, int parity, double delta_e,
                                    double temperature, int n_replicas) {
    if (temperature <= 0.0 || n_replicas <= 0) return jp_base;
    double exponent = delta_e / (2.0 * n_replicas * temperature);
    if (parity == 1) {
        return jp_base * exp(exponent);
    } else {
        return jp_base * exp(-exponent);
    }
}

/* --- SQA sweep --- */

int sqa_sweep(
    const IsingModel *model,
    int n_replicas,
    int8_t **replica_spins,
    ECEvaluator **replica_ec_evs,
    double j_perp_base,
    double parity_suppression,
    double beta,
    double delta_e,
    double temperature,
    int parity_weighted,
    uint64_t *rng_state
) {
    int n = model->n_spins;
    int P = n_replicas;
    int accepted = 0;

    /* Compute parity of each replica */
    int *replica_parities = (int *)malloc(P * sizeof(int));
    for (int r = 0; r < P; r++) {
        replica_parities[r] = compute_parity(replica_spins[r], n);
    }

    for (int r = 0; r < P; r++) {
        int8_t *spins_r = replica_spins[r];
        int r_prev = (r - 1 + P) % P;
        int r_next = (r + 1) % P;

        /* Per-replica J_perp */
        double jp_r;
        if (parity_weighted) {
            jp_r = j_perp_parity_compute(
                j_perp_base, replica_parities[r], delta_e, temperature, P
            );
        } else {
            jp_r = j_perp_base;
        }

        /* Single-spin flip proposals */
        for (int proposal = 0; proposal < n; proposal++) {
            int i = rng_int(rng_state, n);

            /* Intra-replica dE scaled by 1/P */
            double dE_intra = ising_delta_e_single(
                model, spins_r, i,
                replica_ec_evs ? replica_ec_evs[r] : NULL
            ) / P;

            /* Inter-replica coupling dE */
            int8_t s_i = spins_r[i];
            int8_t s_prev = replica_spins[r_prev][i];
            int8_t s_next = replica_spins[r_next][i];
            double dE_inter = jp_r * (-2.0) * s_i * (s_prev + s_next);

            double dE = dE_intra + dE_inter;

            /* Accept with parity suppression */
            double accept_prob;
            if (dE <= 0.0) {
                accept_prob = parity_suppression;
            } else {
                accept_prob = parity_suppression * exp(-beta * dE);
            }

            if (rng_double(rng_state) < accept_prob) {
                spins_r[i] *= -1;
                if (replica_ec_evs && replica_ec_evs[r]) {
                    ec_evaluator_flip_single(replica_ec_evs[r], i);
                }
                replica_parities[r] *= -1;
                accepted++;
            }
        }

        /* Pair-spin flip proposals */
        for (int proposal = 0; proposal < n; proposal++) {
            int i = rng_int(rng_state, n);
            int j = rng_int(rng_state, n);
            if (i == j) continue;

            double dE_intra = ising_delta_e_pair(
                model, spins_r, i, j,
                replica_ec_evs ? replica_ec_evs[r] : NULL
            ) / P;

            int8_t s_i = spins_r[i];
            int8_t s_j = spins_r[j];
            int8_t s_prev_i = replica_spins[r_prev][i];
            int8_t s_next_i = replica_spins[r_next][i];
            int8_t s_prev_j = replica_spins[r_prev][j];
            int8_t s_next_j = replica_spins[r_next][j];
            double dE_inter = jp_r * (-2.0) * (
                s_i * (s_prev_i + s_next_i)
                + s_j * (s_prev_j + s_next_j)
            );

            double dE = dE_intra + dE_inter;

            /* Pair flips preserve parity -- unsuppressed */
            double accept_prob;
            if (dE <= 0.0) {
                accept_prob = 1.0;
            } else {
                accept_prob = exp(-beta * dE);
            }

            if (rng_double(rng_state) < accept_prob) {
                spins_r[i] *= -1;
                spins_r[j] *= -1;
                if (replica_ec_evs && replica_ec_evs[r]) {
                    ec_evaluator_flip_single(replica_ec_evs[r], i);
                    ec_evaluator_flip_single(replica_ec_evs[r], j);
                }
                accepted++;
            }
        }
    }

    free(replica_parities);
    return accepted;
}
