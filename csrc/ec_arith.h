/*
 * ec_arith.h - Elliptic curve point arithmetic over F_p (64-bit primes).
 *
 * Provides fast EC add/neg/multiply for use in the Ising MCMC inner loop.
 * All arithmetic uses 128-bit intermediates to avoid overflow on 64-bit primes.
 *
 * Point representation: (x, y) with the point at infinity encoded as
 * is_inf=1 in ECPoint. Coordinates are in [0, p).
 */

#ifndef EC_ARITH_H
#define EC_ARITH_H

#include <stdint.h>

typedef struct {
    int64_t x;
    int64_t y;
    int     is_inf;  /* 1 = point at infinity */
} ECPoint;

typedef struct {
    int64_t p;       /* prime modulus */
    int64_t a;       /* curve parameter a in y^2 = x^3 + ax + b */
    int64_t b;       /* curve parameter b */
} ECCurve;

/* Core point operations */
ECPoint ec_add(const ECCurve *curve, ECPoint P, ECPoint Q);
ECPoint ec_neg(const ECCurve *curve, ECPoint P);
ECPoint ec_multiply(const ECCurve *curve, ECPoint P, int64_t k);

/* Modular arithmetic helpers */
int64_t mod_inv(int64_t a, int64_t p);

/*
 * ECEvaluator: maintains running point for O(1) spin-flip constraint
 * evaluation. Precomputes 2^i * G for all bit positions.
 */
typedef struct {
    ECCurve  curve;
    ECPoint  generator;
    ECPoint  public_key;
    int      n_bits;
    ECPoint *power_points;      /* 2^i * G, length n_bits */
    ECPoint *neg_power_points;  /* -(2^i * G), length n_bits */
    ECPoint  current_point;     /* k' * G for current spin state */
    int64_t  current_key;       /* candidate key from spins */
} ECEvaluator;

/* ECEvaluator lifecycle */
ECEvaluator *ec_evaluator_create(
    int64_t p, int64_t a, int64_t b,
    int64_t gx, int64_t gy,
    int64_t px, int64_t py,
    int n_bits
);
void ec_evaluator_destroy(ECEvaluator *ev);
ECEvaluator *ec_evaluator_copy(const ECEvaluator *ev);

/* ECEvaluator operations */
void   ec_evaluator_set_state(ECEvaluator *ev, int64_t key);
void   ec_evaluator_set_state_from_spins(ECEvaluator *ev, const int8_t *spins);
double ec_evaluator_constraint_penalty(const ECEvaluator *ev);
double ec_evaluator_flip_single(ECEvaluator *ev, int bit_idx);
double ec_evaluator_peek_flip_single(const ECEvaluator *ev, int bit_idx);
double ec_evaluator_peek_flip_pair(const ECEvaluator *ev, int bit_a, int bit_b);

#endif /* EC_ARITH_H */
