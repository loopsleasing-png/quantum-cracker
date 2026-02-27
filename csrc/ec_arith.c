/*
 * ec_arith.c - Elliptic curve point arithmetic over F_p.
 *
 * Uses __int128 for intermediate products to handle 64-bit primes
 * without overflow. All modular arithmetic keeps values in [0, p).
 */

#include "ec_arith.h"
#include <stdlib.h>
#include <string.h>

/* --- Modular arithmetic with 128-bit intermediates --- */

static inline int64_t mod_pos(int64_t a, int64_t p) {
    int64_t r = a % p;
    return r < 0 ? r + p : r;
}

static inline int64_t mod_mul(int64_t a, int64_t b, int64_t p) {
    __int128 r = (__int128)a * (__int128)b;
    int64_t result = (int64_t)(r % (__int128)p);
    return result < 0 ? result + p : result;
}

int64_t mod_inv(int64_t a, int64_t p) {
    /* Extended Euclidean algorithm */
    int64_t old_r = mod_pos(a, p), r = p;
    int64_t old_s = 1, s = 0;

    while (r != 0) {
        int64_t q = old_r / r;
        int64_t tmp;

        tmp = r;
        r = old_r - q * r;
        old_r = tmp;

        tmp = s;
        s = old_s - q * s;
        old_s = tmp;
    }

    return mod_pos(old_s, p);
}

/* --- EC point operations --- */

static const ECPoint POINT_INF = {0, 0, 1};

ECPoint ec_neg(const ECCurve *curve, ECPoint P) {
    if (P.is_inf) return POINT_INF;
    ECPoint R;
    R.x = P.x;
    R.y = P.y == 0 ? 0 : curve->p - P.y;
    R.is_inf = 0;
    return R;
}

ECPoint ec_add(const ECCurve *curve, ECPoint P, ECPoint Q) {
    if (P.is_inf) return Q;
    if (Q.is_inf) return P;

    int64_t p = curve->p;

    /* Check if P == -Q */
    if (P.x == Q.x && P.y == mod_pos(p - Q.y, p)) {
        return POINT_INF;
    }

    int64_t lam;
    if (P.x == Q.x && P.y == Q.y) {
        /* Point doubling: lam = (3*x1^2 + a) / (2*y1) */
        if (P.y == 0) return POINT_INF;
        int64_t num = mod_pos(
            mod_mul(3, mod_mul(P.x, P.x, p), p) + curve->a, p
        );
        int64_t den = mod_mul(2, P.y, p);
        lam = mod_mul(num, mod_inv(den, p), p);
    } else {
        /* Point addition: lam = (y2 - y1) / (x2 - x1) */
        if (P.x == Q.x) return POINT_INF;
        int64_t num = mod_pos(Q.y - P.y, p);
        int64_t den = mod_pos(Q.x - P.x, p);
        lam = mod_mul(num, mod_inv(den, p), p);
    }

    ECPoint R;
    R.is_inf = 0;
    R.x = mod_pos(mod_mul(lam, lam, p) - P.x - Q.x, p);
    R.y = mod_pos(mod_mul(lam, mod_pos(P.x - R.x, p), p) - P.y, p);
    return R;
}

ECPoint ec_multiply(const ECCurve *curve, ECPoint P, int64_t k) {
    if (k < 0) {
        P = ec_neg(curve, P);
        k = -k;
    }
    if (k == 0 || P.is_inf) return POINT_INF;

    ECPoint result = POINT_INF;
    ECPoint addend = P;
    while (k > 0) {
        if (k & 1) {
            result = ec_add(curve, result, addend);
        }
        addend = ec_add(curve, addend, addend);
        k >>= 1;
    }
    return result;
}

/* --- ECEvaluator --- */

ECEvaluator *ec_evaluator_create(
    int64_t p, int64_t a, int64_t b,
    int64_t gx, int64_t gy,
    int64_t px, int64_t py,
    int n_bits
) {
    ECEvaluator *ev = (ECEvaluator *)calloc(1, sizeof(ECEvaluator));
    if (!ev) return NULL;

    ev->curve.p = p;
    ev->curve.a = a;
    ev->curve.b = b;
    ev->generator.x = gx;
    ev->generator.y = gy;
    ev->generator.is_inf = 0;
    ev->public_key.x = px;
    ev->public_key.y = py;
    ev->public_key.is_inf = 0;
    ev->n_bits = n_bits;
    ev->current_point = POINT_INF;
    ev->current_key = 0;

    /* Precompute 2^i * G */
    ev->power_points = (ECPoint *)malloc(n_bits * sizeof(ECPoint));
    ev->neg_power_points = (ECPoint *)malloc(n_bits * sizeof(ECPoint));
    if (!ev->power_points || !ev->neg_power_points) {
        free(ev->power_points);
        free(ev->neg_power_points);
        free(ev);
        return NULL;
    }

    ECPoint pt = ev->generator;
    for (int i = 0; i < n_bits; i++) {
        ev->power_points[i] = pt;
        ev->neg_power_points[i] = ec_neg(&ev->curve, pt);
        pt = ec_add(&ev->curve, pt, pt);
    }

    return ev;
}

void ec_evaluator_destroy(ECEvaluator *ev) {
    if (!ev) return;
    free(ev->power_points);
    free(ev->neg_power_points);
    free(ev);
}

ECEvaluator *ec_evaluator_copy(const ECEvaluator *ev) {
    if (!ev) return NULL;
    ECEvaluator *copy = (ECEvaluator *)malloc(sizeof(ECEvaluator));
    if (!copy) return NULL;

    *copy = *ev;
    /* Share precomputed tables (read-only) by allocating and copying */
    copy->power_points = (ECPoint *)malloc(ev->n_bits * sizeof(ECPoint));
    copy->neg_power_points = (ECPoint *)malloc(ev->n_bits * sizeof(ECPoint));
    if (!copy->power_points || !copy->neg_power_points) {
        free(copy->power_points);
        free(copy->neg_power_points);
        free(copy);
        return NULL;
    }
    memcpy(copy->power_points, ev->power_points, ev->n_bits * sizeof(ECPoint));
    memcpy(copy->neg_power_points, ev->neg_power_points, ev->n_bits * sizeof(ECPoint));
    return copy;
}

void ec_evaluator_set_state(ECEvaluator *ev, int64_t key) {
    ev->current_key = key;
    ev->current_point = ec_multiply(&ev->curve, ev->generator, key);
}

void ec_evaluator_set_state_from_spins(ECEvaluator *ev, const int8_t *spins) {
    int64_t key = 0;
    for (int j = 0; j < ev->n_bits; j++) {
        if (spins[j] == -1) {
            key |= ((int64_t)1 << j);
        }
    }
    ec_evaluator_set_state(ev, key);
}

static inline int ec_point_eq(ECPoint a, ECPoint b) {
    if (a.is_inf && b.is_inf) return 1;
    if (a.is_inf || b.is_inf) return 0;
    return a.x == b.x && a.y == b.y;
}

double ec_evaluator_constraint_penalty(const ECEvaluator *ev) {
    return ec_point_eq(ev->current_point, ev->public_key) ? 0.0 : 1.0;
}

double ec_evaluator_flip_single(ECEvaluator *ev, int bit_idx) {
    int was_set = (ev->current_key >> bit_idx) & 1;
    if (was_set) {
        ev->current_point = ec_add(&ev->curve, ev->current_point,
                                   ev->neg_power_points[bit_idx]);
    } else {
        ev->current_point = ec_add(&ev->curve, ev->current_point,
                                   ev->power_points[bit_idx]);
    }
    ev->current_key ^= ((int64_t)1 << bit_idx);
    return ec_point_eq(ev->current_point, ev->public_key) ? 0.0 : 1.0;
}

double ec_evaluator_peek_flip_single(const ECEvaluator *ev, int bit_idx) {
    int was_set = (ev->current_key >> bit_idx) & 1;
    ECPoint new_pt;
    if (was_set) {
        new_pt = ec_add(&ev->curve, ev->current_point,
                        ev->neg_power_points[bit_idx]);
    } else {
        new_pt = ec_add(&ev->curve, ev->current_point,
                        ev->power_points[bit_idx]);
    }
    return ec_point_eq(new_pt, ev->public_key) ? 0.0 : 1.0;
}

double ec_evaluator_peek_flip_pair(const ECEvaluator *ev, int bit_a, int bit_b) {
    ECPoint pt = ev->current_point;

    int was_a = (ev->current_key >> bit_a) & 1;
    if (was_a) {
        pt = ec_add(&ev->curve, pt, ev->neg_power_points[bit_a]);
    } else {
        pt = ec_add(&ev->curve, pt, ev->power_points[bit_a]);
    }

    int was_b = (ev->current_key >> bit_b) & 1;
    if (was_b) {
        pt = ec_add(&ev->curve, pt, ev->neg_power_points[bit_b]);
    } else {
        pt = ec_add(&ev->curve, pt, ev->power_points[bit_b]);
    }

    return ec_point_eq(pt, ev->public_key) ? 0.0 : 1.0;
}
