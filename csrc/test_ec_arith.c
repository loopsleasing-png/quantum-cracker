/*
 * test_ec_arith.c - Quick smoke tests for EC arithmetic.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "ec_arith.h"

static void test_add_identity(void) {
    ECCurve curve = {97, 0, 7};
    ECPoint inf = {0, 0, 1};
    ECPoint P = {1, 28, 0};

    ECPoint R = ec_add(&curve, P, inf);
    assert(!R.is_inf && R.x == P.x && R.y == P.y);

    R = ec_add(&curve, inf, P);
    assert(!R.is_inf && R.x == P.x && R.y == P.y);

    printf("  PASS: add_identity\n");
}

static void test_add_inverse(void) {
    ECCurve curve = {97, 0, 7};
    ECPoint P = {1, 28, 0};
    ECPoint neg_P = ec_neg(&curve, P);
    assert(neg_P.y == 97 - 28);

    ECPoint R = ec_add(&curve, P, neg_P);
    assert(R.is_inf);

    printf("  PASS: add_inverse\n");
}

static void test_multiply_order(void) {
    /* y^2 = x^3 + 7 over F_97. Generator (1,28). */
    ECCurve curve = {97, 0, 7};
    ECPoint G = {1, 28, 0};

    /* Find the order: smallest k > 0 where k*G = O */
    int order = 0;
    for (int k = 1; k <= 200; k++) {
        ECPoint R = ec_multiply(&curve, G, k);
        if (R.is_inf) {
            order = k;
            break;
        }
    }
    assert(order > 0);
    printf("  PASS: multiply_order (order=%d)\n", order);

    /* Verify: (order-1)*G + G = O */
    ECPoint Q = ec_multiply(&curve, G, order - 1);
    ECPoint R = ec_add(&curve, Q, G);
    assert(R.is_inf);
    printf("  PASS: order-1 + 1 = infinity\n");
}

static void test_evaluator(void) {
    /* y^2 = x^3 + 7 over F_97 */
    ECCurve curve = {97, 0, 7};
    ECPoint G = {1, 28, 0};

    /* Pick key=5, compute public = 5*G */
    ECPoint pub = ec_multiply(&curve, G, 5);
    assert(!pub.is_inf);

    ECEvaluator *ev = ec_evaluator_create(97, 0, 7, G.x, G.y, pub.x, pub.y, 8);
    assert(ev != NULL);

    /* Set state to key=5, should have zero penalty */
    ec_evaluator_set_state(ev, 5);
    assert(ec_evaluator_constraint_penalty(ev) == 0.0);

    /* Set state to key=3, should have nonzero penalty */
    ec_evaluator_set_state(ev, 3);
    assert(ec_evaluator_constraint_penalty(ev) == 1.0);

    /* Test flip_single: flip bit 0 (key 3 -> 2) */
    double pen = ec_evaluator_flip_single(ev, 0);
    assert(ev->current_key == 2);
    /* 2*G should not be pub (5*G) */
    assert(pen == 1.0);

    /* Test peek_flip_single */
    double peek_pen = ec_evaluator_peek_flip_single(ev, 0);
    /* Peek flip bit 0: key 2 -> 3. 3*G != pub */
    assert(ev->current_key == 2);  /* state unchanged */
    (void)peek_pen;

    /* Test copy */
    ECEvaluator *ev2 = ec_evaluator_copy(ev);
    assert(ev2->current_key == ev->current_key);
    ec_evaluator_flip_single(ev2, 0);
    assert(ev2->current_key == 3);
    assert(ev->current_key == 2);  /* original unchanged */

    ec_evaluator_destroy(ev);
    ec_evaluator_destroy(ev2);

    printf("  PASS: evaluator\n");
}

int main(void) {
    printf("EC arithmetic tests:\n");
    test_add_identity();
    test_add_inverse();
    test_multiply_order();
    test_evaluator();
    printf("All EC tests passed!\n");
    return 0;
}
