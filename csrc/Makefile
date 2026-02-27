# Makefile for quantum-cracker C accelerators
#
# Builds shared libraries for ctypes loading.
# Usage: make           (build all)
#        make clean      (remove build artifacts)
#        make test       (build and run C unit tests)

CC      = gcc
CFLAGS  = -O3 -march=native -fPIC -Wall -Wextra -std=c11
LDFLAGS = -shared -lm

# Output directory (relative to project root)
OUTDIR  = ../lib

TARGETS = $(OUTDIR)/libqc_accel.so

.PHONY: all clean test

all: $(OUTDIR) $(TARGETS)

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(OUTDIR)/libqc_accel.so: ec_arith.c ising_core.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -rf $(OUTDIR)/libqc_accel.so

test: all test_ec_arith
	./test_ec_arith

test_ec_arith: test_ec_arith.c ec_arith.c
	$(CC) -O2 -Wall -std=c11 -o $@ $^ -lm
