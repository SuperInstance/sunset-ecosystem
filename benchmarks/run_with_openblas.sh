#!/bin/bash
# Wrapper to run turbovec benchmarks with OpenBLAS preloaded.
# The turbovec wheel is not linked against libopenblas, so cblas_sgemm
# is unresolved at runtime. Preloading fixes it.

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so

cd "$(dirname "$0")"
python3 "$@"
