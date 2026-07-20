# Conv GEMM Optimization Implementation Plan

**Goal:** Drastically cut training time beyond the multi-core work by
fixing the arithmetic-per-memory-access profile of the hot loops.

**Baseline after parallelization:** 59 s/epoch quick mode (~68% CPU on 8
logical cores). Conv layers ≈ 95% of compute.

**Why the current conv is slow even parallelized:** the 7-deep loop does
full flat-index arithmetic (`((b*h+ih)*w+iw)*inC+ic` etc.) plus bounds
branches per multiply-add, and the innermost `kw` walk strides by `inC`
through x while the `oc` walk strides by `inC*k*k` through W — poor cache
behavior, no compiler vectorization.

## Task 1: GEMM-style Conv2D forward/backward (`nn/conv.go`)

Fused im2col/axpy formulation, keeping the existing `parallelChunks`
batch parallelism and per-chunk weight-gradient partials:

- Pre-transpose weights once per Forward/Backward into
  `wT[(kh*k+kw)*inC+ic][outC]` (flat `[]float32`, ≤18k floats) so the
  innermost loops read W contiguously.
- **Forward**, per output position `(b, oh, ow)`:
  `outRow[0:outC]` starts as bias copy; for each in-bounds `(kh, kw)` the
  input pixel `xSeg = x[base : base+inC]` is contiguous, and for each
  `(ic, xv)` the update is `outRow[n] += xv * wT[row][n]` over the
  contiguous `wT` row — a pure axpy over `outC` elements with zero index
  math in the inner loop (range over slices ⇒ minimal bounds checks).
- **Backward**, per output position: `gRow = gradOut[m*outC:...]`;
  for each in-bounds `(kh, kw, ic)`:
  - `gradIn[base+ic] += dot(gRow, wTRow)` (contiguous dot),
  - `wGradT[row][n] += xv * gRow[n]` (axpy into the chunk's partial).
  Partials are accumulated transposed and folded back into the
  `[oc][ic][kh][kw]` layout during the serial chunk-order reduce.
- B.Grad unchanged (per-chunk partials as today).

Verification: existing conv tests + `nn` gradient checks + `-race` run
must pass unchanged (same math, same summation order per element within
a batch chunk is NOT preserved — per-element accumulation order changes —
so gradcheck tolerances validate correctness, exact float equality with
the old code is not expected).

## Task 2: BatchNorm modulo elimination (`nn/norm.go`)

Every pass currently computes `c = i % bn.channels` per element (integer
division, several passes over up-to-524k-element tensors per batch).
Because channels are the fastest axis, replace with a wrapping counter
(`c++; if c == channels { c = 0 }`) seeded once per chunk/loop start.
Applies to: train-mode mean/variance reductions, normalize pass,
inference pass, and all four backward passes.

Verification: existing BatchNorm tests + gradient checks.

## Task 3: Benchmark

Quick-mode run: measure steady-state epoch time vs the 59 s baseline and
CPU utilization; leave the run to completion to confirm accuracy is
unchanged (~57% val acc trajectory).
