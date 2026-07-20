# Parallel Training (Multi-Core nn Hot Loops) Implementation Plan

**Goal:** Use all CPU cores during training. Today every layer loop is
single-goroutine, so training pegs one core (~12% on a 8c/16t machine).

**Where the time goes (cifar10_cnn showcase model, per sample forward):**
conv1 ≈ 442k MACs, conv2 ≈ 1.18M, conv3 ≈ 1.18M, linear1 ≈ 131k →
convolutions are ~95% of compute (backward roughly doubles it). Pooling,
BatchNorm, activations are low single digits combined.

**Approach:** Data parallelism over the batch dimension — each worker
goroutine processes a contiguous chunk of batch elements. This keeps every
tensor write disjoint except conv/linear weight gradients, which get
per-chunk partial accumulators reduced serially in chunk order
(deterministic results independent of goroutine scheduling).

**Scope:** `nn/parallel.go` (new) + `nn/conv.go`, `nn/pooling.go`,
`nn/linear.go`. BatchNorm/activations left serial (cheap, and BatchNorm's
batch reductions don't chunk trivially). No API changes; worker count =
`runtime.GOMAXPROCS(0)`. No new dependencies. No commits (user
preference).

## Tasks

### Task 1: parallel helpers (`nn/parallel.go` + `nn/parallel_test.go`)

- `chunkRanges(n, maxChunks int) [][2]int` — split [0,n) into ≤maxChunks
  contiguous non-empty ranges covering n exactly.
- `parallelChunks(n int, fn func(chunk, start, end int))` — run fn per
  chunk in its own goroutine (chunk count = min(GOMAXPROCS, n)), wait for
  completion. Serial fast path when 1 chunk.
- Tests: ranges cover/don't overlap; every index visited exactly once for
  odd sizes (n=1, n=7 with many cores); chunk index passed correctly.

### Task 2: Conv2D

- `Forward`: outer `b` loop → `parallelChunks(batch, …)`; output writes
  are per-`b` disjoint. Weight reads are shared/read-only. Safe.
- `Backward`: per-chunk local `wGrad`/`bGrad` buffers (W is ≤18k floats —
  per-step allocation is negligible next to activation tensors); `gradIn`
  written directly (per-`b` disjoint). After the wait, add partials into
  `W.Grad`/`B.Grad` in chunk order.
- Verification: existing `TestConv2D*` + `nn` gradient-check tests pass.

### Task 3: Pooling

- `MaxPool2D.Forward` / `AvgPool2D.Forward`: batch-chunked; `maxIdx` and
  output writes are per-`b` disjoint.
- `MaxPool2D.Backward`: iterate `maxIdx` in batch-element ranges (layout
  is contiguous per `b`; a max index always points inside its own batch
  element, so cross-chunk write collisions are impossible even with
  overlapping windows).
- `AvgPool2D.Backward`: batch-chunked, per-`b` disjoint writes.
- Verification: existing pooling tests + gradient checks.

### Task 4: Linear

- `Forward`: batch-chunked (out rows disjoint).
- `Backward`: batch-chunked with per-chunk `wGrad`/`bGrad` partials
  (biggest W here is 1024×128 ≈ 0.5 MB/chunk — acceptable churn),
  `gradIn` rows disjoint, serial chunk-order reduce.
- Verification: existing linear tests + gradient checks.

### Task 5: Full verification + benchmark

- `go build ./... && go vet ./... && go test ./...` all green.
- Quick-mode training run: expect CPU utilization near 100% (vs 12%) and
  epoch time well under the ~6.5 min single-thread baseline; run left to
  completion as the end-to-end test training.
