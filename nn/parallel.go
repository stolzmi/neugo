// nn/parallel.go
package nn

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// deterministic is a process-wide switch (analogous in spirit to
// GOMAXPROCS itself, or PyTorch's use_deterministic_algorithms) rather
// than an explicit parameter threaded through every call, because it
// changes *how* the existing parallelism executes, not a source of
// randomness a caller supplies — unlike this package's *rand.Rand
// convention, there's no per-call value to thread through; every hot
// path already reads runtime.GOMAXPROCS() the same way.
var deterministic atomic.Bool

// SetDeterministic forces every parallelized hot path in this package
// (Conv2D/Conv1D/ConvTranspose2D, Linear, BatchNorm/GroupNorm/LayerNorm/
// RMSNorm/InstanceNorm, the activations, attention, the RNN/LSTM/GRU
// family, ...) to run single-threaded in a fixed reduction order, so
// repeated runs on the same inputs produce bit-exact identical results
// regardless of GOMAXPROCS. It's off by default — parallel, matching
// every prior release — since this trades throughput for exact
// reproducibility; turn it on only when you need bit-for-bit repeatable
// results (verifying a paper's reported numbers, debugging a suspected
// numerical issue), not for normal training.
func SetDeterministic(enabled bool) {
	deterministic.Store(enabled)
}

// IsDeterministic reports whether SetDeterministic(true) is currently in
// effect.
func IsDeterministic() bool {
	return deterministic.Load()
}

// chunkRanges splits [0, n) into at most maxChunks contiguous, non-empty,
// non-overlapping [start, end) ranges that cover it exactly.
func chunkRanges(n, maxChunks int) [][2]int {
	if n <= 0 {
		return nil
	}
	if maxChunks > n {
		maxChunks = n
	}
	if maxChunks < 1 {
		maxChunks = 1
	}
	size := (n + maxChunks - 1) / maxChunks
	ranges := make([][2]int, 0, maxChunks)
	for start := 0; start < n; start += size {
		end := start + size
		if end > n {
			end = n
		}
		ranges = append(ranges, [2]int{start, end})
	}
	return ranges
}

// parallelChunks runs fn over [0, n) split into one contiguous chunk per
// worker (worker count = min(GOMAXPROCS, n)) and waits for all of them.
// fn must only write to state disjoint per chunk; chunk is the range's
// index for callers that keep per-chunk accumulators.
// numParallelChunks returns how many chunks parallelChunks(n, ...) will
// actually split n into — callers that pre-size a per-chunk partial-
// accumulator slice (see LinearLayer.Backward for the pattern) must use
// this instead of calling chunkRanges/GOMAXPROCS directly, so that size
// stays correct under SetDeterministic(true) too.
func numParallelChunks(n int) int {
	return len(chunkRanges(n, maxParallelChunks()))
}

func maxParallelChunks() int {
	if deterministic.Load() {
		return 1
	}
	return runtime.GOMAXPROCS(0)
}

func parallelChunks(n int, fn func(chunk, start, end int)) {
	ranges := chunkRanges(n, maxParallelChunks())
	if len(ranges) == 0 {
		return
	}
	if len(ranges) == 1 {
		fn(0, ranges[0][0], ranges[0][1])
		return
	}
	var wg sync.WaitGroup
	wg.Add(len(ranges))
	for i, r := range ranges {
		go func(chunk, start, end int) {
			defer wg.Done()
			fn(chunk, start, end)
		}(i, r[0], r[1])
	}
	wg.Wait()
}
