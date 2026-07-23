// train/parallel.go
package train

import (
	"runtime"
	"sync"

	"github.com/stolzmi/neugo/nn"
)

// parallelUpdateThreshold is the minimum element count a parameter needs
// before its per-element optimizer update is split across goroutines.
// Below this, goroutine dispatch overhead outweighs the parallel work
// itself — optimizer updates are a handful of FLOPs per element, far
// cheaper per-element than the forward/backward math nn.parallelChunks
// splits, so the break-even point sits much higher than "any slice at
// all".
const parallelUpdateThreshold = 4096

// chunkRanges splits [0, n) into at most maxChunks contiguous, non-empty,
// non-overlapping [start, end) ranges that cover it exactly. Mirrors
// nn.chunkRanges; duplicated here because it's unexported in nn and this
// package cannot reach across the boundary for it.
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

func maxParallelChunks() int {
	if nn.IsDeterministic() {
		return 1
	}
	return runtime.GOMAXPROCS(0)
}

// numParallelChunks returns how many chunks parallelChunks(n, ...) will
// actually split n into — callers that pre-size a per-chunk partial
// accumulator (see sumSquares below) must use this rather than computing
// GOMAXPROCS directly, so sizing stays correct under
// nn.SetDeterministic(true) too.
func numParallelChunks(n int) int {
	return len(chunkRanges(n, maxParallelChunks()))
}

// parallelChunks runs fn over [0, n) split into one contiguous chunk per
// worker and waits for all of them. fn must only write to state disjoint
// per chunk; chunk is the range's index for callers keeping per-chunk
// accumulators.
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

// parallelFor runs fn(0, n) inline when n is below parallelUpdateThreshold,
// otherwise splits [0, n) across goroutines via parallelChunks. This is
// the shape every Optimizer.Step below uses for its per-element update —
// small parameters (biases, norm scales) stay single-threaded; large ones
// (weight matrices, embedding tables) get split.
func parallelFor(n int, fn func(start, end int)) {
	if n < parallelUpdateThreshold {
		fn(0, n)
		return
	}
	parallelChunks(n, func(_, start, end int) { fn(start, end) })
}

// sumSquares computes sum(x*x) over data, splitting the reduction across
// goroutines (each chunk sums into its own slot, summed serially after)
// once data is large enough to be worth it.
func sumSquares(data []float32) float64 {
	n := len(data)
	if n < parallelUpdateThreshold {
		var s float64
		for _, v := range data {
			s += float64(v) * float64(v)
		}
		return s
	}
	partial := make([]float64, numParallelChunks(n))
	parallelChunks(n, func(chunk, start, end int) {
		var s float64
		for i := start; i < end; i++ {
			s += float64(data[i]) * float64(data[i])
		}
		partial[chunk] = s
	})
	var total float64
	for _, s := range partial {
		total += s
	}
	return total
}
