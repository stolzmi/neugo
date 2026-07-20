// nn/parallel.go
package nn

import (
	"runtime"
	"sync"
)

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
func parallelChunks(n int, fn func(chunk, start, end int)) {
	ranges := chunkRanges(n, runtime.GOMAXPROCS(0))
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
