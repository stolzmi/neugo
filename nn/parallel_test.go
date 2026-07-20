package nn

import (
	"sync"
	"testing"
)

func TestChunkRangesCoverExactlyWithoutOverlap(t *testing.T) {
	for _, tc := range []struct{ n, maxChunks int }{
		{1, 8}, {7, 16}, {16, 4}, {33, 8}, {5, 5}, {100, 3},
	} {
		ranges := chunkRanges(tc.n, tc.maxChunks)
		if len(ranges) > tc.maxChunks {
			t.Errorf("n=%d maxChunks=%d: got %d chunks", tc.n, tc.maxChunks, len(ranges))
		}
		seen := make([]int, tc.n)
		for _, r := range ranges {
			if r[0] >= r[1] {
				t.Errorf("n=%d maxChunks=%d: empty or inverted range %v", tc.n, tc.maxChunks, r)
			}
			for i := r[0]; i < r[1]; i++ {
				seen[i]++
			}
		}
		for i, count := range seen {
			if count != 1 {
				t.Fatalf("n=%d maxChunks=%d: index %d covered %d times", tc.n, tc.maxChunks, i, count)
			}
		}
	}
}

func TestChunkRangesDegenerate(t *testing.T) {
	if got := chunkRanges(0, 4); got != nil {
		t.Errorf("chunkRanges(0, 4) = %v, want nil", got)
	}
	if got := chunkRanges(3, 0); len(got) != 1 || got[0] != [2]int{0, 3} {
		t.Errorf("chunkRanges(3, 0) = %v, want [[0 3]]", got)
	}
}

func TestParallelChunksVisitsEveryIndexOnce(t *testing.T) {
	const n = 137
	var mu sync.Mutex
	seen := make([]int, n)
	parallelChunks(n, func(chunk, start, end int) {
		for i := start; i < end; i++ {
			mu.Lock()
			seen[i]++
			mu.Unlock()
		}
	})
	for i, count := range seen {
		if count != 1 {
			t.Fatalf("index %d visited %d times", i, count)
		}
	}
}
