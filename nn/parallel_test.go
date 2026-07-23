package nn

import (
	"runtime"
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

func TestIsDeterministicReflectsSetDeterministic(t *testing.T) {
	defer SetDeterministic(false)
	SetDeterministic(true)
	if !IsDeterministic() {
		t.Error("IsDeterministic() = false after SetDeterministic(true)")
	}
	SetDeterministic(false)
	if IsDeterministic() {
		t.Error("IsDeterministic() = true after SetDeterministic(false)")
	}
}

func TestSetDeterministicForcesSingleChunk(t *testing.T) {
	defer SetDeterministic(false)
	origGOMAXPROCS := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(origGOMAXPROCS)
	runtime.GOMAXPROCS(8)

	SetDeterministic(true)
	if got := numParallelChunks(1000); got != 1 {
		t.Errorf("numParallelChunks(1000) under SetDeterministic(true) = %d, want 1", got)
	}
	var chunksSeen []int
	parallelChunks(1000, func(chunk, start, end int) { chunksSeen = append(chunksSeen, chunk) })
	if len(chunksSeen) != 1 || chunksSeen[0] != 0 {
		t.Errorf("parallelChunks under SetDeterministic(true) invoked chunks %v, want exactly [0]", chunksSeen)
	}
}

func TestNumParallelChunksRespectsGOMAXPROCSWhenNotDeterministic(t *testing.T) {
	SetDeterministic(false)
	origGOMAXPROCS := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(origGOMAXPROCS)
	runtime.GOMAXPROCS(4)

	if got := numParallelChunks(1000); got != 4 {
		t.Errorf("numParallelChunks(1000) with GOMAXPROCS=4 = %d, want 4", got)
	}
}

// TestSetDeterministicGivesBitExactResultsAcrossGOMAXPROCS is the point of
// the whole feature: the same layer, the same input, run under two
// different GOMAXPROCS settings, must produce bit-identical output when
// SetDeterministic(true) is active — proving the parallel reduction order
// really is pinned, not just "usually close enough" at float32 precision.
func TestSetDeterministicGivesBitExactResultsAcrossGOMAXPROCS(t *testing.T) {
	defer SetDeterministic(false)
	origGOMAXPROCS := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(origGOMAXPROCS)
	SetDeterministic(true)

	rng := NewRNG(1)
	l := Linear(rng, 32, 16, XavierInit())
	x := NewTensor([]int{64, 32})
	for i := range x.Data {
		x.Data[i] = float32(i%23)*0.037 - 0.4
	}
	ctx := &Context{Mode: Train}

	runtime.GOMAXPROCS(1)
	out1, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward (GOMAXPROCS=1): %v", err)
	}
	got1 := append([]float32(nil), out1.Data...)
	gradOut := NewTensor(out1.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	if _, err := l.Backward(ctx, gradOut); err != nil {
		t.Fatalf("Backward (GOMAXPROCS=1): %v", err)
	}
	wGrad1 := append([]float32(nil), l.W.Grad.Data...)

	runtime.GOMAXPROCS(8)
	out2, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward (GOMAXPROCS=8): %v", err)
	}
	got2 := append([]float32(nil), out2.Data...)
	if _, err := l.Backward(ctx, gradOut); err != nil {
		t.Fatalf("Backward (GOMAXPROCS=8): %v", err)
	}
	wGrad2 := append([]float32(nil), l.W.Grad.Data...)

	for i := range got1 {
		if got1[i] != got2[i] {
			t.Fatalf("Forward output[%d] differs across GOMAXPROCS with SetDeterministic(true): %v vs %v", i, got1[i], got2[i])
		}
	}
	for i := range wGrad1 {
		if wGrad1[i] != wGrad2[i] {
			t.Fatalf("W.Grad[%d] differs across GOMAXPROCS with SetDeterministic(true): %v vs %v", i, wGrad1[i], wGrad2[i])
		}
	}
}
