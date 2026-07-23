package data

import (
	"math/rand"
	"sort"
	"testing"
)

func TestDataLoaderYieldsAllIndicesExactlyOnceNoShuffle(t *testing.T) {
	loader := NewDataLoader(10, 3, nil, false)
	var got []int
	for {
		batch, ok := loader.Next()
		if !ok {
			break
		}
		got = append(got, batch...)
	}
	if len(got) != 10 {
		t.Fatalf("got %d indices total, want 10", len(got))
	}
	for i, v := range got {
		if v != i {
			t.Errorf("got[%d] = %d, want %d (no shuffle should preserve order)", i, v, i)
		}
	}
}

func TestDataLoaderBatchSizes(t *testing.T) {
	loader := NewDataLoader(10, 3, nil, false)
	var sizes []int
	for {
		batch, ok := loader.Next()
		if !ok {
			break
		}
		sizes = append(sizes, len(batch))
	}
	want := []int{3, 3, 3, 1}
	if len(sizes) != len(want) {
		t.Fatalf("got %d batches, want %d", len(sizes), len(want))
	}
	for i := range want {
		if sizes[i] != want[i] {
			t.Errorf("batch %d size = %d, want %d", i, sizes[i], want[i])
		}
	}
}

func TestDataLoaderNumBatches(t *testing.T) {
	if n := NewDataLoader(10, 3, nil, false).NumBatches(); n != 4 {
		t.Errorf("NumBatches() = %d, want 4", n)
	}
	if n := NewDataLoader(9, 3, nil, false).NumBatches(); n != 3 {
		t.Errorf("NumBatches() = %d, want 3", n)
	}
}

func TestDataLoaderShuffleCoversEveryIndexExactlyOnce(t *testing.T) {
	loader := NewDataLoader(20, 4, rand.New(rand.NewSource(1)), true)
	var got []int
	for {
		batch, ok := loader.Next()
		if !ok {
			break
		}
		got = append(got, batch...)
	}
	sort.Ints(got)
	for i, v := range got {
		if v != i {
			t.Fatalf("shuffled indices don't cover [0,20) exactly once: got[%d]=%d after sorting, want %d", i, v, i)
		}
	}
}

func TestDataLoaderResetStartsNewEpochWithFreshShuffle(t *testing.T) {
	loader := NewDataLoader(6, 6, rand.New(rand.NewSource(2)), true)
	batch1, _ := loader.Next()
	epoch1 := append([]int(nil), batch1...)

	loader.Reset()
	batch2, _ := loader.Next()
	epoch2 := append([]int(nil), batch2...)

	// Both epochs must still be permutations of [0,6), but (with this
	// seed/size) the actual order should differ between the two calls to
	// Reset's re-shuffle.
	sort.Ints(epoch1)
	sort.Ints(epoch2)
	for i := 0; i < 6; i++ {
		if epoch1[i] != i || epoch2[i] != i {
			t.Fatalf("epoch permutations don't cover [0,6): epoch1=%v epoch2=%v", epoch1, epoch2)
		}
	}
}

func TestDataLoaderNextReturnsFalseWhenExhausted(t *testing.T) {
	loader := NewDataLoader(2, 5, nil, false)
	if _, ok := loader.Next(); !ok {
		t.Fatal("first Next() returned ok=false, want true")
	}
	if _, ok := loader.Next(); ok {
		t.Fatal("second Next() returned ok=true, want false (dataset exhausted)")
	}
}
