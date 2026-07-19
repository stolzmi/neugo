package data

import (
	"math/rand"
	"testing"
)

func newTestRNG(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

func TestShuffleDataIsDeterministicPerRNG(t *testing.T) {
	x := [][]float32{{1}, {2}, {3}, {4}, {5}}
	y := [][]float32{{10}, {20}, {30}, {40}, {50}}
	sx1, sy1 := ShuffleData(newTestRNG(7), x, y)
	sx2, sy2 := ShuffleData(newTestRNG(7), x, y)
	for i := range sx1 {
		if sx1[i][0] != sx2[i][0] || sy1[i][0] != sy2[i][0] {
			t.Fatalf("ShuffleData with the same seed produced different orders at index %d", i)
		}
	}
}

func TestSplitDataRatios(t *testing.T) {
	n := 100
	x := make([][]float32, n)
	y := make([][]float32, n)
	for i := 0; i < n; i++ {
		x[i] = []float32{float32(i)}
		y[i] = []float32{float32(i)}
	}
	split := SplitData(newTestRNG(1), x, y, SplitConfig{TrainRatio: 0.7, ValRatio: 0.15, TestRatio: 0.15, Shuffle: true})
	if len(split.TrainX) != 70 || len(split.ValX) != 15 || len(split.TestX) != 15 {
		t.Fatalf("split sizes = (%d,%d,%d), want (70,15,15)", len(split.TrainX), len(split.ValX), len(split.TestX))
	}
}
