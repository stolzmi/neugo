package data

import "testing"

func TestBalanceDatasetOversampleIncreasesMinorityCount(t *testing.T) {
	// 9 majority (label 0), 1 minority (label 1)
	features := make([][]float32, 10)
	labels := make([][]float32, 10)
	for i := 0; i < 10; i++ {
		features[i] = []float32{float32(i)}
		if i == 0 {
			labels[i] = []float32{1}
		} else {
			labels[i] = []float32{0}
		}
	}
	bx, by := BalanceDataset(newTestRNG(1), features, labels, 0.4, true)
	if len(bx) != len(by) {
		t.Fatalf("balanced features/labels length mismatch: %d vs %d", len(bx), len(by))
	}
	var minority int
	for _, l := range by {
		if l[0] > 0.5 {
			minority++
		}
	}
	dist := AnalyzeClassDistribution(by, 0.5)
	if !dist.IsBalanced && minority <= 1 {
		t.Fatalf("BalanceDataset did not increase minority count: got %d minority samples out of %d", minority, len(by))
	}
}
