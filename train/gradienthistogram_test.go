// train/gradienthistogram_test.go
package train

import (
	"strings"
	"testing"

	"github.com/stolzmi/neugo/nn"
)

func TestGradientMagnitudeHistogramExcludesZeros(t *testing.T) {
	p := nn.NewParam(nn.NewTensor([]int{4}))
	p.Grad.Data[0] = 0
	p.Grad.Data[1] = 0
	p.Grad.Data[2] = 1
	p.Grad.Data[3] = 10
	_, counts := gradientMagnitudeHistogram([]*nn.Param{p}, 4)
	var total int
	for _, c := range counts {
		total += c
	}
	if total != 2 {
		t.Fatalf("total bucketed count = %d, want 2 (the two zero gradients must be excluded)", total)
	}
}

func TestGradientMagnitudeHistogramAllZeroReturnsNil(t *testing.T) {
	p := nn.NewParam(nn.NewTensor([]int{3}))
	bucketLogMin, counts := gradientMagnitudeHistogram([]*nn.Param{p}, 4)
	if bucketLogMin != nil || counts != nil {
		t.Errorf("got (%v, %v), want (nil, nil) when every gradient is zero", bucketLogMin, counts)
	}
}

func TestGradientMagnitudeHistogramBucketsSpanMinMax(t *testing.T) {
	// Gradients of magnitude 1 (log10=0) and 100 (log10=2) with 2 bins:
	// bucket 0 should hold the small one, bucket 1 the large one.
	p := nn.NewParam(nn.NewTensor([]int{2}))
	p.Grad.Data[0] = 1
	p.Grad.Data[1] = 100
	bucketLogMin, counts := gradientMagnitudeHistogram([]*nn.Param{p}, 2)
	if len(counts) != 2 {
		t.Fatalf("got %d buckets, want 2", len(counts))
	}
	if counts[0] != 1 || counts[1] != 1 {
		t.Errorf("counts = %v, want [1 1] (one small, one large gradient)", counts)
	}
	if bucketLogMin[0] != 0 {
		t.Errorf("bucketLogMin[0] = %v, want 0 (log10(1))", bucketLogMin[0])
	}
	if diff := bucketLogMin[1] - 1; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("bucketLogMin[1] = %v, want 1 (halfway between log10(1)=0 and log10(100)=2)", bucketLogMin[1])
	}
}

func TestRenderHistogramScalesLargestBucketToBarWidth(t *testing.T) {
	out := renderHistogram([]float64{0, 1}, []int{5, 10}, 20)
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2", len(lines))
	}
	// bucket 1 has the max count (10), so its bar should be exactly
	// barWidth (20) characters; bucket 0 (count 5) should be half that.
	if !strings.Contains(lines[1], strings.Repeat("#", 20)) {
		t.Errorf("largest bucket's line %q doesn't contain a full-width bar", lines[1])
	}
	if !strings.Contains(lines[0], strings.Repeat("#", 10)) {
		t.Errorf("half-count bucket's line %q doesn't contain a half-width bar", lines[0])
	}
}

func TestRenderHistogramEmptyInput(t *testing.T) {
	out := renderHistogram(nil, nil, 20)
	if !strings.Contains(out, "no nonzero gradients") {
		t.Errorf("renderHistogram(nil, nil, _) = %q, want a message about no nonzero gradients", out)
	}
}

func TestGradientHistogramCallbackRespectsPrintEvery(t *testing.T) {
	// Just a no-panic smoke test — the printing itself goes to stdout,
	// which isn't worth capturing/asserting on here; the bucketing and
	// rendering logic above already has direct unit coverage.
	cb := GradientHistogram(4, 2)
	p := nn.NewParam(nn.NewTensor([]int{2}))
	p.Grad.Data[0] = 1
	cb.OnEpochEnd(0, 1.0, nil, []*nn.Param{p}) // epoch 0: 0%2==0, prints
	cb.OnEpochEnd(1, 1.0, nil, []*nn.Param{p}) // epoch 1: 1%2!=0, skipped
}
