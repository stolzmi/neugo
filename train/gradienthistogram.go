// train/gradienthistogram.go
package train

import (
	"fmt"
	"math"
	"strings"

	"github.com/stolzmi/neugo/nn"
)

// gradientMagnitudeHistogram buckets the base-10 log magnitude of every
// nonzero gradient element across params into bins equal-width buckets
// spanning the observed [min, max] log-magnitude range — log-scale
// because gradient magnitudes routinely span many orders of magnitude,
// where a linear histogram would just show one occupied bucket. Zero
// gradients are excluded (log(0) is undefined, and a mostly-zero
// embedding-table gradient would otherwise swamp the real signal in a
// single spike). Returns nil, nil when every gradient is zero.
func gradientMagnitudeHistogram(params []*nn.Param, bins int) (bucketLogMin []float64, counts []int) {
	if bins <= 0 {
		bins = 1
	}
	var logs []float64
	for _, p := range params {
		for _, g := range p.Grad.Data {
			if g == 0 {
				continue
			}
			logs = append(logs, math.Log10(math.Abs(float64(g))))
		}
	}
	if len(logs) == 0 {
		return nil, nil
	}

	minLog, maxLog := logs[0], logs[0]
	for _, l := range logs {
		if l < minLog {
			minLog = l
		}
		if l > maxLog {
			maxLog = l
		}
	}
	span := maxLog - minLog

	counts = make([]int, bins)
	bucketLogMin = make([]float64, bins)
	for i := range bucketLogMin {
		bucketLogMin[i] = minLog + span*float64(i)/float64(bins)
	}
	for _, l := range logs {
		idx := 0
		if span > 0 {
			idx = int((l - minLog) / span * float64(bins))
		}
		if idx >= bins {
			idx = bins - 1
		}
		if idx < 0 {
			idx = 0
		}
		counts[idx]++
	}
	return bucketLogMin, counts
}

// renderHistogram draws an ASCII bar chart of counts, one line per
// bucket, each bar scaled so the largest bucket fills barWidth characters
// — the same relative-scaling idea as renderSparkline, just as a
// multi-line bar chart instead of a single line of block characters.
func renderHistogram(bucketLogMin []float64, counts []int, barWidth int) string {
	if len(counts) == 0 {
		return "(no nonzero gradients)\n"
	}
	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}
	var b strings.Builder
	for i, c := range counts {
		barLen := 0
		if maxCount > 0 {
			barLen = c * barWidth / maxCount
		}
		fmt.Fprintf(&b, "1e%+.1f  %s (%d)\n", bucketLogMin[i], strings.Repeat("#", barLen), c)
	}
	return b.String()
}

// GradientHistogramCallback prints an ASCII histogram of gradient
// magnitudes (log10 scale) every PrintEvery epochs — a quick way to spot
// vanishing (everything clustered at very negative exponents) or
// exploding (a long tail at large positive exponents) gradients without
// needing a GUI like TensorBoard.
type GradientHistogramCallback struct {
	BaseCallback
	Bins       int
	PrintEvery int
}

// GradientHistogram creates a callback with bins histogram buckets,
// printing every printEvery epochs (printEvery <= 0 disables printing
// entirely, effectively a no-op callback).
func GradientHistogram(bins, printEvery int) *GradientHistogramCallback {
	return &GradientHistogramCallback{Bins: bins, PrintEvery: printEvery}
}

func (g *GradientHistogramCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	if g.PrintEvery <= 0 || epoch%g.PrintEvery != 0 {
		return
	}
	bucketLogMin, counts := gradientMagnitudeHistogram(params, g.Bins)
	fmt.Printf("gradient magnitude histogram (epoch %d):\n%s", epoch+1, renderHistogram(bucketLogMin, counts, 40))
}
