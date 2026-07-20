package train

import (
	"fmt"
	"strings"
)

// FormatConfusionMatrix renders m.ConfusionMatrix as an aligned text table
// with rows = actual class and columns = predicted class. classNames labels
// both axes; when nil or the wrong length, classes are labeled c0..cN-1.
// Returns "" for a nil Metrics or an empty matrix.
func FormatConfusionMatrix(m *Metrics, classNames []string) string {
	if m == nil || len(m.ConfusionMatrix) == 0 {
		return ""
	}
	cm := m.ConfusionMatrix
	n := len(cm)

	labels := classNames
	if len(labels) != n {
		labels = make([]string, n)
		for i := range labels {
			labels[i] = fmt.Sprintf("c%d", i)
		}
	}

	w := len("actual\\pred")
	maxCount := 0
	for i := 0; i < n; i++ {
		if len(labels[i]) > w {
			w = len(labels[i])
		}
		for j := 0; j < n; j++ {
			if cm[i][j] > maxCount {
				maxCount = cm[i][j]
			}
		}
	}
	if cw := len(fmt.Sprintf("%d", maxCount)); cw > w {
		w = cw
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%*s", w, "actual\\pred")
	for j := 0; j < n; j++ {
		fmt.Fprintf(&b, " %*s", w, labels[j])
	}
	b.WriteByte('\n')
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "%*s", w, labels[i])
		for j := 0; j < n; j++ {
			fmt.Fprintf(&b, " %*d", w, cm[i][j])
		}
		if i < n-1 {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// PlotLoss renders an ASCII line plot of the training loss recorded in h,
// with the validation loss as a second series when one was recorded per
// epoch. width and height are the plot area in characters; the returned
// string adds a min/max header, an epoch axis line, and a legend.
// Returns "" for a nil History, no recorded losses, or width/height < 2.
func (h *History) PlotLoss(width, height int) string {
	if h == nil || len(h.TrainLoss) == 0 || width < 2 || height < 2 {
		return ""
	}
	train := h.TrainLoss
	var val []float32
	if len(h.ValLoss) == len(train) {
		val = h.ValLoss
	}

	lo, hi := train[0], train[0]
	for _, series := range [][]float32{train, val} {
		for _, v := range series {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
	}
	span := hi - lo
	if span == 0 {
		span = 1 // constant series: all points land on one row instead of NaN
	}

	grid := make([][]byte, height)
	for r := range grid {
		grid[r] = make([]byte, width)
		for c := range grid[r] {
			grid[r][c] = ' '
		}
	}
	plot := func(series []float32, glyph byte) {
		for i, v := range series {
			x := 0
			if len(series) > 1 {
				x = i * (width - 1) / (len(series) - 1)
			}
			y := int(float32(height-1) * (1 - (v - lo) / span))
			grid[y][x] = glyph
		}
	}
	plot(train, '*')
	if val != nil {
		plot(val, 'o')
	}

	var b strings.Builder
	fmt.Fprintf(&b, "loss (max %.4f, min %.4f)\n", hi, lo)
	for _, row := range grid {
		b.WriteByte('|')
		b.Write(row)
		b.WriteByte('\n')
	}
	fmt.Fprintf(&b, "+%s\n", strings.Repeat("-", width))
	fmt.Fprintf(&b, "epoch 0..%d\n", len(train)-1)
	if val != nil {
		b.WriteString("* train loss   o val loss")
	} else {
		b.WriteString("* train loss")
	}
	return b.String()
}
