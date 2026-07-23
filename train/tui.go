// train/tui.go
package train

import (
	"fmt"
	"math"
	"strings"

	"github.com/stolzmi/neugo/nn"
)

var sparkChars = []rune("▁▂▃▄▅▆▇█")

// renderSparkline maps each value in vals to one of 8 Unicode block-
// element levels, scaled between the slice's own min and max — a
// dependency-free sparkline, no plotting library needed. Flat input (all
// equal, span == 0) renders as the lowest level throughout rather than
// dividing by zero.
func renderSparkline(vals []float32) string {
	if len(vals) == 0 {
		return ""
	}
	minV, maxV := vals[0], vals[0]
	for _, v := range vals {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	span := maxV - minV
	var b strings.Builder
	for _, v := range vals {
		level := 0
		if span > 0 {
			level = int((v-minV)/span*float32(len(sparkChars)-1) + 0.5)
		}
		if level < 0 {
			level = 0
		}
		if level >= len(sparkChars) {
			level = len(sparkChars) - 1
		}
		b.WriteRune(sparkChars[level])
	}
	return b.String()
}

// gradNormOf computes the global L2 norm across every parameter's current
// gradient — a cheap, standard proxy for "is training blowing up/stalling"
// at a glance, without needing a GUI.
func gradNormOf(params []*nn.Param) float32 {
	var sumSq float64
	for _, p := range params {
		for _, g := range p.Grad.Data {
			sumSq += float64(g) * float64(g)
		}
	}
	return float32(math.Sqrt(sumSq))
}

// renderDashboardLines builds the TUI's two content lines — a stats
// summary and a loss sparkline over the most recent min(len(history),
// width) epochs — as plain strings, kept separate from TUICallback's
// actual terminal I/O so the rendering logic itself is easy to unit test
// without needing a real terminal or asserting on raw ANSI bytes.
func renderDashboardLines(epoch, totalEpochs int, loss, lr, gradNorm float32, history []float32, width int) [2]string {
	trimmed := history
	if len(trimmed) > width {
		trimmed = trimmed[len(trimmed)-width:]
	}
	stats := fmt.Sprintf("epoch %d/%d  loss %.4f  lr %.5f  |grad| %.4f", epoch+1, totalEpochs, loss, lr, gradNorm)
	return [2]string{stats, renderSparkline(trimmed)}
}

// TUICallback redraws a compact two-line terminal dashboard (current
// stats, and a sparkline of recent training loss) in place after every
// epoch, using only ANSI cursor-movement escapes — no external TUI
// library, consistent with this project's dependency-free design.
// Redirecting stdout to a file (rather than a real terminal) will show
// the raw escape sequences; use ProgressBarCallback instead for
// non-interactive output.
type TUICallback struct {
	BaseCallback
	opt         Optimizer
	totalEpochs int
	width       int
	history     []float32
	drawn       bool
}

// TUI creates a live training dashboard callback wrapping opt (to read
// its current LR) and totalEpochs (for the "epoch N/total" line). The
// sparkline shows up to the most recent 60 epochs' training loss.
func TUI(opt Optimizer, totalEpochs int) *TUICallback {
	return &TUICallback{opt: opt, totalEpochs: totalEpochs, width: 60}
}

func (c *TUICallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	c.history = append(c.history, trainLoss)
	lines := renderDashboardLines(epoch, c.totalEpochs, trainLoss, c.opt.GetLR(), gradNormOf(params), c.history, c.width)
	if c.drawn {
		fmt.Printf("\033[%dA", len(lines)) // cursor up to the first dashboard line
	}
	for _, line := range lines {
		fmt.Printf("\033[K%s\n", line) // clear the line, then print it
	}
	c.drawn = true
}
