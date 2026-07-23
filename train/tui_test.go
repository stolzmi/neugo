// train/tui_test.go
package train

import (
	"strings"
	"testing"

	"github.com/stolzmi/neugo/nn"
)

func TestRenderSparklineFlatInputUsesLowestLevel(t *testing.T) {
	got := renderSparkline([]float32{1, 1, 1, 1})
	want := strings.Repeat(string(sparkChars[0]), 4)
	if got != want {
		t.Errorf("renderSparkline(flat) = %q, want %q", got, want)
	}
}

func TestRenderSparklineMinMaxHitEndpoints(t *testing.T) {
	got := []rune(renderSparkline([]float32{0, 10}))
	if got[0] != sparkChars[0] {
		t.Errorf("first char = %q, want lowest level %q", string(got[0]), string(sparkChars[0]))
	}
	if got[1] != sparkChars[len(sparkChars)-1] {
		t.Errorf("second char = %q, want highest level %q", string(got[1]), string(sparkChars[len(sparkChars)-1]))
	}
}

func TestRenderSparklineEmptyInput(t *testing.T) {
	if got := renderSparkline(nil); got != "" {
		t.Errorf("renderSparkline(nil) = %q, want empty", got)
	}
}

func TestRenderSparklineMonotonicWithValue(t *testing.T) {
	// Strictly increasing input should produce a non-decreasing sequence
	// of levels (allowing repeats where the 8-level quantization can't
	// distinguish two close values).
	vals := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	levels := make([]int, len(vals))
	for i, r := range []rune(renderSparkline(vals)) {
		for lvl, c := range sparkChars {
			if c == r {
				levels[i] = lvl
			}
		}
	}
	for i := 1; i < len(levels); i++ {
		if levels[i] < levels[i-1] {
			t.Fatalf("levels not monotonic: %v", levels)
		}
	}
}

func TestRenderDashboardLinesTruncatesToWidth(t *testing.T) {
	history := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	lines := renderDashboardLines(9, 10, 10, 0.01, 1.5, history, 3)
	sparkRunes := []rune(lines[1])
	if len(sparkRunes) != 3 {
		t.Fatalf("sparkline has %d chars, want 3 (width-truncated to the most recent 3 values)", len(sparkRunes))
	}
	// The most recent 3 values are [8, 9, 10] — strictly increasing, so
	// the truncated sparkline (not the full 10-value history) must be
	// increasing here too, proving it used the tail, not the head.
	full := renderSparkline([]float32{8, 9, 10})
	if lines[1] != full {
		t.Errorf("sparkline = %q, want %q (sparkline of the most recent 3 values)", lines[1], full)
	}
}

func TestRenderDashboardLinesStatsLineContent(t *testing.T) {
	lines := renderDashboardLines(4, 10, 0.1234, 0.02, 1.5, []float32{0.1234}, 60)
	stats := lines[0]
	for _, want := range []string{"epoch 5/10", "loss 0.1234", "lr 0.02000", "|grad| 1.5000"} {
		if !strings.Contains(stats, want) {
			t.Errorf("stats line %q missing %q", stats, want)
		}
	}
}

func TestGradNormOfComputesGlobalL2Norm(t *testing.T) {
	p1 := nn.NewParam(nn.NewTensor([]int{2}))
	p1.Grad.Data[0], p1.Grad.Data[1] = 3, 0
	p2 := nn.NewParam(nn.NewTensor([]int{2}))
	p2.Grad.Data[0], p2.Grad.Data[1] = 4, 0
	got := gradNormOf([]*nn.Param{p1, p2})
	if diff := got - 5.0; diff > 1e-5 || diff < -1e-5 {
		t.Errorf("gradNormOf = %v, want 5.0 (3-4-5 triangle)", got)
	}
}

func TestTUICallbackOnEpochEndDoesNotPanic(t *testing.T) {
	opt := SGD(0.1)
	tui := TUI(opt, 10)
	p := nn.NewParam(nn.NewTensor([]int{2}))
	p.Grad.Data[0] = 1
	// Two calls exercise both the "first draw" and "redraw" branches.
	tui.OnEpochEnd(0, 1.0, nil, []*nn.Param{p})
	tui.OnEpochEnd(1, 0.5, nil, []*nn.Param{p})
	if len(tui.history) != 2 {
		t.Errorf("history has %d entries, want 2", len(tui.history))
	}
}
