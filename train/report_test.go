package train

import (
	"strings"
	"testing"
)

func TestFormatConfusionMatrixExactLayout(t *testing.T) {
	m := &Metrics{ConfusionMatrix: [][]int{{1, 0}, {1, 2}}}
	got := FormatConfusionMatrix(m, []string{"no", "yes"})
	want := "actual\\pred          no         yes\n" +
		"         no           1           0\n" +
		"        yes           1           2"
	if got != want {
		t.Errorf("got:\n%s\nwant:\n%s", got, want)
	}
}

func TestFormatConfusionMatrixFallbackLabels(t *testing.T) {
	m := &Metrics{ConfusionMatrix: [][]int{{3, 1}, {0, 4}}}
	out := FormatConfusionMatrix(m, nil)
	if !strings.Contains(out, "c0") || !strings.Contains(out, "c1") {
		t.Errorf("expected fallback labels c0/c1 in:\n%s", out)
	}
	out = FormatConfusionMatrix(m, []string{"only-one"})
	if !strings.Contains(out, "c0") {
		t.Errorf("expected fallback labels for wrong-length classNames in:\n%s", out)
	}
}

func TestFormatConfusionMatrixEmpty(t *testing.T) {
	if got := FormatConfusionMatrix(&Metrics{}, []string{"a"}); got != "" {
		t.Errorf("expected empty string for empty matrix, got %q", got)
	}
	if got := FormatConfusionMatrix(nil, nil); got != "" {
		t.Errorf("expected empty string for nil metrics, got %q", got)
	}
}

func TestPlotLossDimensions(t *testing.T) {
	h := &History{TrainLoss: []float32{2.3, 1.8, 1.5, 1.3, 1.2}}
	out := h.PlotLoss(20, 6)
	lines := strings.Split(out, "\n")
	if len(lines) != 10 {
		t.Fatalf("expected 10 lines, got %d:\n%s", len(lines), out)
	}
	if lines[0] != "loss (max 2.3000, min 1.2000)" {
		t.Errorf("header = %q", lines[0])
	}
	for i := 1; i <= 6; i++ {
		if len(lines[i]) != 21 {
			t.Errorf("grid line %d length = %d, want 21 (%q)", i, len(lines[i]), lines[i])
		}
	}
	if lines[7] != "+"+strings.Repeat("-", 20) {
		t.Errorf("axis line = %q", lines[7])
	}
	if lines[8] != "epoch 0..4" {
		t.Errorf("epoch line = %q", lines[8])
	}
	if lines[9] != "* train loss" {
		t.Errorf("legend = %q", lines[9])
	}
}

func TestPlotLossWithValidation(t *testing.T) {
	h := &History{
		TrainLoss: []float32{2.3, 1.8, 1.5},
		ValLoss:   []float32{2.4, 1.9, 1.6},
	}
	out := h.PlotLoss(20, 6)
	if !strings.Contains(out, "* train loss   o val loss") {
		t.Errorf("expected two-series legend in:\n%s", out)
	}
}

func TestPlotLossDegenerate(t *testing.T) {
	if got := (&History{}).PlotLoss(20, 6); got != "" {
		t.Errorf("empty history: expected \"\", got %q", got)
	}
	var nilHist *History
	if got := nilHist.PlotLoss(20, 6); got != "" {
		t.Errorf("nil history: expected \"\", got %q", got)
	}
	constant := &History{TrainLoss: []float32{0.5, 0.5, 0.5}}
	if got := constant.PlotLoss(10, 4); got == "" {
		t.Error("constant series: expected a plot, got empty string")
	}
	single := &History{TrainLoss: []float32{0.5}}
	if got := single.PlotLoss(10, 4); got == "" {
		t.Error("single point: expected a plot, got empty string")
	}
}
