# CIFAR-10 Example Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich `examples/cifar10_cnn` with per-epoch progress, an ASCII loss curve, and a confusion matrix on a held-out 20% validation split.

**Architecture:** Two reusable text-reporting helpers in a new `train/report.go` (`FormatConfusionMatrix`, `History.PlotLoss`) that take plain data and return strings; the example wires in the existing `train.ProgressBar` callback, an 80/20 split via `data.SplitImageData`, and prints the two reports after training.

**Tech Stack:** Go (stdlib only), module `neugo`, spec: `docs/superpowers/specs/2026-07-19-cifar10-example-reporting-design.md`.

## Global Constraints

- Zero new dependencies — stdlib only (`fmt`, `strings` in the library; `math/rand` added to the example).
- Library helpers return strings; they never write to stdout (`train.ProgressBar` and `nn.Summary` remain the only library printers).
- No changes to `train.Metrics`, `train.History`, the trainer loop, or any other example.
- Tests use the plain `testing` package with hand-computed expectations, matching `train/metrics_test.go` style.
- All metrics stay `float32` (project convention).

---

### Task 1: `FormatConfusionMatrix` in `train/report.go`

**Files:**
- Create: `train/report.go`
- Test: `train/report_test.go`

**Interfaces:**
- Consumes: `train.Metrics` (`train/metrics.go`) — field `ConfusionMatrix [][]int`.
- Produces: `func FormatConfusionMatrix(m *Metrics, classNames []string) string` — rows = actual class, columns = predicted class; nil receiver or empty matrix returns `""`; nil or wrong-length `classNames` falls back to `c0..cN-1`. Used by Task 3.

- [ ] **Step 1: Write the failing test**

Create `train/report_test.go`:

```go
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./train/ -run TestFormatConfusionMatrix -v`
Expected: FAIL — compile error `undefined: FormatConfusionMatrix`

- [ ] **Step 3: Write minimal implementation**

Create `train/report.go`:

```go
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./train/ -run TestFormatConfusionMatrix -v`
Expected: PASS — `ok  	neugo/train`

- [ ] **Step 5: Commit**

```bash
git add train/report.go train/report_test.go
git commit -m "feat(train): add FormatConfusionMatrix text table renderer"
```

---

### Task 2: `History.PlotLoss` in `train/report.go`

**Files:**
- Modify: `train/report.go` (append the method)
- Test: `train/report_test.go` (append tests)

**Interfaces:**
- Consumes: `train.History` (`train/callback.go:37`) — fields `TrainLoss []float32`, `ValLoss []float32`.
- Produces: `func (h *History) PlotLoss(width, height int) string` — ASCII plot of train loss (`*`), plus val loss (`o`) when `len(ValLoss) == len(TrainLoss)`; returns `""` for nil receiver, empty `TrainLoss`, or `width`/`height` < 2. Used by Task 3.

- [ ] **Step 1: Write the failing test**

Append to `train/report_test.go`:

```go
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./train/ -run TestPlotLoss -v`
Expected: FAIL — compile error `h.PlotLoss undefined (type *History has no field or method PlotLoss)`

- [ ] **Step 3: Write minimal implementation**

Append to `train/report.go`:

```go
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./train/ -v -run 'TestPlotLoss|TestFormatConfusionMatrix'`
Expected: PASS — all Task 1 + Task 2 tests `ok  	neugo/train`

- [ ] **Step 5: Run the full train suite and vet**

Run: `go test ./train/ && go vet ./train/`
Expected: PASS, no vet output

- [ ] **Step 6: Commit**

```bash
git add train/report.go train/report_test.go
git commit -m "feat(train): add History.PlotLoss ASCII loss curve"
```

---

### Task 3: Rewrite `examples/cifar10_cnn/main.go`

**Files:**
- Modify: `examples/cifar10_cnn/main.go` (only the `main` function and the import block; helpers `cifarImagesToTensor`, `cifarLabelsToTensor`, `syntheticCIFAR10`, `downloadAndExtractTarGz`, `loadRealOrSynthetic` stay byte-identical)

**Interfaces:**
- Consumes: `data.SplitImageData(rng *rand.Rand, images []*Image, labels [][]float32, config SplitConfig) ImageSplit` (`data/image.go:154`) with `data.SplitConfig{TrainRatio, ValRatio, TestRatio float64; Shuffle bool}` (`data/preprocessing.go:129`); `train.Validation(x, y *nn.Tensor) FitOption` (`train/trainer.go:56`); `train.ProgressBar(totalEpochs, printEvery int)` (`train/callback.go:172`); `train.FormatConfusionMatrix` and `(*History).PlotLoss` from Tasks 1–2.
- Produces: nothing consumed elsewhere — this is the demo binary.

- [ ] **Step 1: Replace the import block and `main`**

In `examples/cifar10_cnn/main.go`, replace the import block with:

```go
import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"neugo/data"
	"neugo/nn"
	"neugo/train"
)
```

Replace the `epochs`-less const block's trailing part by adding one const (keep the existing `maxRealSamples` comment untouched):

```go
const (
	cifar10URL      = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
	cifar10Dir      = "dataset/cifar10"
	cifar10BatchOne = "dataset/cifar10/data_batch_1.bin"
	// maxRealSamples caps how many real images we train on in this demo.
	// The full batch (10000 images) trains correctly but takes a long time
	// with this library's pure-Go, non-SIMD conv/pooling loops — this cap
	// keeps `go run` fast while still exercising real data end to end.
	maxRealSamples = 5000
	epochs         = 15
)
```

Replace the whole `main` function with:

```go
func main() {
	dataset := loadRealOrSynthetic()

	split := data.SplitImageData(rand.New(rand.NewSource(42)), dataset.Images, dataset.Labels,
		data.SplitConfig{TrainRatio: 0.8, ValRatio: 0.2, Shuffle: true})
	fmt.Printf("training on %d images, validating on %d held-out images\n", len(split.TrainX), len(split.ValX))

	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{len(split.TrainX), 32, 32, 3},
		nn.Conv2D(rng, 3, 8, 3, nn.HeInit()),
		nn.ReLU(),
		nn.MaxPool2D(2, 2),
		nn.Conv2D(rng, 8, 16, 3, nn.HeInit()),
		nn.ReLU(),
		nn.MaxPool2D(2, 2),
		nn.Flatten(),
		nn.Linear(rng, 0, 64, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 64, 10, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}

	x := cifarImagesToTensor(split.TrainX)
	y := cifarLabelsToTensor(split.TrainY)
	valX := cifarImagesToTensor(split.ValX)
	valY := cifarLabelsToTensor(split.ValY)

	trainer := train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.CrossEntropy())
	hist, err := trainer.Fit(x, y,
		train.Epochs(epochs), train.BatchSize(16), train.Shuffle(true), train.Seed(2),
		train.Validation(valX, valY),
		train.Callbacks(train.ProgressBar(epochs, 1)),
	)
	if err != nil {
		fmt.Println("fit:", err)
		return
	}

	fmt.Println(hist.PlotLoss(60, 12))

	metrics, err := trainer.Evaluate(valX, valY)
	if err != nil {
		fmt.Println("evaluate:", err)
		return
	}
	fmt.Printf("validation: loss %.4f - acc %.2f%% - precision %.2f - recall %.2f - f1 %.2f\n",
		metrics.Loss, metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1Score)
	fmt.Println(train.FormatConfusionMatrix(&metrics, dataset.ClassNames))
	nn.Save(model, "cifar_10")
}
```

- [ ] **Step 2: Build and vet**

Run: `go build ./... && go vet ./...`
Expected: both succeed silently

- [ ] **Step 3: Run the example and verify the output**

Run from the repo root (dataset is already present under `dataset/cifar10/`; training 4000 images × 15 epochs with pure-Go convs takes minutes — use a background task or a long timeout):

Run: `go run ./examples/cifar10_cnn`
Expected, in order:
1. `using 5000 real CIFAR-10 images`
2. `training on 4000 images, validating on 1000 held-out images`
3. 15 lines `Epoch N/15 - loss: ... - val_loss: ... - val_acc: ...`
4. The ASCII plot: header `loss (max ..., min ...)`, 12 `|`-prefixed grid rows, axis line, `epoch 0..14`, legend `* train loss   o val loss`
5. `validation: loss ... - acc ...% - precision ... - recall ... - f1 ...`
6. A 10×10 confusion table with CIFAR-10 class names on both axes

- [ ] **Step 4: Run the whole test suite**

Run: `go test ./...`
Expected: PASS for all packages

- [ ] **Step 5: Commit**

```bash
git add examples/cifar10_cnn/main.go
git commit -m "docs(examples): enrich cifar10_cnn with progress, loss curve, confusion matrix"
```

---

## Self-Review Notes

- Spec coverage: §2 library helpers → Tasks 1–2; §3 example changes → Task 3; §5 verification → Task 2 Step 5, Task 3 Steps 2–4. Image display intentionally absent (dropped in spec §4).
- Type consistency: `FormatConfusionMatrix(m *Metrics, classNames []string) string` and `func (h *History) PlotLoss(width, height int) string` are spelled identically in Tasks 1, 2, and 3; Task 3 passes `&metrics` because `Evaluate` returns a `Metrics` value (`train/trainer.go:212`).
- Commits: the user declined committing the spec doc, so confirm once before the first commit step instead of committing silently.
