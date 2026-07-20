# CIFAR-10 Example: Confusion Matrix + Training Progress — Design

Date: 2026-07-19
Status: Approved (design), pending implementation plan

## 1. Background

`examples/cifar10_cnn/main.go` trains a small CNN on up to 5000 real CIFAR-10
images (synthetic fallback: 200 patterned images) for 15 epochs and prints a
single train-set accuracy line. The library already computes everything needed
for a richer report — `train.Metrics.ConfusionMatrix` (`train/metrics.go`),
per-epoch `train.ProgressBar` output, and the `train.History` accumulator
(`train/callback.go`) — the example just doesn't use any of it.

Decisions from brainstorming:

- No image display (user explicitly dropped the "show a few images" idea).
- Confusion matrix and final metrics are reported on a held-out 20% validation
  split, not on training data.
- Training progress = existing per-epoch `train.ProgressBar` lines plus an
  ASCII loss curve printed from `History` after training.
- The confusion-matrix formatter and the curve plotter are reusable library
  helpers in `train`, not example-local code (user choice), so other examples
  can use them later.

## 2. Library additions: `train/report.go`

One new file with two text-reporting helpers, in the package's existing
"plain data in, string out" style (no stdout writes from library code —
`ProgressBar` and `nn.Summary` remain the only printers; these return strings
for the caller to print).

### `func FormatConfusionMatrix(m *Metrics, classNames []string) string`

- Renders `m.ConfusionMatrix` as an aligned text table: rows = actual class,
  columns = predicted class, cells = counts. Header row plus a row label per
  actual class.
- Column width adapts to the longest class name and the largest count, so the
  table stays aligned for CIFAR-10's 10 classes.
- If `classNames` is nil or its length doesn't match the matrix dimension,
  falls back to `c0`, `c1`, ... labels.
- Works for any NxN matrix including the 2x2 binary case; returns an empty
  string for a nil/empty matrix.

### `func (h *History) PlotLoss(width, height int) string`

- ASCII line plot of `h.TrainLoss` over epochs; when `h.ValLoss` is present,
  plotted as a second series with a different glyph, plus a one-line legend.
- Y-axis scaled to data min/max with the min/max values printed next to the
  plot; X-axis = epoch index.
- Degenerate cases are handled, not crashed on: empty history returns an
  empty string; a single point or a constant series must not divide by zero.

Both helpers get unit tests in `train/report_test.go`: table alignment and
fallback labels, plot dimensions, and each degenerate case.

## 3. Example changes: `examples/cifar10_cnn/main.go`

Flow after loading the dataset:

1. Split images/labels 80/20 with shuffle via the existing
   `data.SplitImageData` (its `SplitConfig{TrainRatio: 0.8, ValRatio: 0.2,
   Shuffle: true}`), seeded so runs are reproducible.
2. `Fit` on the 80% with `train.Callbacks(train.ProgressBar(epochs, 1))` so
   every epoch prints `loss / val_loss / val_acc` (the existing
   `train.Validation` option supplies the held-out split, which also makes
   the ProgressBar's val columns work), and capture the returned `*History`
   instead of discarding it with `_`.
3. Print `hist.PlotLoss(60, 12)`.
4. `trainer.Evaluate` on the held-out 20% and print loss, accuracy,
   precision, recall, F1, then `FormatConfusionMatrix` with
   `dataset.ClassNames`.
5. Keep the final `nn.Save(model, "cifar_10")` unchanged.

The model definition, data download/loading, and tensor conversion helpers
are untouched. No new dependencies (stdlib only).

## 4. Out of scope

- Rendering sample images (explicitly dropped by the user).
- Changes to `Metrics`, `History`, the trainer loop, or other examples.
- Normalized/percentage confusion matrices, per-class precision tables,
  colorized terminal output.

## 5. Verification

- `go test ./train/...` — new helper tests plus the existing suite stay green.
- `go vet ./...`.
- `go run ./examples/cifar10_cnn` (dataset already present under
  `dataset/cifar10/`) — eyeball that epoch lines, the curve, and the
  10x10 confusion table render sensibly.
