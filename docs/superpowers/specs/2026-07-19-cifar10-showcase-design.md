# CIFAR-10 Example Showcase — Design

Date: 2026-07-19
Status: approved (approach A + horizontal-flip augmentation)

## Goal

Turn `examples/cifar10_cnn` into a showcase of neugo's capabilities. The
current example peaks at ~51% validation accuracy, overfits after epoch ~11
(train loss 0.67 vs val loss 1.67), and uses only a small slice of the
library's API. The improved example should both score meaningfully better
(~65–70% test accuracy expected) and demonstrate the training
infrastructure the library actually has.

## Modes

- **Default (`go run ./examples/cifar10_cnn`)** — full overnight-scale
  showcase run: all 50,000 training images plus the official 10,000-image
  test batch, flip-augmented to ~100,000 training images.
- **`-quick`** — preserves today's fast behavior for smoke tests: one batch
  file, 5,000 images, 80/20 train/val split, 15 epochs, no augmentation.
  Runs in minutes.

Synthetic-data fallback (download failure) is kept unchanged.

## Data pipeline

1. Download/extract as today (`downloadAndExtractTarGz`, unchanged).
2. Full mode: `data.LoadCIFAR10BinaryBatch` over `data_batch_1..5.bin` for
   training; `data.LoadCIFAR10Binary("test_batch.bin")` for evaluation — a
   real train/test protocol instead of splitting one batch.
   Quick mode: `data_batch_1.bin` capped at 5,000 images, split 80/20 via
   `data.SplitImageData` as today.
3. `data.NormalizeImages` applied to the combined train+eval image slice
   (per-channel zero-mean/unit-std). Note: stats are computed over the
   passed slice; combining train+test introduces trivial leakage that is
   acceptable for a demo and keeps the API usage simple.
4. **Augmentation (full mode only):** every training image is duplicated
   as a horizontal mirror (example-local helper `flipHorizontal`), doubling
   the training set to ~100k images. Applied after normalization, before
   tensor conversion. Eval images are never augmented.

Memory note: the 100k-image training tensor is ~1.2 GB of float32 plus the
intermediate `[]*data.Image` slices — expect a few GB peak; fine on a
typical dev machine.

## Model (~190k params)

```
Sequential(input [N,32,32,3])
  Conv2DSame(rng, 3, 16, 3, HeInit)  + BatchNorm(16) + ReLU + MaxPool2D(2,2)  // 32→16
  Conv2DSame(rng, 16, 32, 3, HeInit) + BatchNorm(32) + ReLU + MaxPool2D(2,2)  // 16→8
  Conv2DSame(rng, 32, 64, 3, HeInit) + BatchNorm(64) + ReLU + MaxPool2D(2,2)  // 8→4
  Flatten
  Dropout(0.5)
  Linear(rng, 0, 128, HeInit) + GELU
  Linear(rng, 128, 10, XavierInit) + Softmax
```

Rationale: `Conv2DSame` keeps spatial dims (the old example silently
shrank them with valid convs); BatchNorm + Dropout target the observed
overfitting; GELU in the head demonstrates activation variety. Trainer
keeps the fused softmax+CCE path since Softmax stays last.

## Training

- `train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.CrossEntropy())`
- `train.ClipGrad(5)`
- Batch size 32; shuffle on; fixed seeds for reproducibility.
- Epochs: 40 full / 15 quick.
- Callbacks:
  - `train.ProgressBar(epochs, 1)`
  - `train.CosineAnnealing(opt, 1e-5, epochs)` — LR decays 1e-3 → 1e-5.
  - `train.EarlyStopping(6)` — monitors val loss, restores best weights.
  - `train.ModelCheckpoint("cifar_10", "loss", "min", true)` wired with
    `train.WithSaveFunc(nn.Save)` so the *best* epoch, not the last, lands
    on disk. The final unconditional `nn.Save` call is removed.
- `nn.Summary(model, inputShape)` printed before training starts.

## Reporting (unchanged mechanics, better data)

`hist.PlotLoss(60, 12)`, final `Evaluate` metrics line, and
`train.FormatConfusionMatrix` — all computed against the real held-out
test batch in full mode.

## Error handling

Same style as today: each failure prints a message and falls back (synthetic
data) or returns from `main`. Missing `test_batch.bin` in full mode falls
back to the quick-mode split rather than aborting.

## Testing / verification

- `go vet ./...` and the existing test suite must stay green (example is
  a `main` package; no new library code is added).
- One `-quick` end-to-end run verifies wiring, output, and checkpoint file.
- The full overnight run is launched separately and its report reviewed.
