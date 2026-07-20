# NeuGo

A dependency-free neural network library for Go. Everything is a
`Module` — dense layers, convolutions, normalization, attention, and
activations all compose the same way through `Sequential`, and one
`Trainer` handles fitting, prediction, and evaluation for all of them.
Trained models can be exported to standalone Go source, served over
HTTP with hot-swap and online learning, and tuned with parallel
hyperparameter search.

## Install

    go get neugo

Requires Go 1.22+. No third-party dependencies — only the standard library.

## Quickstart

```go
package main

import (
    "fmt"
    "neugo/nn"
    "neugo/train"
)

func main() {
    rng := nn.NewRNG(1)
    model, err := nn.Sequential([]int{4, 2},
        nn.Linear(rng, 2, 8, nn.HeInit()),
        nn.ReLU(),
        nn.Linear(rng, 8, 1, nn.XavierInit()),
        nn.Sigmoid(),
    )
    if err != nil {
        panic(err)
    }

    x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
    y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

    trainer := train.New(model, train.Adam(0.05, 0.9, 0.999, 1e-8), train.BCELoss())
    hist, err := trainer.Fit(x, y, train.Epochs(2000), train.BatchSize(4), train.Seed(1))
    if err != nil {
        panic(err)
    }
    fmt.Println("final loss:", hist.TrainLoss[len(hist.TrainLoss)-1])
}
```

## Examples

Nine runnable examples live in `examples/` — run any of them with
`go run ./examples/<name>`:

- `xor` — the Quickstart above, end to end.
- `wine_quality` — a real dataset (`dataset/wine_quality/winequality-red.csv`).
- `fashion_mnist` — convolutional; falls back to synthetic data unless
  you've downloaded a Fashion-MNIST CSV yourself.
- `cifar10_cnn` — the showcase example: a BatchNorm/Dropout/GELU CNN
  with flip augmentation, cosine LR annealing, early stopping, and
  metadata-bundled checkpointing, trained on the full 50k-image dataset
  plus the official test batch (downloaded from `cs.toronto.edu` on
  first run, ~170MB). Pass `-quick` for a fast smoke-test path (one
  batch, 5k images, no augmentation) instead of the full run.
- `cifar100_cnn` — convolutional; downloads and extracts the real
  dataset (~160MB) on first run, training on a capped subset for speed,
  and falls back to synthetic data if the download fails (e.g. no
  network).
- `callbacks` — early stopping, checkpointing, LR scheduling, progress
  reporting.
- `crossval` — k-fold cross-validation.
- `serve_xor` — HTTP serving with hot-swap, metrics, and online learning.
- `tune_wine` — hyperparameter search with ASHA pruning.

## Features

- **Modules** (`nn`): `Linear` (rank-agnostic — accepts any `[...,
  features]` input, not just `[batch, features]`), `Conv2D`/
  `Conv2DSame`/`Conv2DStrided`, `Conv1D`/`Conv1DSame`/`Conv1DStrided`,
  `ConvTranspose2D`, `MaxPool2D`, `AvgPool2D`, `Flatten`, `Dropout`,
  `BatchNorm`, `LayerNorm`, `GroupNorm`, `Embedding`, activations
  (`ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `GELU`, `Softmax`) — all
  compose via `Sequential`, which validates the whole chain's shapes at
  construction.
- **Composite modules**: `Residual(shortcut, inner...)` for ResNet-style
  skip connections (identity or projection shortcut); `Frozen(module)`
  excludes a layer's weights from optimizer updates for fine-tuning,
  while gradients still flow through it to earlier layers.
- **Attention** (`nn`): `MultiHeadAttention` (self-attention, causal or
  non-causal masking, implements `Module` so it composes normally),
  `CrossAttention` (query/context attention with independent sequence
  lengths — takes two inputs, so it's called directly rather than
  composed via `Sequential`), `PositionalEmbedding`, and
  `TransformerBlock` — a constructor (not a new type) assembling a full
  attention + feed-forward encoder block from the above; since it
  returns a `*SequentialModel`, which already implements `Module`,
  blocks stack by simply listing several `TransformerBlock(...)` calls
  inside an outer `Sequential`.
- **Initializers**: Xavier, He, Zeros, Uniform, Normal — explicit
  `*rand.Rand` throughout, no global RNG state anywhere in the library.
- **Training** (`train`): one `Trainer.Fit` loop with per-epoch shuffling,
  batching, optional gradient clipping, and validation metrics.
  Optimizers: SGD, Momentum, Adam, AdamW (decoupled weight decay),
  RMSprop, plus a `ClipNorm` wrapper. Losses: MSE, MAE, BCE,
  CrossEntropy (with a fused softmax+cross-entropy gradient shortcut
  when the model ends in `Softmax`).
- **Callbacks**: `History` (always returned by `Fit`), `EarlyStopping`
  (with in-memory best-weight restore), `ModelCheckpoint`, `ProgressBar`,
  and five LR schedulers (`StepDecay`, `ExponentialDecay`,
  `CosineAnnealing`, `Warmup`, `ReduceLROnPlateau`).
- **Evaluation**: `Trainer.Evaluate` returns accuracy/precision/recall/F1/
  confusion-matrix `Metrics`, macro-averaged for multiclass;
  `train.KFoldSplits`/`StratifiedKFoldSplits`/`CrossValidate` for
  cross-validation; `train.FormatConfusionMatrix` and `History.PlotLoss`
  for terminal-friendly reporting.
- **Serialization**: `nn.Save`/`nn.Load` — one JSON format for any
  module tree, from a single dense layer to a full Transformer block;
  `nn.SaveWithMetadata`/`nn.LoadWithMetadata` additionally bundle input
  shape, class names, and per-channel normalization stats with the
  weights, so a saved file is self-sufficient for inference; `Marshal`/
  `Unmarshal`/`Clone` for in-memory copies.
- **Performance**: the hot paths in `Conv2D`, `Conv1D`, `Linear`,
  `BatchNorm`/`GroupNorm`/`LayerNorm`, and the activations are
  batch/row-parallel across all CPU cores, with a GEMM-style,
  cache-friendly inner loop for the convolutions — a ~19x wall-clock
  improvement over a naive single-threaded implementation on the
  `cifar10_cnn` benchmark. Run `go test ./nn/ -bench . -benchmem` to
  measure on your own machine.
- **`data`**: CSV loading, z-score/min-max normalization, train/val/test
  splitting, class balancing (oversample/undersample), horizontal-flip
  augmentation, MNIST-style and CIFAR-10/CIFAR-100 image loaders — all
  with explicit `*rand.Rand`, no global state.
- **Export** (`export`): Convert trained models to standalone Go source code
  with zero dependencies. Single-file inference functions work anywhere Go
  runs — native, WASM, TinyGo. Bit-exact parity with training engine.
  See [`docs/EXPORT_GUIDE.md`](docs/EXPORT_GUIDE.md).
- **Serve** (`serve`): Hot-swap model serving with online learning. Stateless
  HTTP API, Prometheus metrics, holdout gate for validation, automatic rollback.
  See `examples/serve_xor` for a canonical walkthrough.
- **Tune** (`tune`): Parallel hyperparameter search with ASHA early stopping.
  Supports log-uniform floats, integers, and categorical choices. Runs trials
  in worker pools across all CPUs. See [`docs/TUNE_GUIDE.md`](docs/TUNE_GUIDE.md).

## Export Example

Train and save a model, then export it to standalone Go:

```bash
go run ./cmd/neugo export -model trained.json -out model_gen.go -pkg model
```

Use the generated function in any Go project:

```go
predictions := model.Predict([]float32{0.1, 0.2, ...})
```

The generated code has no external imports and compiles to any platform Go
supports (native binary, WASM, ARM64, etc.).

## Serve Example

Create a server with online learning:

```go
server, _ := serve.New(model, serve.Config{
    InputDim: 2, Loss: train.BCELoss(), Holdout: holdout,
})
server.StartOnline(ctx)
server.ListenAndServe(":8080")  // Hot-swap, Prometheus metrics, rollback
```

See `examples/serve_xor` for a full walkthrough with curl commands.

## Tune Example

Search for optimal hyperparameters with ASHA:

```go
space := tune.NewSpace().LogFloat("lr", 1e-4, 0.5).Int("hidden", 4, 64)
results, _ := tune.Run(ctx, space, func(trial *tune.Trial) (float64, error) {
    return trainModel(trial.Params.Float("lr"), trial.Params.Int("hidden")), nil
}, tune.Config{Trials: 60, Workers: 4, ASHA: &tune.ASHAConfig{...}})
```

See `examples/tune_wine` and [`docs/TUNE_GUIDE.md`](docs/TUNE_GUIDE.md) for details.

## Layout

    nn/       modules (dense, conv, attention, normalization), tensors, initializers, serialization
    train/    trainer, optimizers, losses, callbacks, schedulers, cross-validation, reporting
    data/     CSV/image loading, normalization, splitting, balancing, augmentation
    export/   model JSON -> dependency-free Go inference source
    serve/    HTTP serving: hot-swap, metrics, online learning, rollback
    tune/     search spaces, worker-pool random search, ASHA pruning
    cmd/neugo CLI (currently: export)
    examples/ runnable demos (see above)
    docs/     full guides

## Testing

    go build ./...
    go vet ./...
    go test ./...
    go test -race ./...              # concurrency-sensitive: nn's layers parallelize internally
    go test ./nn/ -bench . -benchmem # Conv2D/Linear/BatchNorm throughput

## Documentation

- [`docs/GUIDE.md`](docs/GUIDE.md) — full API walkthrough.
- [`docs/EXPORT_GUIDE.md`](docs/EXPORT_GUIDE.md) — exporting models to Go source.
- [`docs/TUNE_GUIDE.md`](docs/TUNE_GUIDE.md) — hyperparameter tuning and ASHA.

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
