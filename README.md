# NeuGo

A zero-dependency neural network library for Go. Everything is a
`Module` — dense layers, convolutions, pooling, dropout, batch norm, and
activations all compose the same way, and one `Trainer` handles fitting,
prediction, and evaluation for both.

## Install

    go get neugo

Requires Go 1.25+. No third-party dependencies — only the standard library.

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

Seven runnable examples live in `examples/`: `xor`, `wine_quality` (a real
dataset, `dataset/wine_quality/winequality-red.csv`), `fashion_mnist`
(convolutional; falls back to synthetic data unless you've downloaded a
Fashion-MNIST CSV yourself), `cifar10_cnn` and `cifar100_cnn`
(convolutional; each downloads and extracts the real dataset from
`cs.toronto.edu` on first run — ~170MB/~160MB — training on a capped subset
for speed, and falls back to synthetic data if the download fails, e.g. no
network), `callbacks` (early stopping, checkpointing, LR scheduling,
progress reporting), and `crossval` (k-fold cross-validation). Run any of
them with `go run ./examples/<name>`.

## Features

- **Modules** (`nn`): `Linear`, `Conv2D`/`Conv2DSame`, `MaxPool2D`,
  `AvgPool2D`, `Flatten`, `Dropout`, `BatchNorm`, activations (`ReLU`,
  `Sigmoid`, `Tanh`, `LeakyReLU`, `GELU`, `Softmax`) — all compose via
  `Sequential`, which validates the whole chain's shapes at construction.
- **Initializers**: Xavier, He, Zeros, Uniform, Normal — explicit
  `*rand.Rand` throughout, no global RNG state anywhere in the library.
- **Training** (`train`): one `Trainer.Fit` loop with per-epoch shuffling,
  batching, optional gradient clipping, and validation metrics.
  Optimizers: SGD, Momentum, Adam, RMSprop, plus a `ClipNorm` wrapper.
  Losses: MSE, MAE, BCE, CrossEntropy (with a fused softmax+cross-entropy
  gradient shortcut when the model ends in `Softmax`).
- **Callbacks**: `History` (always returned by `Fit`), `EarlyStopping`
  (with in-memory best-weight restore), `ModelCheckpoint`, `ProgressBar`,
  and five LR schedulers (`StepDecay`, `ExponentialDecay`,
  `CosineAnnealing`, `Warmup`, `ReduceLROnPlateau`).
- **Evaluation**: `Trainer.Evaluate` returns accuracy/precision/recall/F1/
  confusion-matrix `Metrics`, macro-averaged for multiclass;
  `train.KFoldSplits`/`StratifiedKFoldSplits`/`CrossValidate` for
  cross-validation.
- **Serialization**: `nn.Save`/`nn.Load` — one JSON format for any module
  tree, dense or convolutional.
- **`data`**: CSV loading, z-score/min-max normalization, train/val/test
  splitting, class balancing (oversample/undersample), MNIST-style and
  CIFAR-10 image loaders — all with explicit `*rand.Rand`, no global state.
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

## Testing

    go build ./...
    go vet ./...
    go test ./...

## Documentation

See [`docs/GUIDE.md`](docs/GUIDE.md) for a full API walkthrough.

## Design

See [`docs/superpowers/specs/2026-07-17-flax-restructure-design.md`](docs/superpowers/specs/2026-07-17-flax-restructure-design.md)
for the architecture this library follows and the bugs it fixed.

## Non-goals

Autodiff (backprop is manual per-module), backward compatibility with the
pre-restructure API and JSON format, GPU/SIMD/goroutine-parallel
execution, recurrent layers, data augmentation, and regression metrics
(RMSE/R²) are all explicitly out of scope — see the design doc §2–3 for
the full rationale.
