# NeuGo Flax-Style Restructure — Design

Date: 2026-07-17
Status: Approved (design), pending implementation plan

## 1. Background

NeuGo is a zero-dependency Go neural network engine (~10.3k LOC) with packages
`Network/` (capital N, non-idiomatic), `data/`, `tensor/`, a root `main.go` demo,
and 21 example programs in a single `examples/` directory.

A codebase audit found the following problems, verified by inspection:

**Build breakage**

- `data/` does not compile: unused `hasHeader` (`data/image.go:36,106`),
  `float64 >= float32` comparison (`data/image.go:130`), undefined `NewRNG`
  (`data/image.go:201`), unused imports (`data/cifar10.go:4,5`).
- `examples/` holds 21 `package main` files with 21 `func main()` in one
  directory — not buildable as a unit; 8 of them import the broken `data`.
- Zero `*_test.go` files anywhere; no CI; no `.gitignore`.

**Functional bugs**

- Softmax is declared but never wired: `GetActivationFunction` has no
  `case Softmax` (`Network/activation.go:86-126`), so every "softmax" layer is
  silently a sigmoid layer; `ApplySoftmax` is dead code and the CCE+softmax
  gradient shortcuts (`Network/network.go:161`, `Network/batch.go:97`) are dead.
- The entire `Optimizer` tree (SGD/Adam/Momentum, RMSprop enum) is dead code:
  `Optimizer.Update` is never called; every training loop does raw-lr vanilla
  SGD. README advertises Adam/Momentum/RMSprop (`README.md:16`).
- `CNN.Train`/`CNN.Evaluate` are binary-only despite `QuickCNNMultiClass`
  existing: loss uses output neuron 0 vs. the whole one-hot vector
  (`Network/cnn.go:161-163`, `Network/fit.go:175`, `Network/cnn.go:200`).
- `QuickCNNMultiClass` mutates the caller's slice (`Network/cnn_builder.go:155`).
- `Train` appends a fresh `History` to the caller's shared `CallbackList` on
  every call (`Network/training.go:41`).
- `GELUFunc` computes a SiLU approximation, not GELU (`Network/nnx.go:279-286`).
- Hand-rolled `exp`/`sqrt` instead of `math.Exp`/`math.Sqrt` (`Network/nnx.go:315,388`,
  `data/cifar10.go:218`).
- Silent no-ops on input-size mismatch (`Network/network.go:87-89,136`).

**Structural problems**

- Six duplicated training loops: `Fit`, `FitWithValidation`, `FitWithScheduler`,
  `Train`, `CNN.Train`, `CNN.QuickFit`.
- Four competing construction APIs: low-level `NewLayer`/`NewNetwork`,
  `Sequential`, functional `Stack`/`L`, NNX modules (forward-only, untrainable).
- Duplicated forward pass (`ForwardPass` vs `forwardPassWithDropout`),
  duplicated gradients (`BackPropagation` vs `calculateGradients`),
  duplicated batch step (`TrainBatch` vs `TrainBatchWithRegularization`),
  four hardcoded weight-init blocks.
- Inconsistent error philosophy: panics in builders, silent returns in math,
  proper errors in serialization/data.
- Global RNG state (`rand.Seed` — deprecated) in `data/`; global `math/rand`
  in network/conv init. Builtin `println` in 7 library files.
- `Network` package name violates Go conventions and stutters
  (`Network.Network`).
- Repo hygiene: 8 trained-model JSONs + `test_predictions.csv` committed at
  root; 9 overlapping Markdown guides (~3.4k lines); stale `Network/doc.go`.

**Verified-missing features (selected for this project)**

- Working Adam/RMSprop wiring; dropout and BatchNorm as composable layers;
  gradient clipping; per-epoch shuffling; `Predict` for dense nets;
  initializer API; minibatch CNN training; multiclass CNN correctness;
  CNN serialization; model summary/parameter count.

**Out of scope (explicitly not selected)**

- Recurrent layers (RNN/LSTM/GRU), GPU/parallelism, data augmentation,
  one-hot CSV encoding, regression metrics (RMSE/R²), AUC/ROC,
  full autodiff engine.

## 2. Goals

1. Restructure into a Flax-NNX-feeling, idiomatic Go library: everything is a
   `Module`; MLPs and CNNs are the same kind of object; one training engine.
2. Fix every functional bug listed in §1 (softmax, data build, CNN multiclass,
   GELU, math functions, silent failures, slice mutation, History accumulation).
3. Add the selected missing features: wired Adam/RMSprop, `Dropout` and
   `BatchNorm` modules, gradient clipping, per-epoch shuffle, `Predict`,
   initializer API, minibatch CNN, CNN serialization, model summary.
4. Establish a test suite (unit + gradient checks + end-to-end convergence).
5. Clean repo hygiene: buildable examples, one guide, `.gitignore`, no
   committed artifacts.

## 3. Non-goals

- Autodiff: backprop stays manual per-module (JAX-style `grad` transforms are
  out of scope).
- Backward compatibility: the old `Network`/`CNN`/NNX APIs and the old JSON
  model format are deleted, not deprecated. All in-repo callers are rewritten.
- Performance work beyond what batching naturally gives (no SIMD, no
  goroutine parallelism, no GPU).
- Any feature listed as out-of-scope in §1.

## 4. Target architecture

### 4.1 Package layout

```
neugo/
├── nn/                      # package nn — model definition
│   ├── module.go            #   Module, Param, Context, Mode
│   ├── tensor.go            #   Tensor: n-dim float32, batch-first
│   ├── linear.go            #   Linear (dense)
│   ├── conv.go              #   Conv2D, MaxPool2D, AvgPool2D, Flatten
│   ├── norm.go              #   BatchNorm (trainable)
│   ├── dropout.go           #   Dropout
│   ├── activation.go        #   ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax
│   ├── sequential.go        #   Sequential composition + chain validation
│   ├── init.go              #   Initializers: Xavier, He, Zeros, Uniform, Normal
│   ├── rng.go               #   explicit RNG plumbing
│   ├── summary.go           #   model summary + parameter count
│   └── serialize.go         #   JSON save/load for any module tree
├── train/                   # package train — optimization
│   ├── trainer.go           #   Trainer: the single Fit loop; Predict
│   ├── optimizer.go         #   SGD, Momentum, Adam, RMSprop; ClipNorm wrapper
│   ├── loss.go              #   MSE, BCE, CrossEntropy (softmax+CCE shortcut), MAE
│   ├── scheduler.go         #   Step/Exponential/Cosine/Warmup/Plateau as callbacks
│   ├── callback.go          #   Callback iface, History, EarlyStopping,
│   │                        #   ModelCheckpoint, ProgressBar
│   ├── metrics.go           #   Evaluate → Metrics (acc/prec/rec/F1/confusion)
│   └── crossval.go          #   k-fold + stratified k-fold (with shuffle)
├── data/                    # fixed to compile; same feature set; no global RNG
├── examples/                # one subdirectory per example (buildable)
│   ├── xor/
│   ├── wine_quality/
│   ├── fashion_mnist/
│   ├── cifar10_cnn/
│   ├── callbacks/
│   └── crossval/
└── docs/
    ├── GUIDE.md             # single consolidated user guide
    └── superpowers/specs/   # design docs
```

Import direction: `train` imports `nn`; `nn` never imports `train`. `data` is
standalone (no dependency on `nn`/`train`).

### 4.2 Core abstractions (`nn`)

```go
// Tensor is the single data type flowing through the network.
// Batch-first: dense is [batch, features], conv is [batch, h, w, channels].
type Tensor struct {
    Data  []float32
    Shape []int
}

// Param is a trainable tensor pair an Optimizer can see and update.
type Param struct {
    Value *Tensor
    Grad  *Tensor
}

// Mode distinguishes training from inference (Dropout, BatchNorm).
type Mode int
const (
    Inference Mode = iota
    Train
)

// Context threads mode and RNG through forward/backward.
type Context struct {
    Mode Mode
    RNG  *rand.Rand // required in Train mode for Dropout; nil-safe otherwise
}

type Module interface {
    Forward(ctx *Context, x *Tensor) (*Tensor, error)
    Backward(ctx *Context, gradOut *Tensor) (*Tensor, error)
    Params() []*Param              // nil for stateless modules
    OutputShape(inShape []int) ([]int, error) // build-time validation
}
```

Composition:

```go
model := nn.Sequential(
    nn.Conv2D(1, 16, 3, nn.HeInit()), nn.ReLU(), nn.MaxPool2D(2),
    nn.Flatten(),
    nn.Linear(0, 128), nn.ReLU(), nn.Dropout(0.25),
    nn.Linear(0, 10), nn.Softmax(),
)
// inFeatures == 0 means "infer from the preceding module at build time".
```

`Sequential` validates the whole chain via `OutputShape` at construction and
returns an error naming the offending module index — no runtime shape
surprises, no silent no-ops.

- **Linear**: y = xW + b, batched; accumulates `Param.Grad` averaged over the
  batch. `Params()` returns [weight, bias].
- **Conv2D**: cross-correlation, stride 1, optional same-padding; batched 4D.
- **MaxPool2D/AvgPool2D**: with backward (argmax routing / gradient spreading).
- **Flatten**: [b, h, w, c] → [b, h*w*c]; backward reshapes.
- **Dropout(p)**: inverted dropout in `Train` mode using `ctx.RNG`; identity
  in `Inference`. Stateless (no params).
- **BatchNorm**: trainable gamma/beta (`Params()`), running mean/variance
  updated in `Train` mode, used in `Inference`. Over the feature/channel
  dimension for both 2D and 4D inputs.
- **Activations**: stateless modules; `Softmax` normalizes over the last axis.
  `GELU` uses the exact formula `0.5*x*(1+erf(x/sqrt(2)))` via `math.Erf`
  (replacing the current SiLU mislabel).

### 4.3 Training engine (`train`)

```go
type Optimizer interface {
    Step(params []*nn.Param) // reads .Grad, mutates .Value
}
```

`SGD(lr)`, `Momentum(lr, beta)`, `Adam(lr, b1, b2, eps)`, `RMSprop(lr, rho, eps)`
keep per-param state keyed by `*nn.Param` identity. `ClipNorm(opt, maxNorm)`
wraps any optimizer, rescaling gradients by global norm before `Step`.

```go
type Loss interface {
    // Returns scalar loss (batch-mean) and dLoss/dPred.
    Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error)
}
```

`CrossEntropy()` detects a trailing `Softmax` module (the trainer records the
model's last module) and uses the fused `pred - target` gradient; otherwise it
composes the generic CCE derivative with the softmax Jacobian. This makes the
shortcut real instead of dead code.

```go
trainer := train.New(model, train.Adam(1e-3), train.CrossEntropy())
hist, err := trainer.Fit(x, y,
    train.Epochs(30), train.BatchSize(32),
    train.Shuffle(true),               // per-epoch, seeded
    train.Validation(valX, valY),
    train.ClipGrad(1.0),               // optional gradient clipping
    train.Callbacks(
        train.EarlyStopping(5),
        train.ModelCheckpoint("model.json"),
        train.StepDecay(0.5, 10),
    ),
)
preds, err := trainer.Predict(xTest)   // inference mode, no dropout
```

Epoch data flow (the only training loop in the codebase):

1. Shuffle sample indices (seeded RNG).
2. Slice batch → `model.Forward(ctx{Train})`.
3. `loss.Loss` → value + input gradient.
4. `model.Backward` → param grads accumulated (batch-averaged).
5. `optimizer.Step(model.Params())`.
6. Batch callbacks → end of epoch: train/val metrics → scheduler →
   early-stopping (restores best weights via an in-memory param snapshot) →
   history.

`Evaluate(x, y)` runs the model in `Inference` mode and returns the existing
`Metrics` struct (loss, accuracy, precision, recall, F1, confusion matrix,
macro-averaged for multiclass). `nn.Summary(model)` prints per-module output
shapes and parameter counts plus a total.

### 4.4 Serialization

`nn.Save(model, path)` / `nn.Load(path)` — JSON document containing a module
tree: each node `{type, config, params}`, with a type registry mapping names
("linear", "conv2d", "relu", ...) to constructors. Works identically for dense
and conv models. Old `Version: "2.0"` format is not readable (hard break, noted
in README). RNG seed and optimizer state are NOT serialized (training-resume
is out of scope).

### 4.5 Data package

Fix the four compile errors (remove unused vars/imports, fix the float
comparison, replace the undefined `NewRNG` with an explicit `*rand.Rand`
parameter). Replace deprecated global `rand.Seed` by adding an explicit
`*rand.Rand` first argument to `ShuffleData`, `BalanceDataset`, and
`SplitData` (callers are all in-repo and get updated). No new data features.

### 4.6 Error handling — one philosophy

- Constructors and `Sequential` return `(T, error)` on invalid configuration.
- `Forward`/`Backward` return errors only for shape violations that slipped
  past build-time checks (should be impossible for validated chains).
- `Fit`/`Predict`/`Evaluate`/`Save`/`Load` return `error`.
- No `panic` in library code; no builtin `println`; no writes to stdout except
  from `ProgressBar`/`Summary`, which are explicitly user-facing output.
- Library code never mutates caller-owned slices/tensors unless documented
  (fixes the `QuickCNNMultiClass` class of bug by construction).

### 4.7 Testing strategy

New `*_test.go` files alongside each source file:

- **Gradient checks**: finite-difference verification of every `Backward`
  (Linear, Conv2D, MaxPool2D, AvgPool2D, Flatten, BatchNorm, activations,
  losses). Tolerance ~1e-3 in float32 with central differences.
- **Unit tests**: tensor ops, initializers (moment bounds), dropout rate
  statistics, BatchNorm normalization properties, softmax rows sum to 1,
  serialization round-trip equality, scheduler schedules, metrics correctness
  on hand-computed confusion matrices, crossval fold sizes/overlap.
- **End-to-end**: XOR converges with SGD and with Adam (loss < 0.05 within a
  fixed epoch budget, fixed seed); a tiny 2-class CNN converges on synthetic
  8x8 images; early stopping restores best weights; `Predict` matches
  `Forward` in inference mode.
- **data**: CSV load/save round-trip, normalization inverse property, split
  ratios, balancing counts.
- All tests seeded deterministically; `go test ./...` must be green, and
  `go vet ./...` clean.

### 4.8 Examples & docs

- Rewrite to ~6 examples, one directory each (list in §4.1), all using the new
  API; each builds via `go build ./...` and runs via `go run ./examples/xor`.
- Delete root `main.go`, all old `examples/*.go`, `Network/`, `tensor/`,
  and the root-level trained-model JSONs + `test_predictions.csv` (after the
  new examples regenerate equivalents where useful).
- Consolidate the 9 root guides into `README.md` (quickstart, feature list
  matching reality) + `docs/GUIDE.md` (full API walkthrough). Update or delete
  every claim that referenced dead/broken features.
- Add `.gitignore`: trained-model `*.json` at repo root and generated
  prediction CSVs (e.g. `test_predictions.csv`).

## 5. Implementation phases

Build must stay green at the end of every phase; the old `Network` package is
deleted only when its replacement is proven by tests.

1. **nn core**: Tensor, Module/Param/Context, initializers, RNG, Linear,
   activations (incl. real Softmax + exact GELU), Sequential with shape
   validation. Gradient-check tests green.
2. **train core**: losses (incl. fused softmax+CCE), SGD/Momentum/Adam/
   RMSprop, ClipNorm, Trainer (shuffle, batching, Predict), callbacks
   (History, EarlyStopping, ModelCheckpoint, ProgressBar), schedulers as
   callbacks. XOR/wine end-to-end tests green.
3. **Conv stack + regularization**: Conv2D/MaxPool/AvgPool/Flatten modules,
   Dropout, BatchNorm, metrics port, crossval port, model summary. Synthetic
   CNN convergence test green. `Network/` deleted at the end of this phase.
4. **Serialization + data + repo**: serialize.go, data package fixes + tests,
   examples rewrite, docs consolidation, `.gitignore`, artifact cleanup.
   Full `go build ./... && go vet ./... && go test ./...` green.

## 6. Risks and mitigations

- **Float32 gradient checks are noisy** → central differences, tolerance
  1e-3, small tensors, fixed seeds.
- **Batched conv backward is the most error-prone math** → implement against
  the existing (working) single-sample `conv2d.go` backward as reference,
  verify with gradient checks before deleting the old code.
- **Scope creep toward autodiff** → explicitly out of scope (§3); any
  "just make it a graph" idea is deferred.
- **Old-format model files** → hard break, called out in README; in-repo
  artifacts are regenerated by the new examples.
