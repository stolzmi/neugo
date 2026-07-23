# NeuGo Guide

A full walkthrough of the `nn`, `train`, and `data` packages. For a
30-second version, see the [README](../README.md). Every snippet in this
guide compiles against the actual API — none of it is aspirational. For a
flat, auto-generated list of every layer constructor (regenerated from
source via `go generate ./...`, see `cmd/gendocs`), see
[`LAYERS.generated.md`](LAYERS.generated.md).

## Table of contents

1. [Building models](#1-building-models)
2. [Initializers](#2-initializers)
3. [Training](#3-training)
4. [Callbacks and schedulers](#4-callbacks-and-schedulers)
5. [Convolutional models](#5-convolutional-models)
6. [Evaluation and cross-validation](#6-evaluation-and-cross-validation)
7. [Serialization](#7-serialization)
8. [The `data` package](#8-the-data-package)
9. [Streaming training](#9-streaming-training)
10. [Recurrent layers and RoPE attention](#10-recurrent-layers-and-rope-attention)
11. [The `text` package](#11-the-text-package)
12. [Developer and research tooling](#12-developer-and-research-tooling)
13. [Migrating from the old `Network` API](#13-migrating-from-the-old-network-api)

---

## 1. Building models

### The `Module` interface

Every layer — dense, convolutional, pooling, dropout, batch norm,
activations — implements the same four-method interface
(`nn/module.go`):

```go
type Module interface {
    Forward(ctx *Context, x *Tensor) (*Tensor, error)
    Backward(ctx *Context, gradOut *Tensor) (*Tensor, error)
    Params() []*Param
    OutputShape(inShape []int) ([]int, error)
}
```

There is no separate "model" type distinct from a "layer" type — a
`*SequentialModel` (built by `Sequential`, see below) is itself a
`Module`, which is what makes CNNs and MLPs "the same kind of object."

### `Param`

A trainable tensor pair an `Optimizer` can see and update:

```go
type Param struct {
    Value *Tensor
    Grad  *Tensor
}
```

`Module.Params()` returns every `*Param` a layer owns (`nil` for
parameter-free layers like activations, pooling, and `Flatten`).
`Sequential`'s `Params()` concatenates its children's params, so
`trainer.Fit` and any `Optimizer` only ever see one flat `[]*Param` for
the whole model.

### `Context` and `Mode`

`Forward`/`Backward` thread a `*Context` through the call, carrying the
current mode and an RNG for anything stochastic:

```go
type Mode int

const (
    Inference Mode = iota
    Train
)

type Context struct {
    Mode Mode
    RNG  *rand.Rand
}
```

`Dropout` only drops units (and only needs `ctx.RNG`) when `ctx.Mode ==
Train`; `BatchNorm` computes and uses batch statistics in `Train` mode
and falls back to its running statistics in `Inference` mode.
`Trainer.Fit` builds this context for you (`Train` mode, a fresh RNG
seeded from `train.Seed`); `Trainer.Predict` and `Trainer.Evaluate` build
an `Inference`-mode context with no RNG. If you drive `Forward` yourself
in `Train` mode on a model containing `Dropout`, you must supply a
non-nil `ctx.RNG`.

### Layer constructors

| Constructor | Signature | Notes |
|---|---|---|
| `Linear` | `Linear(rng *rand.Rand, inFeatures, outFeatures int, init Initializer) *LinearLayer` | dense layer; see shape-inference rule below |
| `Conv2D` | `Conv2D(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer` | stride-1, no padding ("valid") |
| `Conv2DSame` | `Conv2DSame(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer` | stride-1, padding `(kernelSize-1)/2`; `kernelSize` should be odd for symmetric padding — this is the caller's responsibility, not a checked invariant |
| `MaxPool2D` | `MaxPool2D(poolSize, stride int) *MaxPool2DLayer` | no learnable params |
| `AvgPool2D` | `AvgPool2D(poolSize, stride int) *AvgPool2DLayer` | no learnable params |
| `AdaptiveAvgPool2D` | `AdaptiveAvgPool2D(outH, outW int) *AdaptiveAvgPool2DLayer` | pools to a fixed output size regardless of input spatial size (bins may overlap when it doesn't divide evenly, matching PyTorch) |
| `GlobalAvgPool2D` | `GlobalAvgPool2D() *AdaptiveAvgPool2DLayer` | sugar for `AdaptiveAvgPool2D(1, 1)` |
| `GlobalMaxPool2D` | `GlobalMaxPool2D() *GlobalMaxPool2DLayer` | max over the entire spatial extent, per channel |
| `Flatten` | `Flatten() *FlattenLayer` | `[batch, h, w, c]` → `[batch, h*w*c]` |
| `Dropout` | `Dropout(rate float32) *DropoutLayer` | inverted dropout; no-op outside `Train` mode |
| `BatchNorm` | `BatchNorm(channels int) *BatchNormLayer` | normalizes the last (channel) axis; works for both dense `[batch, features]` and conv `[batch, h, w, c]` inputs |
| `LayerNorm` | `LayerNorm(channels int) *LayerNormLayer` | normalizes each row's `channels` independently; no running stats |
| `GroupNorm` | `GroupNorm(groups, channels int) *GroupNormLayer` | normalizes per (sample, group) over channels+spatial |
| `InstanceNorm` | `InstanceNorm(channels int) *InstanceNormLayer` | `GroupNorm(channels, channels)` under its own name/serialization tag — per (sample, channel), spatial-only |
| `RMSNorm` | `RMSNorm(channels int) *RMSNormLayer` | scale-only, no mean-subtraction/bias — the LLaMA/Gemma-style LayerNorm replacement |
| `ReLU` | `ReLU() *ActivationModule` | |
| `Sigmoid` | `Sigmoid() *ActivationModule` | |
| `Tanh` | `Tanh() *ActivationModule` | |
| `LeakyReLU` | `LeakyReLU(alpha float32) *ActivationModule` | |
| `GELU` | `GELU() *ActivationModule` | exact erf formula, not a SiLU approximation |
| `ELU` | `ELU(alpha float32) *ActivationModule` | |
| `SELU` | `SELU() *ActivationModule` | fixed self-normalizing constants, no configurable alpha |
| `SiLU` | `SiLU() *ActivationModule` | aka Swish: `x*sigmoid(x)` |
| `Softplus` | `Softplus() *ActivationModule` | `log(1+exp(x))`, numerically stable for large \|x\| |
| `Mish` | `Mish() *ActivationModule` | `x*tanh(softplus(x))` |
| `Hardswish` | `Hardswish() *ActivationModule` | piecewise-linear SiLU approximation (MobileNetV3) |
| `PReLU` | `PReLU(channels int) *PReLULayer` | learnable per-channel negative-slope (unlike `LeakyReLU`'s fixed `alpha`) |
| `Softmax` | `Softmax() *SoftmaxModule` | expects `[batch, classes]`; see §3 for the fused-loss shortcut |
| `RNN` | `RNN(rng, features, hidden int, init Initializer) *RNNLayer` | vanilla recurrent layer; see §10 |
| `LSTM` | `LSTM(rng, features, hidden int, init Initializer) *LSTMLayer` | see §10 |
| `GRU` | `GRU(rng, features, hidden int, init Initializer) *GRULayer` | see §10 |
| `LastTimestep` | `LastTimestep() *LastTimestepLayer` | `[batch, seqLen, hidden]` → `[batch, hidden]`, composes after RNN/LSTM/GRU |
| `RotaryMultiHeadAttention` | `RotaryMultiHeadAttention(rng, dModel, numHeads int, causal bool, init Initializer) *RotaryMultiHeadAttentionLayer` | `MultiHeadAttention` with RoPE instead of an added positional embedding; see §10 |

If `init` is `nil`, `Linear` defaults to `XavierInit()` and `Conv2D`/
`Conv2DSame` default to `HeInit()`.

### The `inFeatures == 0` shape-inference rule

`Linear(rng, 0, outFeatures, init)` defers allocating its weight tensor
until the model is validated by `Sequential`, which calls each module's
`OutputShape` in order and passes the real preceding shape. This means
you never have to hand-compute the flattened size after a `Conv2D` /
`MaxPool2D` stack — write `0` and let `Flatten`'s actual output size fill
it in:

```go
rng := nn.NewRNG(1)
model, err := nn.Sequential([]int{32, 28, 28, 1}, // [batch, h, w, c]
    nn.Conv2DSame(rng, 1, 8, 3, nn.HeInit()),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Flatten(),
    nn.Linear(rng, 0, 64, nn.HeInit()), // 0 == infer from Flatten's output
    nn.ReLU(),
    nn.Linear(rng, 64, 10, nn.XavierInit()),
    nn.Softmax(),
)
```

Once built this way, `l.inFeatures` is fixed — a second call to
`OutputShape` with a different feature count returns an error rather
than silently rebuilding the layer.

### `Sequential`'s validation behavior

```go
func Sequential(inputShape []int, modules ...Module) (*SequentialModel, error)
```

`Sequential` walks `modules` in order, calling `OutputShape` on each with
the shape returned by the previous one (starting from `inputShape`).
Any shape mismatch — wrong rank, wrong channel/feature count, a
`Conv2D`'s kernel too large for its input — fails at construction time,
before you ever call `Forward`. The error names the offending module's
index in the chain:

```go
_, err := nn.Sequential([]int{4, 3},
    nn.Linear(rng, 5, 8, nn.HeInit()), // configured for 5 features, input has 3
    nn.ReLU(),
)
// err: "nn: Sequential module 0: nn: Linear configured for 5 input features, got 3"
```

The same wrapping (`"nn: Sequential module %d: %w"`) applies to errors
`Forward` and `Backward` return during actual training, so a shape bug
that only shows up at a later batch size is still traceable to a layer
index.

---

## 2. Initializers

All five initializers share one signature:

```go
type Initializer func(rng *rand.Rand, shape []int) *Tensor
```

They all read fan-in/fan-out from the weight shape itself — `[in, out]`
for `Linear`, `[outC, inC, kh, kw]` for `Conv2D`/`Conv2DSame` — so the
same `Initializer` value works for both dense and convolutional layers.

| Initializer | Use it for | Distribution |
|---|---|---|
| `XavierInit()` | layers followed by a symmetric/saturating activation (`Sigmoid`, `Tanh`, or the final `Softmax`/regression output) | uniform in `±sqrt(6/(fanIn+fanOut))` |
| `HeInit()` | layers followed by `ReLU`/`LeakyReLU`/`GELU` | normal with `std = sqrt(2/fanIn)` |
| `ZerosInit()` | reconstructing a layer's shape during `nn.Load` before its saved weights are copied in (see §7) | all zeros |
| `UniformInit(low, high float32)` | reproducing a specific paper's or framework's init scheme | uniform in `[low, high)` |
| `NormalInit(mean, std float32)` | same, when you want a specific Gaussian instead | normal with given mean/std |

`Linear`'s default (when `init` is `nil`) is `XavierInit()`; `Conv2D`/
`Conv2DSame`'s default is `HeInit()` — matching the ReLU-heavy
convolutional stacks they're normally used in.

```go
rng := nn.NewRNG(7)
hidden := nn.Linear(rng, 20, 64, nn.HeInit())   // feeds into ReLU below
out := nn.Linear(rng, 64, 3, nn.XavierInit())   // feeds into Softmax
model, err := nn.Sequential([]int{1, 20}, hidden, nn.ReLU(), out, nn.Softmax())
```

---

## 3. Training

### `Trainer`

```go
func New(model *nn.SequentialModel, opt Optimizer, loss Loss) *Trainer
```

`New` also inspects the model: if `loss` is a `*CrossEntropyLoss` and the
model's last module is `*nn.SoftmaxModule`, it flips on the fused
softmax+cross-entropy gradient shortcut (more on this below) — no extra
call needed.

### The `Fit` option list

```go
func (t *Trainer) Fit(x, y *nn.Tensor, opts ...FitOption) (*History, error)
```

`x`/`y` are the full training tensors — `Fit` handles per-epoch
shuffling and batching internally. Every option is a `FitOption`
constructed by a small function; unset options fall back to defaults
(`Epochs` is the only one with no usable default — omitting it is an
error).

| Option | Signature | Default | What it does |
|---|---|---|---|
| `Epochs` | `Epochs(n int) FitOption` | — (required, `n>0`) | number of passes over the data |
| `BatchSize` | `BatchSize(n int) FitOption` | full dataset (one batch/epoch) | mini-batch size |
| `Shuffle` | `Shuffle(enabled bool) FitOption` | `false` | reshuffle sample order every epoch |
| `Seed` | `Seed(seed int64) FitOption` | `42` | seeds the per-epoch shuffle RNG and the `Context.RNG` used by `Dropout` |
| `Validation` | `Validation(x, y *nn.Tensor) FitOption` | none | evaluated (in `Inference` mode) at the end of every epoch; populates `History.ValLoss`/`ValAcc`/`ValF1` and is what `EarlyStopping`/schedulers/`ModelCheckpoint` monitor when supplied |
| `ClipGrad` | `ClipGrad(maxNorm float32) FitOption` | off | wraps the optimizer in `ClipNorm(opt, maxNorm)` for this `Fit` call |
| `Callbacks` | `Callbacks(cbs ...Callback) FitOption` | none | see §4 |
| `WithSaveFunc` | `WithSaveFunc(fn func(*nn.SequentialModel, string) error) FitOption` | none | wires any `ModelCheckpoint` callback's save path to `fn(model, path)` — normally `nn.Save` |

Worked example, one option at a time (against the XOR toy problem from
the README):

```go
rng := nn.NewRNG(1)
model, _ := nn.Sequential([]int{4, 2},
    nn.Linear(rng, 2, 8, nn.HeInit()),
    nn.ReLU(),
    nn.Linear(rng, 8, 1, nn.XavierInit()),
    nn.Sigmoid(),
)
x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})
trainer := train.New(model, train.Adam(0.05, 0.9, 0.999, 1e-8), train.BCELoss())

// Epochs — required; how many passes over x/y.
trainer.Fit(x, y, train.Epochs(500))

// BatchSize — split each epoch into mini-batches of 2 rows.
trainer.Fit(x, y, train.Epochs(500), train.BatchSize(2))

// Shuffle — reorder the 4 rows before each epoch's batching.
trainer.Fit(x, y, train.Epochs(500), train.BatchSize(2), train.Shuffle(true))

// Seed — make that shuffle (and any Dropout) reproducible.
trainer.Fit(x, y, train.Epochs(500), train.Shuffle(true), train.Seed(123))

// Validation — evaluate on held-out data every epoch (here, x/y itself).
trainer.Fit(x, y, train.Epochs(500), train.Validation(x, y))

// ClipGrad — rescale gradients if their global L2 norm exceeds 1.0.
trainer.Fit(x, y, train.Epochs(500), train.ClipGrad(1.0))

// Callbacks — see §4 for EarlyStopping/ModelCheckpoint/schedulers/ProgressBar.
trainer.Fit(x, y, train.Epochs(500), train.Callbacks(train.EarlyStopping(20)))

// WithSaveFunc — required alongside a ModelCheckpoint callback so it can
// actually write a file.
checkpoint := train.ModelCheckpoint("best.json", "loss", "min", true)
trainer.Fit(x, y, train.Epochs(500),
    train.Validation(x, y),
    train.Callbacks(checkpoint),
    train.WithSaveFunc(nn.Save),
)
```

`Fit` returns a `*History` (always fresh — never accumulated across
calls) with `TrainLoss`, `ValLoss`, `ValAcc`, `ValF1` slices (one entry
per epoch) plus `StartTime`/`EndTime`/`Duration()`.

`Trainer.Predict(x *nn.Tensor) (*nn.Tensor, error)` runs the model in
`Inference` mode. `Trainer.Evaluate(x, y *nn.Tensor) (Metrics, error)`
does the same and additionally computes loss and the full `Metrics` set
(§6).

### Fused softmax + cross-entropy

`CrossEntropyLoss.Loss` normally applies softmax internally before
computing the gradient. If the model's last module is already
`nn.Softmax()`, running softmax twice would be wrong (and wasteful) — so
`train.New` detects that case and sets the loss's internal `fused` flag,
which changes two things: `CrossEntropyLoss.Loss` treats its input as
already-normalized probabilities, and `Trainer.Fit` skips calling
`Backward` on the final `Softmax` module, feeding `(probs - target) /
batch` straight into the second-to-last module instead (the standard
combined-gradient shortcut).

You can check whether it's active:

```go
softmaxRNG := nn.NewRNG(1)
softmaxModel, _ := nn.Sequential([]int{1, 4},
    nn.Linear(softmaxRNG, 4, 3, nn.HeInit()),
    nn.ReLU(),
    nn.Linear(softmaxRNG, 3, 3, nn.XavierInit()),
    nn.Softmax(), // last module — this is what triggers fusion
)
ce := train.CrossEntropy()
trainer := train.New(softmaxModel, train.Adam(1e-3, 0.9, 0.999, 1e-8), ce)
fmt.Println(ce.Fused()) // true
```

If you instead end the model in raw logits (no `Softmax` module) and use
`train.CrossEntropy()`, `Fused()` stays `false` and the loss applies
softmax internally on every call — both are correct, the fused path is
just the small efficiency/numerical-stability shortcut Flax-style APIs
also take when the model already ends in `Softmax`.

### Optimizers

```go
train.SGD(lr float32) *SGDOptimizer
train.Momentum(lr, beta float32) *MomentumOptimizer
train.Adam(lr, beta1, beta2, eps float32) *AdamOptimizer
train.AdamW(lr, beta1, beta2, eps, weightDecay float32) *AdamOptimizer
train.RMSprop(lr, rho, eps float32) *RMSpropOptimizer
train.Adagrad(lr, eps float32) *AdagradOptimizer
train.Adadelta(lr, rho, eps float32) *AdadeltaOptimizer
train.Nadam(lr, beta1, beta2, eps float32) *NadamOptimizer
train.Lion(lr, beta1, beta2 float32) *LionOptimizer
train.ClipNorm(inner Optimizer, maxNorm float32) *ClipNormOptimizer
train.L1Reg(inner Optimizer, lambda float32) *L1RegOptimizer
train.L2Reg(inner Optimizer, lambda float32) *L2RegOptimizer
```

Every optimizer implements `Optimizer` (`Step([]*nn.Param)`,
`SetLR(float32)`, `GetLR() float32`); `SetLR`/`GetLR` are what the
schedulers in §4 call every epoch. `ClipNorm`/`L1Reg`/`L2Reg` all wrap
another `Optimizer` and are composable with each other (e.g.
`L2Reg(ClipNorm(Adam(...), 1.0), 1e-4)`) — `Trainer.Fit`'s `ClipGrad`
option builds a `ClipNorm` wrapper for you, but `L1Reg`/`L2Reg` (classical
gradient-based regularization, distinct from `AdamW`'s decoupled
`WeightDecay`) are always constructed directly around whichever optimizer
you pass to `train.New`. `Nadam` needs its own step counter like `Adam`;
`Lion` steps by the *sign* of an interpolated momentum term, so every
parameter moves by the same size step each update, scaled only by `LR`.

Every optimizer's `Step` splits each parameter's per-element update across
goroutines once that parameter is large enough (~4096 elements) for the
dispatch overhead to pay off — small parameters (biases, norm scales)
always run inline. This is transparent: the math is identical either way,
and `nn.SetDeterministic(true)` forces the inline, single-threaded path
everywhere, same as it does for `nn`'s own layers.

### Losses

```go
train.MSELoss() *MeanSquaredError                    // regression
train.MAELoss() *MeanAbsoluteError                   // regression, robust to outliers
train.HuberLoss(delta float32) *Huber                // regression, quadratic near 0 / linear beyond delta
train.SmoothL1Loss() *Huber                          // HuberLoss(1.0)
train.BCELoss() *BinaryCrossEntropy                  // single-output binary classification, target in {0,1}
train.CrossEntropy() *CrossEntropyLoss               // multiclass, target one-hot; see fused note above
train.KLDivergenceLoss() *KLDivergence               // KL(target || pred), both as [batch, classes] distributions
train.HingeLoss() *Hinge                             // margin classification, target in {-1, +1}
train.FocalLoss(gamma, alpha float32) *Focal         // BCE variant that down-weights easy examples
train.CosineSimilarityLoss() *CosineSimilarity       // 1-cos_sim(pred_row, target_row), e.g. for embeddings
train.LabelSmoothingLoss(inner Loss, epsilon float32, numClasses int) *LabelSmoothing // wraps any Loss
```

Every `Loss` implements `Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error)`,
returning the batch-mean scalar loss and `dLoss/dPred`. `LabelSmoothing`
is a decorator (same shape as the optimizer wrappers above): it smooths
`target` toward `epsilon/numClasses` before delegating to `inner`, so
`train.LabelSmoothingLoss(train.CrossEntropy(), 0.1, 10)` behaves like
`CrossEntropy` but discourages overconfident predictions. If you want a
label-smoothed loss to also use the fused softmax shortcut, call
`ce.SetFused(true)` on the inner `*CrossEntropyLoss` yourself before
wrapping it — `train.New` only auto-detects fusion on a `*CrossEntropyLoss`
passed to it directly.

---

## 4. Callbacks and schedulers

A `Callback` observes training events via five hooks
(`OnTrainBegin`/`OnTrainEnd`/`OnEpochBegin`/`OnEpochEnd`/`OnBatchEnd`);
embed `train.BaseCallback` to get no-op defaults and only override what
you need. `train.Callbacks(cbs...)` fans every hook out to each
registered callback in the order given.

Built-in callbacks and schedulers:

- `train.EarlyStopping(patience int) *EarlyStoppingCallback` — stops
  once `patience` epochs pass without the monitored loss (validation
  loss if `Validation` was supplied, else train loss) improving by at
  least `MinDelta`; snapshots the best in-memory weights and restores
  them into the model when it stops.
- `train.ModelCheckpoint(filepath, monitor, mode string, saveBestOnly bool) *ModelCheckpointCallback` —
  `monitor` is `"loss"`, `"accuracy"`, or `"f1"`; `mode` is `"min"` or
  `"max"`; requires both a `Validation` tensor pair (it reads
  `valMetrics`) and `train.WithSaveFunc(...)` to actually write a file.
  Save failures land in `.LastError` rather than aborting `Fit`.
- `train.ProgressBar(totalEpochs, printEvery int) *ProgressBarCallback` —
  prints one line every `printEvery` epochs (and always the last one).
- Five per-epoch LR schedulers, each wrapping an `Optimizer` and
  adjusting its LR from `OnEpochBegin`/`OnEpochEnd`:
  `train.StepDecay(opt, decayRate float32, decaySteps int)`,
  `train.ExponentialDecay(opt, decayRate float32)`,
  `train.CosineAnnealing(opt, minLR float32, maxEpochs int)`,
  `train.Warmup(opt, targetLR float32, warmupSteps int)`, and
  `train.ReduceLROnPlateau(opt, factor float32, patience int, minLR float32, mode string)`.
- Two per-*batch* LR schedulers, adjusting LR from `OnBatchEnd` instead
  (since `Fit` has no `OnBatchBegin`/global-step hook, both keep their own
  internal step counter incremented once per call rather than trusting
  the `batch` argument, which resets every epoch):
  `train.OneCycleLR(opt, maxLR float32, totalSteps int)` — Smith's
  "1cycle" policy, cosine ramp from `maxLR/25` up to `maxLR` over the
  first 30% of `totalSteps`, then cosine back down to `maxLR/25/1e4`;
  construct with `totalSteps = epochs * batchesPerEpoch` — and
  `train.CyclicLR(opt, baseLR, maxLR float32, stepSizeUp, stepSizeDown int)` —
  a repeating triangular ramp between `baseLR` and `maxLR`. Both set the
  optimizer's LR immediately on construction (to the schedule's step-0
  value), then advance on every `OnBatchEnd` call.

Combined example (this mirrors `examples/callbacks/main.go`):

```go
rng := nn.NewRNG(1)
model, _ := nn.Sequential([]int{4, 2},
    nn.Linear(rng, 2, 8, nn.HeInit()),
    nn.ReLU(),
    nn.Linear(rng, 8, 1, nn.XavierInit()),
    nn.Sigmoid(),
)
x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

opt := train.Adam(0.05, 0.9, 0.999, 1e-8)
scheduler := train.StepDecay(opt, 0.5, 200)          // halve LR every 200 epochs
earlyStop := train.EarlyStopping(50)                  // stop after 50 stale epochs
checkpoint := train.ModelCheckpoint("best_model.json", "loss", "min", true)
progress := train.ProgressBar(1000, 100)              // print every 100th epoch

trainer := train.New(model, opt, train.BCELoss())
hist, err := trainer.Fit(x, y,
    train.Epochs(1000), train.BatchSize(4), train.Seed(1),
    train.Validation(x, y),
    train.Callbacks(scheduler, earlyStop, checkpoint, progress),
    train.WithSaveFunc(nn.Save),
)
if err != nil {
    panic(err)
}
fmt.Printf("stopped after %d epochs\n", len(hist.TrainLoss))
if checkpoint.LastError != nil {
    fmt.Println("checkpoint save error:", checkpoint.LastError)
}
```

Note the ordering matters only in the sense that all four callbacks see
every hook every epoch, in the order passed to `Callbacks`; the
scheduler adjusts `opt`'s LR in `OnEpochBegin`, before that epoch's
batches run, while `earlyStop`/`checkpoint` react in `OnEpochEnd`, after
that epoch's validation metrics are computed.

---

## 5. Convolutional models

Every tensor in this codebase is batch-first; the convolutional
convention (`nn/conv.go`, `nn/pooling.go`) is **channels-last**:
`[batch, height, width, channels]`. This is also the layout `Conv2D`,
`Conv2DSame`, `MaxPool2D`, `AvgPool2D`, `BatchNorm`, and the `data`
package's `Image` type all agree on — no transposes needed between
loading data and feeding a model.

`Conv2D` vs. `Conv2DSame`:

- `Conv2D(rng, inChannels, outChannels, kernelSize, init)` — stride 1,
  zero padding ("valid"): output spatial size shrinks by `kernelSize-1`.
- `Conv2DSame(rng, inChannels, outChannels, kernelSize, init)` — stride
  1, padding `(kernelSize-1)/2` ("same"): output spatial size equals
  input spatial size when `kernelSize` is odd (the caller's
  responsibility — an even `kernelSize` isn't rejected, it just produces
  asymmetric padding).

A full CNN example (mirrors `examples/fashion_mnist/main.go`, using
`data.LoadMNISTFromCSV` — see §8 — with a synthetic fallback):

```go
func imagesToTensor(images []*data.Image) *nn.Tensor {
    h, w, c := images[0].Height, images[0].Width, images[0].Channels
    t := nn.NewTensor([]int{len(images), h, w, c})
    for i, img := range images {
        for hh := 0; hh < h; hh++ {
            for ww := 0; ww < w; ww++ {
                base := ((i*h+hh)*w + ww) * c
                copy(t.Data[base:base+c], img.Data[hh][ww])
            }
        }
    }
    return t
}

func labelsToTensor(labels [][]float32) *nn.Tensor {
    cols := len(labels[0])
    flat := make([]float32, len(labels)*cols)
    for i, row := range labels {
        copy(flat[i*cols:(i+1)*cols], row)
    }
    t, _ := nn.NewTensorFromData(flat, []int{len(labels), cols})
    return t
}

dataset, err := data.LoadMNISTFromCSV("dataset/fashion_mnist/fashion-mnist_train.csv")
if err != nil {
    panic(err)
}

rng := nn.NewRNG(1)
model, err := nn.Sequential([]int{len(dataset.Images), dataset.Height, dataset.Width, dataset.Channels},
    nn.Conv2DSame(rng, dataset.Channels, 8, 3, nn.HeInit()),
    nn.ReLU(),
    nn.BatchNorm(8),
    nn.MaxPool2D(2, 2),
    nn.Conv2DSame(rng, 8, 16, 3, nn.HeInit()),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Flatten(),
    nn.Linear(rng, 0, 64, nn.HeInit()), // inferred from Flatten's output
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(rng, 64, 10, nn.XavierInit()),
    nn.Softmax(),
)
if err != nil {
    panic(err)
}

x := imagesToTensor(dataset.Images)
y := labelsToTensor(dataset.Labels)

trainer := train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.CrossEntropy())
if _, err := trainer.Fit(x, y, train.Epochs(20), train.BatchSize(16), train.Shuffle(true), train.Seed(2)); err != nil {
    panic(err)
}
metrics, err := trainer.Evaluate(x, y)
if err != nil {
    panic(err)
}
fmt.Printf("train-set accuracy: %.2f%%\n", metrics.Accuracy)
```

`BatchNorm(8)` here normalizes over the 8 output channels of the first
`Conv2DSame` — the same `BatchNormLayer` type works identically after a
`Linear` layer in a dense model, since it always normalizes the last
(fastest-varying) axis of its input tensor.

`examples/cifar10_cnn` and `examples/cifar100_cnn` follow the same shape
but with `data.LoadCIFAR10Binary`/`data.LoadCIFAR100Binary` instead of
`LoadMNISTFromCSV`, and one more layer of "real data" convenience: on
first run, if the expected local file (`dataset/cifar10/data_batch_1.bin`
or `dataset/cifar100/train.bin`) is missing, each example downloads and
extracts the official binary distribution from `cs.toronto.edu` (~170MB /
~160MB) before training, capping the number of images it actually trains
on so the demo finishes quickly on this library's pure-Go, non-SIMD conv
loops. If the download itself fails (no network, host unreachable), both
fall back to the same kind of synthetic data `fashion_mnist` uses, rather
than erroring out. `data.LoadCIFAR100ClassNames(fineNamesPath,
coarseNamesPath)` reads the human-readable label names
(`fine_label_names.txt`/`coarse_label_names.txt`) that ship inside the
CIFAR-100 archive, rather than hardcoding a 100-entry name table in
source — one line per label, in label-index order.

---

## 6. Evaluation and cross-validation

### `Metrics`

```go
type Metrics struct {
    Loss            float32
    Accuracy        float32
    Precision       float32
    Recall          float32
    F1Score         float32
    ConfusionMatrix [][]int
    ROCAUC          float32
    PRAUC           float32
    Top5Accuracy    float32
    Perplexity      float32
}
```

`Trainer.Evaluate` computes this from a model's predictions: for
single-output `[batch, 1]` predictions it's binary (threshold 0.5,
2x2 confusion matrix `[[TN, FP], [FN, TP]]`); for `[batch, classes]`
predictions with `classes > 1` it's multiclass, using `argmax` per row
and macro-averaging precision/recall (and, from those, F1) across
classes. `Accuracy` is always a percentage (0–100).

`ROCAUC`/`PRAUC` are macro-averaged one-vs-rest for multiclass (each
class scored as its own binary problem via a rank-sum AUC / step-function
average-precision estimator, averaged over classes that have at least one
positive and one negative example — classes with degenerate support are
skipped, same convention as `Precision`/`Recall`). `Top5Accuracy` is
top-`min(5, classes)` accuracy — for the binary (`classes == 1`) case
there's no meaningful top-5 to take, so it's just set equal to `Accuracy`
there. `Perplexity` is `exp(Loss)` — only a meaningful quantity when the
model was trained with a natural-log cross-entropy loss, but it's
populated regardless of which `Loss` was used.

### Cross-validation

```go
func KFoldSplits(rng *rand.Rand, x, y [][]float32, k int, shuffle bool) []Fold
func StratifiedKFoldSplits(rng *rand.Rand, x, y [][]float32, k int) []Fold
func CrossValidate(folds []Fold, trainFold func(fold Fold) (Metrics, error)) (CrossValResult, error)
```

`Fold` holds `TrainX`/`TrainY`/`TestX`/`TestY` as `[][]float32`.
`StratifiedKFoldSplits` assumes binary labels (`y[i][0] > 0.5` == class
1) and keeps the class ratio consistent across folds — use it over
plain `KFoldSplits` whenever your binary classes are imbalanced.
`CrossValidate` calls `trainFold` once per fold — you're responsible for
building a fresh model and `Trainer` inside `trainFold`, since the whole
point of a fold is an independent training run — and returns a
`CrossValResult` with per-fold `Metrics` plus mean/std of accuracy, F1,
and loss across folds, and the index of the best/worst fold by accuracy.

Folds run concurrently across a worker pool bounded by `GOMAXPROCS`
(mirroring `tune.Run`'s pattern), so `trainFold` must be safe to call from
multiple goroutines at once — in practice this just falls out of building
a fresh model/optimizer per call, which the example below already does.
With `nn.SetDeterministic(true)` in effect, folds instead run
sequentially in fold order.

Full example (mirrors `examples/crossval/main.go`):

```go
func toTensor(rows [][]float32) *nn.Tensor {
    cols := len(rows[0])
    flat := make([]float32, len(rows)*cols)
    for i, row := range rows {
        copy(flat[i*cols:(i+1)*cols], row)
    }
    t, _ := nn.NewTensorFromData(flat, []int{len(rows), cols})
    return t
}

dataRNG := nn.NewRNG(1)
folds := train.KFoldSplits(dataRNG, x, y, 5, true) // x, y are [][]float32

result, err := train.CrossValidate(folds, func(fold train.Fold) (train.Metrics, error) {
    modelRNG := nn.NewRNG(2)
    model, err := nn.Sequential([]int{1, 2},
        nn.Linear(modelRNG, 2, 8, nn.HeInit()),
        nn.ReLU(),
        nn.Linear(modelRNG, 8, 1, nn.XavierInit()),
        nn.Sigmoid(),
    )
    if err != nil {
        return train.Metrics{}, err
    }
    trainer := train.New(model, train.Adam(0.05, 0.9, 0.999, 1e-8), train.BCELoss())
    if _, err := trainer.Fit(toTensor(fold.TrainX), toTensor(fold.TrainY), train.Epochs(200), train.Seed(3)); err != nil {
        return train.Metrics{}, err
    }
    return trainer.Evaluate(toTensor(fold.TestX), toTensor(fold.TestY))
})
if err != nil {
    panic(err)
}
fmt.Printf("mean accuracy: %.2f%% (± %.2f)  mean F1: %.4f\n", result.MeanAccuracy, result.StdAccuracy, result.MeanF1)
fmt.Printf("best fold: %d  worst fold: %d\n", result.BestFold, result.WorstFold)
```

---

## 7. Serialization

```go
func Save(model *SequentialModel, path string) error
func Load(path string) (*SequentialModel, error)
```

`Save` writes the model as an indented JSON document; `Load` reads it
back and reconstructs the module tree with trained weights copied in.

The JSON is a tree of nodes shaped like the `moduleDoc` type in
`nn/serialize.go`:

```go
type moduleDoc struct {
    Type    string              `json:"type"`
    Config  json.RawMessage     `json:"config,omitempty"`
    Params  map[string]paramDoc `json:"params,omitempty"`
    Modules []moduleDoc         `json:"modules,omitempty"`
}
```

A `"sequential"` root node has a `Modules` array of child nodes, one per
layer, in order; each child has a `Type` string identifying the layer
(`"linear"`, `"conv2d"`, `"maxpool2d"`, `"avgpool2d"`,
`"adaptive_avgpool2d"`, `"global_maxpool2d"`, `"flatten"`, `"dropout"`,
`"batchnorm"`, `"layernorm"`, `"groupnorm"`, `"instancenorm"`,
`"rmsnorm"`, `"softmax"`, `"relu"`, `"sigmoid"`, `"tanh"`, `"gelu"`,
`"elu"`, `"selu"`, `"silu"`, `"softplus"`, `"mish"`, `"hardswish"`,
`"leaky_relu"`, `"prelu"`, `"rnn"`, `"lstm"`, `"gru"`, `"last_timestep"`,
`"multihead_attention"`, `"rotary_multihead_attention"`,
`"positional_embedding"`, `"embedding"`, `"conv1d"`,
`"convtranspose2d"`, `"frozen"`, `"residual"`, or `"sequential"` for a
nested sub-model), a type-specific `Config` (e.g. `in_features`/
`out_features` for `"linear"`; `channels`/`running_mean`/`running_var`
for `"batchnorm"`), and, for layers with learnable weights, a `Params`
map (`"W"`/`"B"` for `Linear`/`Conv2D`, `"gamma"`/`"beta"` for
`BatchNorm`, `"Wx"`/`"Wh"`/`"B"` for `RNN`/`LSTM`, `"Wx"`/`"Wh"`/`"Bx"`/
`"Bh"` for `GRU`) of `{shape, data}` tensors. Every stateful new layer
type follows the same rule — a config struct plus one `case` in each
direction of `nn/serialize.go`'s type switch — so see that file directly
for the exact per-type config structs if you need to read or generate
this JSON from outside Go.

```go
if err := nn.Save(model, "model.json"); err != nil {
    panic(err)
}
loaded, err := nn.Load("model.json")
if err != nil {
    panic(err)
}
pred, err := loaded.Forward(&nn.Context{Mode: nn.Inference}, x)
```

**Explicit non-goals**: the saved JSON contains only architecture and
weights. It does **not** include the training RNG seed, optimizer state
(Adam's moment estimates, etc.), or anything else needed to exactly
resume an interrupted training run — loading a saved model and calling
`Fit` again starts a brand-new optimizer from scratch on top of the
loaded weights, not a resumed one. This is intentional (see the design
doc's non-goals) rather than a missing feature.

### `SaveWithMetadata`/`LoadWithMetadata` and reproducibility

```go
func SaveWithMetadata(model *SequentialModel, path string, meta Metadata) error
func LoadWithMetadata(path string) (*SequentialModel, Metadata, error)

type Metadata struct {
    InputShape    []int
    ClassNames    []string
    Normalization *NormalizationStats
    Manifest      map[string]string
}
```

A distinct, opt-in file format from plain `Save`/`Load` (not
cross-compatible — a file written by one must be read by its matching
counterpart) that bundles inference-relevant bookkeeping alongside the
weights. `Manifest` is free-form reproducibility bookkeeping — `nn`
itself never populates it (no `git`/`exec` dependency to keep this
library free of), it's just a place to put whatever answers "what
exactly produced this checkpoint" later:

```go
meta := nn.Metadata{
    ClassNames: []string{"cat", "dog"},
    Manifest: map[string]string{
        "go_version": runtime.Version(),
        "git_commit": commitHash, // however you obtain it in your own build
        "seed":       fmt.Sprintf("%d", seed),
        "lr":         fmt.Sprintf("%g", lr),
    },
}
nn.SaveWithMetadata(model, "checkpoint.json", meta)
```

---

## 8. The `data` package

`data` never imports `nn` (so it can be used, tested, and vendored
independently of the model/training packages) and — like `nn` and
`train` — never touches global RNG state: every function that needs
randomness takes an explicit `*rand.Rand` parameter. Its own image type,
`data.Image` (`[height][width][channel]`), still matches `nn.Tensor`'s
channels-last convention, so batching a `[]*data.Image` into an
`*nn.Tensor` (see the `imagesToTensor` helper in §5) is a straight
copy, no transposes.

**CSV loading**

```go
func LoadCSV(filepath string, config CSVConfig) (*Dataset, error)
```

```go
dataset, err := data.LoadCSV("dataset/wine_quality/winequality-red.csv", data.CSVConfig{
    Delimiter:       ';',
    HasHeader:       true,
    LabelColumn:     -1,  // -1 == last column
    LabelType:       "binary",
    BinaryThreshold: 6.0,
})
```

`Dataset.Features`/`.Labels` are `[][]float32`; `.NumFeatures`/
`.NumSamples`/`.FeatureNames` describe the loaded shape.

**Preprocessing**

```go
func CalculateStats(data [][]float32) Stats
func NormalizeZScore(data [][]float32, stats Stats) [][]float32
func NormalizeMinMax(data [][]float32, stats Stats) [][]float32
func SplitData(rng *rand.Rand, features, labels [][]float32, config SplitConfig) Split
```

```go
stats := data.CalculateStats(dataset.Features)
normalized := data.NormalizeZScore(dataset.Features, stats) // or NormalizeMinMax

rng := nn.NewRNG(2)
split := data.SplitData(rng, normalized, dataset.Labels, data.SplitConfig{
    TrainRatio: 0.7, ValRatio: 0.15, TestRatio: 0.15, Shuffle: true,
})
// split.TrainX/TrainY, split.ValX/ValY, split.TestX/TestY
```

**Class balancing**

```go
func AnalyzeClassDistribution(labels [][]float32, threshold float32) ClassDistribution
func BalanceDataset(rng *rand.Rand, features, labels [][]float32, targetRatio float64, preferOversample bool) ([][]float32, [][]float32)
```

```go
dist := data.AnalyzeClassDistribution(dataset.Labels, 0.5)
if !dist.IsBalanced {
    features, labels := data.BalanceDataset(rng, dataset.Features, dataset.Labels, 0.4, true)
    _ = features
    _ = labels
}
```

`BalanceDataset` picks oversampling (duplicate the minority class,
`preferOversample=true`) or undersampling (drop majority-class rows,
`preferOversample=false`) and is a thin wrapper over the lower-level
`OversampleMinorityClass`/`UndersampleMajorityClass` if you need more
control over the strategy (`"duplicate"`/`"random"` and
`"random"`/`"systematic"` respectively).

**Image loaders**

```go
func LoadMNISTFromCSV(filepath string) (*ImageDataset, error)
func LoadCIFAR10Binary(filepath string) (*CIFAR10Dataset, error)
func LoadCIFAR100Binary(filepath string) (*CIFAR100Dataset, error)
func LoadCIFAR100BinaryBatch(filepaths []string) (*CIFAR100Dataset, error)
func LoadCIFAR100ClassNames(fineNamesPath, coarseNamesPath string) (fine, coarse []string, err error)
```

`LoadMNISTFromCSV` reads the common MNIST/Fashion-MNIST CSV layout
(one row per image: label, then 784 pixel values) into 28×28×1
`*data.Image`s with one-hot 10-class labels, normalizing pixels to
`[0, 1]`. `LoadCIFAR10Binary` reads the CIFAR-10 binary batch format
(3073 bytes/record: 1 label byte + 3072 pixel bytes) into 32×32×3
`*data.Image`s, also one-hot labeled, with `.ClassNames` populated
(`"airplane"`, `"automobile"`, …). `LoadCIFAR100Binary` reads the
CIFAR-100 binary format (3074 bytes/record: 1 coarse-label byte + 1
fine-label byte + 3072 pixel bytes, one byte longer per record than
CIFAR-10's format) into the same 32×32×3 `*data.Image`s, with both
`FineLabels` (100-way one-hot) and `CoarseLabels` (20-way one-hot,
CIFAR-100's built-in superclass grouping) populated — `ClassNames` and
`CoarseClassNames` on the returned `*CIFAR100Dataset` are left empty
until you call `LoadCIFAR100ClassNames`, since the human-readable names
live in separate text files (`fine_label_names.txt`,
`coarse_label_names.txt`) that ship alongside the binary data, not in
the binary files themselves. All three image loaders are RNG-free — but
downstream splitting (`SplitImageData`) follows the same explicit-
`*rand.Rand`/no-global-state convention as everything else in this
package. `NormalizeImages` is deterministic (it computes per-channel
mean/std and applies them) and takes no RNG at all — there's nothing
random about it.

**Folder loader**

```go
func LoadImageFolder(root string) (*ImageDataset, []string, error)
```

Walks `root`, treating each immediate subdirectory as one class (the
standard `root/classA/*.jpg`, `root/classB/*.png`, ... layout), decoding
via the standard library's `image/jpeg`/`image/png` (still
dependency-free) into 3-channel RGB `*data.Image`s normalized to `[0,1]`.
Every image must decode to the same height/width as the first one loaded.
Returned alongside the dataset: `classNames`, the subdirectories' sorted
name order — also each class's one-hot index in `ImageDataset.Labels`.

The directory walk is sequential, but the `image.Decode` calls — the
expensive part for any real dataset — run across a worker pool bounded by
`GOMAXPROCS`; the returned dataset's order and every error message are
identical to a fully sequential load regardless.

**Augmentations**

Beyond `FlipHorizontal`/`AugmentWithFlips`:

```go
func RandomRotate(rng *rand.Rand, img *Image, maxDegrees float64) *Image
func RandomCrop(rng *rand.Rand, img *Image, cropH, cropW, padding int) *Image
func ColorJitter(rng *rand.Rand, img *Image, brightness, contrast, saturation float32) *Image
func Cutout(rng *rand.Rand, img *Image, size int) *Image
```

`RandomRotate` bilinearly resamples around the image center, filling
rotated-in pixels with 0. `RandomCrop` zero-pads by `padding` pixels on
every side then takes a random `cropH x cropW` window (the standard
CIFAR-style "pad and crop"). `ColorJitter` perturbs brightness/contrast/
(3-channel-only) saturation, each by an independent random factor — pass
0 for any adjustment you don't want. `Cutout` zeroes a random `size x
size` square patch (DeVries & Taylor, 2017).

---

## 9. Streaming training

Everything above (`Trainer.Fit`) takes one fully-materialized `x, y
*nn.Tensor` and batches internally. For datasets that don't fit in memory
as a single tensor, `data.DataLoader` plus `Trainer.FitStream` give an
equivalent training loop driven by a stream of index batches instead:

```go
func data.NewDataLoader(n, batchSize int, rng *rand.Rand, shuffle bool) *DataLoader
func (d *DataLoader) Next() (batch []int, ok bool)
func (d *DataLoader) Reset()

func (t *Trainer) TrainOnBatch(xb, yb *nn.Tensor) (float32, error)
func (t *Trainer) FitStream(loader *data.DataLoader, convert func(batchIdx []int) (*nn.Tensor, *nn.Tensor, error), opts ...FitOption) (*History, error)
```

`DataLoader` only ever yields `[]int` index batches — it stays agnostic
to whatever actually backs your dataset (`[]*data.Image`, a lazy on-disk
source, ...). `FitStream` calls your `convert` function once per batch to
turn those indices into that batch's `(x, y)` tensors (mirroring
`examples/cifar10_cnn`'s own image-to-tensor conversion — `data` still
never imports `nn`), then trains on them via `TrainOnBatch`, the same
forward/loss/backward/optimizer-step primitive `Fit` itself is built on
(so the fused-softmax shortcut applies identically either way).
`FitStream` reuses `Fit`'s `FitOption` set for `ClipGrad`/`Callbacks`/
`Validation`/`Epochs`; `BatchSize`/`Shuffle`/`Seed` are `Fit`-only (the
loader already owns batching, shuffling, and its own RNG) and are
silently ignored if passed to `FitStream`.

`TrainOnBatch` uses `Trainer`'s own persistent RNG (seeded 42 by default)
across every call, rather than a fresh one per call — unlike `Fit`, which
creates its own RNG from `Fit`'s `Seed(...)` option every time it's
called — so Dropout-style randomness stays consistent across a whole
streamed run.

See `examples/tokenizer_stream/main.go` for a complete, runnable example
(pairs this with the `text` package's tokenizer — §11).

---

## 10. Recurrent layers and RoPE attention

### `RNN`, `LSTM`, `GRU`

```go
func RNN(rng *rand.Rand, features, hidden int, init Initializer) *RNNLayer
func LSTM(rng *rand.Rand, features, hidden int, init Initializer) *LSTMLayer
func GRU(rng *rand.Rand, features, hidden int, init Initializer) *GRULayer
func LastTimestep() *LastTimestepLayer
```

All three share one shape contract: input `[batch, seqLen, features]`,
output `[batch, seqLen, hidden]` — every timestep's hidden state, not
just the last, the same way `TransformerBlock` preserves the sequence
dimension. There's no `returnSequences` flag baked into the layer;
compose `LastTimestep()` after it to reduce to `[batch, hidden]` for
sequence classification, or stack another recurrent layer / a
per-timestep `nn.Linear` (already documented as accepting any `[...,
features]` input) directly on top:

```go
rng := nn.NewRNG(1)
model, err := nn.Sequential([]int{batch, seqLen, vocabSize},
    nn.Embedding(rng, vocabSize, dModel, nn.NormalInit(0, 0.5)),
    nn.LSTM(rng, dModel, hidden, nn.XavierInit()),
    nn.LastTimestep(),
    nn.Linear(rng, hidden, 1, nn.XavierInit()),
    nn.Sigmoid(),
)
```

`LSTM`'s combined gate weights are `Wx [features, 4*hidden]`/
`Wh [hidden, 4*hidden]` in (input, forget, cell-candidate, output) order;
`GRU`'s are `[features/hidden, 3*hidden]` in (reset, update, candidate)
order, with separate input-side (`Bx`) and hidden-side (`Bh`) biases
(PyTorch's convention — the reset gate only gates the hidden-side
contribution to the candidate, which needs the two bias vectors kept
separate rather than combined). Backward performs full
backpropagation-through-time: unlike every other layer in this package,
it cannot parallelize across timesteps (each depends on the previous
one's hidden/cell state), only across the batch dimension within a
timestep. Unidirectional only — no bidirectional wrapper or
packed-variable-length-sequence support (yet). See
`examples/sequence_rnn/main.go` for a complete, runnable example.

### `RotaryMultiHeadAttention`

```go
func RotaryMultiHeadAttention(rng *rand.Rand, dModel, numHeads int, causal bool, init Initializer) *RotaryMultiHeadAttentionLayer
```

Scaled-dot-product multi-head self-attention using Rotary Position
Embedding (Su et al., 2021, "RoFormer") instead of `PositionalEmbedding`:
Q and K are each rotated, per head and per pair of dimensions, by an
angle proportional to sequence position before the dot product — so
attention scores depend on *relative* position, not absolute — while V
is left untouched. Requires an even head dimension (`dModel/numHeads`).
Otherwise a drop-in replacement for `MultiHeadAttention` (same
constructor shape, same `Module` interface); grouped/multi-query
attention and KV-caching for autoregressive decoding are out of scope.

---

## 11. The `text` package

```go
func TrainBPE(corpus []string, vocabSize int) *BPETokenizer
func (b *BPETokenizer) Encode(s string) []int
func (b *BPETokenizer) Decode(ids []int) string
func (b *BPETokenizer) VocabSize() int
func (b *BPETokenizer) Save(path string) error
func LoadBPE(path string) (*BPETokenizer, error)
func LoadLineDataset(path string) ([]string, error)
```

A byte-level BPE tokenizer (GPT-2 style): the base vocabulary is exactly
the 256 possible byte values, and every learned merge combines two
existing token ids into one new one. Operating on raw bytes means every
possible input string is representable — there's no "unknown token" to
handle, and `Decode(Encode(s)) == s` always holds exactly, regardless of
spacing or Unicode content, because merges never cross a whitespace/
non-whitespace boundary (so no byte is ever discarded, just prevented
from merging across that boundary).

```go
tok := text.TrainBPE(corpus, 1000) // target vocab size 1000 (clamped up to >= 256)
ids := tok.Encode("hello, world")
s := tok.Decode(ids) // == "hello, world"

if err := tok.Save("tokenizer.json"); err != nil {
    panic(err)
}
loaded, err := text.LoadBPE("tokenizer.json")
```

`LoadLineDataset` reads a line-delimited text file into `[]string`
(skipping empty lines, stripping a trailing `\r`) — a plain corpus loader
to pair with `TrainBPE`. See `examples/tokenizer_stream/main.go` for a
complete example combining `text` with `data.DataLoader`/
`Trainer.FitStream` (§9) and `nn.Embedding`/`nn.GRU` (§10).

---

## 12. Developer and research tooling

### CLI: `neugo diff` and `neugo new`

```bash
go run ./cmd/neugo diff -a old_model.json -b new_model.json
go run ./cmd/neugo new -arch mlp -out ./my-project -pkg main
```

`diff` compares two `nn.Save`/`nn.SaveWithMetadata` files by decoding
their JSON directly (no dependency on `nn`'s internal types): it walks
both module trees position by position, reporting any architecture
difference (`TYPE CHANGED`, a module present in only one file) and, for
every matching leaf's params, the L2 norm of the weight delta — "did my
fine-tuning actually move this layer" at a glance, or a quick sanity check
when reviewing a PR that touches a committed model file.

`new` scaffolds a complete, working `main.go` for one of four
architectures (`mlp`, `cnn`, `transformer`, `rnn`) into a new directory —
the same "instant hello world" `cargo new`/`npm create` gives you, so you
never start from a blank page. Every generated template is a real,
compiling program against the current API (see
`cmd/neugo/new_test.go`'s `TestNewSubcommandGeneratesBuildableProject`),
not just a docs snippet that can silently rot.

### `nn.ShapeTrace` and `nn.SetDeterministic`

```go
func ShapeTrace(model *SequentialModel, inputShape []int) ([][]int, error)
func SetDeterministic(enabled bool)
func IsDeterministic() bool
```

`ShapeTrace` re-runs the same `OutputShape` chain `Summary` does, but
returns each module's output shape as `[][]int` instead of a formatted
string — for tests/tooling that want to assert on or otherwise consume
shapes programmatically.

`SetDeterministic(true)` forces every parallelized hot path (`Conv2D`,
`Linear`, the norm layers, attention, the RNN family, ...) to run
single-threaded in a fixed reduction order, so repeated runs on the same
inputs produce **bit-exact** identical results regardless of
`GOMAXPROCS` — a process-wide switch (like `GOMAXPROCS` itself, or
PyTorch's `use_deterministic_algorithms`), off by default since it trades
throughput for exact reproducibility. Turn it on when you need to verify
a paper's reported numbers or bisect a suspected numerical issue, not for
normal training. `train`'s own parallelism (`Optimizer.Step`,
`CrossValidate`'s fold-level concurrency) checks the same switch, so
`SetDeterministic(true)` covers both packages with one call.

### Generated docs and fuzzing

`docs/LAYERS.generated.md` is a flat, always-current list of every `nn`
constructor and its doc comment, regenerated via `go generate ./...`
(wired through a `//go:generate` directive in `nn/doc.go` to
`cmd/gendocs`) — it can't silently drift from the code the way a
hand-maintained table can.

A handful of layers (`Linear`, `Conv2D`, `LSTM`, `MultiHeadAttention`,
`RMSNorm`) have `go test -fuzz` targets (e.g. `FuzzLinearGradient`) that
fuzz over shape dimensions and run the same gradcheck helpers the
hand-written tests use — `go test ./nn/ -fuzz=FuzzLinearGradient
-fuzztime=30s` to fuzz one for longer than CI normally would.

`nn`'s own convention of "every layer needs a serialize.go case and a
test" is itself enforced by a test:
`TestEveryModuleTypeHasSerializationAndTests` statically scans the
package for types implementing `Module` and fails if either is missing —
catching a forgotten case the moment a new layer is added.

### Training visibility: `TUI`, `GradientHistogram`, `ExperimentLog`

```go
train.TUI(opt Optimizer, totalEpochs int) *TUICallback
train.GradientHistogram(bins, printEvery int) *GradientHistogramCallback
train.ExperimentLog(path string, meta map[string]string) *ExperimentLogCallback
```

Three opt-in `Callback`s, composed via `train.Callbacks(...)` like any
other:

- **`TUI`** redraws a compact two-line terminal dashboard in place after
  every epoch — current loss/LR/gradient-norm, plus a Unicode-block
  sparkline of recent training loss — using only ANSI cursor-movement
  escapes (no external TUI library). Prefer `ProgressBar` for
  non-interactive/redirected-to-a-file output.
- **`GradientHistogram`** prints an ASCII log-scale histogram of gradient
  magnitudes every `printEvery` epochs — a quick way to spot vanishing
  (everything clustered at very negative exponents) or exploding (a long
  tail at large positive exponents) gradients without a GUI.
- **`ExperimentLog`** appends one JSON line per event (run start, each
  epoch, run end) to a JSONL file, opened in append mode so multiple runs
  (e.g. every trial in a `tune` sweep) can share one log — a dependency-
  free alternative to an external experiment tracker for comparing sweeps
  later. Write failures land in `.LastError`, matching
  `ModelCheckpointCallback`'s convention, rather than panicking or failing
  `Fit`.

```go
trainer := train.New(model, opt, train.BCELoss())
hist, err := trainer.Fit(x, y,
    train.Epochs(500), train.BatchSize(32),
    train.Callbacks(
        train.TUI(opt, 500),
        train.GradientHistogram(10, 50),
        train.ExperimentLog("runs.jsonl", map[string]string{"lr": "0.01"}),
    ),
)
```

Pair `ExperimentLog`'s JSONL output with `nn.SaveWithMetadata`'s
`Manifest` field (§7) for full reproducibility bookkeeping: one records
the training trajectory, the other records what produced the final
checkpoint.

### Browser demo

`examples/wasm_demo` trains a small model, exports it via `export` (§
docs/EXPORT_GUIDE.md) to dependency-free Go source, compiles that for
`GOOS=js GOARCH=wasm`, and runs it in a browser tab with no server-side
inference — `examples/wasm_demo/build.sh` runs the whole pipeline in one
command; see that directory's `README.md`.

---

## 13. Migrating from the old `Network` API

This restructure deleted the old `Network` package entirely rather than
deprecating it — the old code had the functional bugs described in the
design doc (dead softmax, dead optimizers, binary-only CNN training,
`History` accumulating across calls, a mislabeled GELU, and four
different, partially-broken ways to construct a model). There is no
compatibility shim; every concept below has to be rewritten, not just
renamed.

| Old (`Network` package) | New (`nn`/`train`) |
|---|---|
| `Network.NewNetworkWithLoss(...)` and friends (`NewLayer`/`NewNetwork` low-level construction) | `nn.Sequential(inputShape, modules...)` — one construction path, validated at build time |
| The fluent `Network.NewSequential()...Build()` builder style | `nn.Sequential(inputShape, modules...)` — same idea, one supported implementation, and validated at construction instead of at first `Forward` |
| The functional `Stack`/`L` builder API | `nn.Sequential(...)` — same idea, one supported implementation |
| The NNX-style forward-only modules (`Network/nnx.go`) | `nn`'s `Module`s are trainable by default — no separate forward-only variant exists or is needed |
| `Network.Fit` / `FitWithValidation` / `FitWithScheduler` / `Network.Train` (six overlapping training loops) | `train.Trainer.Fit(x, y, opts...)` — one loop; validation, scheduling, and early stopping are all `FitOption`s/`Callback`s on top of it (§3–4) |
| `CNN.Train` / `CNN.QuickFit` (separate, binary-only CNN training path) | the same `train.Trainer.Fit` — CNNs are just `nn.Sequential` models with `Conv2D`/pooling layers, so there's one training loop for both, and it's multiclass-correct |
| `Network.Predict` | `Trainer.Predict(x)` |
| Manually wiring an `Optimizer` (which was actually dead code — training always fell back to raw-LR SGD) | `train.SGD`/`Momentum`/`Adam`/`RMSprop` are wired into `Trainer.Fit` and actually update weights |
| "Softmax" activation (silently aliased to sigmoid due to a missing switch case) | `nn.Softmax()` is a real softmax layer, with the fused-gradient shortcut in `train.CrossEntropyLoss` (§3) |
| `GELUFunc` (a SiLU approximation mislabeled as GELU) | `nn.GELU()` computes the exact `0.5*x*(1+erf(x/√2))` formula |
| JSON models saved by the old `Network`/`CNN` serialization | **not loadable by `nn.Load`** — the JSON schema changed (see §7). Any model trained on the old code must be retrained from scratch on the new API; there is no converter, by design (see the design doc's non-goals). |

If you're porting a training script: replace whichever of the six old
fit variants you used with `trainer.Fit(x, y, ...)` plus the
`FitOption`s that match what you needed (validation → `train.Validation`,
scheduling → a scheduler `Callback`, early stopping →
`train.EarlyStopping`), and replace direct `CNN`/`Network` struct
construction with `nn.Sequential`. The six worked examples in
`examples/` are a faster reference than the old guides ever were —
`callbacks/main.go` in particular exercises the validation + scheduler +
early-stopping + checkpoint + progress-bar combination in one file.
