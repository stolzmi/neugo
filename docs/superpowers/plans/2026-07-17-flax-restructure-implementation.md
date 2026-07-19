# NeuGo Flax-Style Restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure NeuGo from the broken, duplicated `Network`/`data`/`tensor` packages into two idiomatic Go packages — `nn` (model definition, everything is a `Module`) and `train` (one training engine) — fixing every functional bug in the design doc and adding the selected missing features, with a real test suite.

**Architecture:** `nn.Module` is the single abstraction (Linear, Conv2D, pooling, Flatten, Dropout, BatchNorm, activations, Sequential all implement it). `train.Trainer` owns the one epoch loop (shuffle → forward → loss → backward → optimizer step → callbacks). `train` imports `nn`; `nn` never imports `train`; `data` is standalone.

**Tech Stack:** Go 1.25, standard library only (`math`, `math/rand`, `encoding/json`, `testing`) — zero third-party dependencies, matching the existing `go.mod`.

## Global Constraints

- No `panic` in library code (`nn`, `train`, `data`); no builtin `println`; no stdout writes except from `ProgressBar`/`Summary`.
- No global mutable state: no package-level `rand` source anywhere, no `rand.Seed` (deprecated). Every source of randomness is an explicit `*rand.Rand` threaded through constructors/options.
- Library code never mutates a caller-owned slice/tensor unless the function's doc comment says so.
- All floating point is `float32`. Internal math (log, exp, sqrt, erf, pow) uses `float64` via the standard `math` package and casts back — never hand-rolled numeric approximations.
- Constructors and `Sequential` return `(T, error)` on invalid configuration; `Forward`/`Backward` return errors only for shape violations that should be impossible after a validated `Sequential` build.
- Every new source file gets a sibling `_test.go` in the same package. Tests are seeded deterministically (fixed `int64` seeds, never time-based). `go build ./...`, `go vet ./...`, and `go test ./...` must be green at the end of every phase.
- Gradient-check tolerance: central differences, `eps = 1e-2`, absolute-difference tolerance `1e-2`, float32 throughout (per §4.7 of the design doc — float32 checks are inherently noisier than float64, hence the looser-than-1e-3-sounding-but-consistent tolerance chosen here for reliability across CI; if a specific check proves flaky, widening its own local tolerance is acceptable, silently loosening the global default is not).

## Design decisions not fully pinned down by the design doc

The design doc (`docs/superpowers/specs/2026-07-17-flax-restructure-design.md`) is architecturally complete but its code samples are illustrative — they omit error handling (despite §4.2 prose requiring it) and omit RNG plumbing (despite §1 flagging global RNG state as a bug to fix). This plan resolves every such gap explicitly so no task requires improvisation:

1. **RNG threading.** `nn.NewRNG(seed int64) *rand.Rand` (`nn/rng.go`) is the only RNG constructor in the codebase. Every module constructor that needs randomness at build time (`Linear`, `Conv2D`) takes `rng *rand.Rand` as its **first** parameter. `Dropout` needs no RNG at construction — it draws from `ctx.RNG` inside `Forward`, only in `Train` mode. `train.Trainer` owns one `*rand.Rand` (seeded via the `train.Seed(n int64)` option, default seed `42`) used both for per-epoch shuffling and as `ctx.RNG` during every `Forward`/`Backward` call so `Dropout` sees it.
2. **Lazy shape inference (`inFeatures == 0`).** `nn.Sequential` takes an explicit starting shape: `func Sequential(inputShape []int, modules ...Module) (*SequentialModel, error)`. It calls `OutputShape` on each module in order, threading the shape forward. For a `Linear` built with `inFeatures == 0`, its `OutputShape` call is the point where it learns the real input width, allocates `W`/`B`, and initializes them from the `rng`/`Initializer` captured at construction — this is *also* the only place lazy modules get built, so a model must always go through `Sequential` (or call `OutputShape` manually) before `Forward`.
3. **Softmax+CrossEntropy fusion.** `train.CrossEntropyLoss` has an internal `fused bool`, set by `train.New` when the model's last module is `*nn.SoftmaxModule`. In both fused and non-fused mode the returned gradient is the same formula, `(probs - target) / batch`, where `probs` is either `pred` directly (fused — `pred` already *is* softmax output) or an internally-computed `softmax(pred)` (non-fused — `pred` is raw logits, e.g. model ends in `Linear` with no trailing `Softmax`). The **only** behavioral difference is what `Trainer.Fit` does with that gradient: fused mode skips calling `Backward` on the model's last module (the `Softmax`) and feeds the gradient directly into the second-to-last module, exactly realizing the "fused shortcut" the design doc describes; non-fused mode calls `Backward` on every module normally.
4. **Activation derivative convention.** Every `nn.ActivationModule`'s `deriv` function takes the **pre-activation input** `x`, not the post-activation output (unlike the old `Network/activation.go`, which mixed the two). This is required for `GELU`, whose derivative has no clean closed form in terms of its own output, and is applied uniformly to `ReLU`/`Sigmoid`/`Tanh`/`LeakyReLU` for consistency. Each `ActivationModule` caches its input tensor in `Forward` for this reason.
5. **BatchNorm generality.** Channels are always the fastest-varying (last) dimension in this codebase's tensor layout — `[batch, features]` for dense, `[batch, h, w, channels]` for conv. `BatchNorm` exploits this: for a flat index `idx`, its channel is `idx % channels`, and its per-channel statistics are computed over all `N = size/channels` elements sharing that channel. This one implementation serves both the dense and conv cases the design doc calls out in §4.2, with no special-casing on tensor rank.
6. **Gradient normalization happens exactly once, in the Loss.** Every `Loss.Loss` in Task 7 divides its returned gradient by whatever count matches its own definition of "batch-mean" (total elements for MSE/MAE, batch size only for BCE/CrossEntropy, matching each loss's standard mathematical convention). Every module's `Backward` downstream is then plain, unscaled chain rule — no module re-divides its `Param.Grad` by batch size or element count. A module that did would silently shrink its parameter gradients by an extra factor and fail its own gradient-check test (this is not hypothetical — an earlier draft of this plan had exactly that bug in `LinearLayer.Backward`, caught only by working through `BatchNorm`'s backward derivation by hand; the fix is reflected in Task 4's `Backward` above and Task 13's below).
7. **Conv2D shape/padding.** `Conv2D(rng, inChannels, outChannels, kernelSize, init)` is stride-1, zero-padding ("valid"). `Conv2DSame(rng, inChannels, outChannels, kernelSize, init)` requires an odd `kernelSize` and applies padding `(kernelSize-1)/2` ("same"). Padding is applied via bounds-checking during the convolution loops (skip out-of-range taps), not by materializing a padded copy — simpler than the old `addPadding`/`removePadding` pair and equally correct.

## File Structure

```
neugo/
├── nn/
│   ├── tensor.go        # Tensor{Data []float32, Shape []int}; NewTensor, NewTensorFromData, Size, Clone
│   ├── module.go         # Param, Mode, Context, Module interface
│   ├── rng.go             # NewRNG(seed int64) *rand.Rand
│   ├── init.go            # Initializer func type; XavierInit, HeInit, ZerosInit, UniformInit, NormalInit
│   ├── linear.go          # LinearLayer
│   ├── activation.go      # ActivationModule (ReLU/Sigmoid/Tanh/LeakyReLU/GELU) + SoftmaxModule
│   ├── sequential.go      # SequentialModel
│   ├── conv.go            # Conv2DLayer, Conv2DSame
│   ├── pooling.go         # MaxPool2DLayer, AvgPool2DLayer
│   ├── flatten.go         # FlattenLayer
│   ├── dropout.go         # DropoutLayer
│   ├── norm.go             # BatchNormLayer
│   ├── summary.go         # Summary(model, inputShape), ParamCount(model)
│   ├── serialize.go       # Save(model, path), Load(path)
│   └── *_test.go          # one per file above, plus gradcheck_test.go (shared helper)
├── train/
│   ├── loss.go             # Loss interface; MSELoss, BCELoss, CrossEntropyLoss, MAELoss
│   ├── metrics.go         # Metrics struct; computeMetrics
│   ├── optimizer.go       # Optimizer interface; SGD, Momentum, Adam, RMSprop, ClipNorm
│   ├── callback.go         # Callback interface; History, EarlyStopping, ModelCheckpoint, ProgressBar, CallbackList
│   ├── scheduler.go       # StepDecay, ExponentialDecay, CosineAnnealing, Warmup, ReduceLROnPlateau (all Callbacks)
│   ├── trainer.go          # Trainer, New, Fit, Predict, Evaluate, FitOption's
│   ├── crossval.go        # Fold, KFoldSplits, StratifiedKFoldSplits, CrossValidate, CrossValResult
│   └── *_test.go
├── data/                   # fixed in place; same public API shape, explicit RNG, no compile errors
├── examples/
│   ├── xor/main.go
│   ├── wine_quality/main.go
│   ├── fashion_mnist/main.go
│   ├── cifar10_cnn/main.go
│   ├── callbacks/main.go
│   └── crossval/main.go
├── docs/
│   ├── GUIDE.md
│   └── superpowers/{specs,plans}/
├── README.md
└── .gitignore
```

Deleted at the end of Phase 3: `Network/` (whole directory) — per design doc §5, once its replacement is proven by Phase 1–3 tests.
Deleted during Phase 4: `tensor/` (superseded by `nn/tensor.go`) — deferred to Task 22, since `data/image.go` and `data/cifar10.go` still import `neugo/tensor` for `Tensor3D` until that task migrates them to `nn.Tensor`; deleting `tensor/` any earlier would break `go build ./...` on the `data` package. Also deleted during Phase 4: root `main.go`, all `examples/*.go` (flat files), root-level `*.json` model artifacts, `test_predictions.csv`, `API_SUMMARY.md`, `BEFORE_AFTER.md`, `CLEAN_API_GUIDE.md`, `CNN_GUIDE.md`, `FEATURE_GUIDE.md`, `FUNCTIONAL_API_GUIDE.md`, `NNX_API_GUIDE.md`, `QUICK_REFERENCE.md`.

---

## Phase 1: nn core

Tensor, Module/Param/Context, initializers, RNG, Linear, activations (real Softmax + exact GELU), Sequential with shape validation. Gradient-check tests green.

### Task 1: Tensor

**Files:**
- Create: `nn/tensor.go`
- Test: `nn/tensor_test.go`

**Interfaces:**
- Produces: `type Tensor struct { Data []float32; Shape []int }`, `func NewTensor(shape []int) *Tensor`, `func NewTensorFromData(data []float32, shape []int) (*Tensor, error)`, `func (t *Tensor) Size() int`, `func (t *Tensor) Clone() *Tensor`.

- [ ] **Step 1: Write the failing test**

```go
package nn

import "testing"

func TestNewTensorZeroed(t *testing.T) {
	tn := NewTensor([]int{2, 3})
	if tn.Size() != 6 {
		t.Fatalf("Size() = %d, want 6", tn.Size())
	}
	if len(tn.Data) != 6 {
		t.Fatalf("len(Data) = %d, want 6", len(tn.Data))
	}
	for i, v := range tn.Data {
		if v != 0 {
			t.Errorf("Data[%d] = %v, want 0", i, v)
		}
	}
}

func TestNewTensorFromDataShapeMismatch(t *testing.T) {
	_, err := NewTensorFromData([]float32{1, 2, 3}, []int{2, 2})
	if err == nil {
		t.Fatal("expected error for mismatched shape/data length, got nil")
	}
}

func TestTensorClone(t *testing.T) {
	orig, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	clone := orig.Clone()
	clone.Data[0] = 99
	if orig.Data[0] == 99 {
		t.Fatal("Clone shares underlying data with original")
	}
	if clone.Shape[0] != 2 || clone.Shape[1] != 2 {
		t.Fatalf("Clone shape = %v, want [2 2]", clone.Shape)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./nn/... -run TestNewTensor -v`
Expected: FAIL — `nn` package doesn't exist / `NewTensor` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import "fmt"

// Tensor is the single data type flowing through the network.
// Batch-first: dense is [batch, features], conv is [batch, h, w, channels].
type Tensor struct {
	Data  []float32
	Shape []int
}

func shapeSize(shape []int) int {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return size
}

func NewTensor(shape []int) *Tensor {
	return &Tensor{
		Data:  make([]float32, shapeSize(shape)),
		Shape: append([]int(nil), shape...),
	}
}

func NewTensorFromData(data []float32, shape []int) (*Tensor, error) {
	if want := shapeSize(shape); want != len(data) {
		return nil, fmt.Errorf("nn: data length %d does not match shape %v (size %d)", len(data), shape, want)
	}
	return &Tensor{Data: data, Shape: append([]int(nil), shape...)}, nil
}

func (t *Tensor) Size() int {
	return len(t.Data)
}

func (t *Tensor) Clone() *Tensor {
	d := make([]float32, len(t.Data))
	copy(d, t.Data)
	return &Tensor{Data: d, Shape: append([]int(nil), t.Shape...)}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./nn/... -run TestNewTensor -v` and `go test ./nn/... -run TestTensorClone -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn/tensor.go nn/tensor_test.go
git commit -m "feat(nn): add Tensor core type"
```

### Task 2: Param, Mode, Context, Module interface, RNG

**Files:**
- Create: `nn/module.go`, `nn/rng.go`
- Test: `nn/module_test.go`

**Interfaces:**
- Consumes: `nn.Tensor`, `nn.NewTensor` (Task 1).
- Produces: `type Param struct { Value, Grad *Tensor }`, `func NewParam(value *Tensor) *Param`, `func (p *Param) ZeroGrad()`, `type Mode int` with `Inference`/`Train` constants, `type Context struct { Mode Mode; RNG *rand.Rand }`, `type Module interface { Forward(ctx *Context, x *Tensor) (*Tensor, error); Backward(ctx *Context, gradOut *Tensor) (*Tensor, error); Params() []*Param; OutputShape(inShape []int) ([]int, error) }`, `func NewRNG(seed int64) *rand.Rand`.

- [ ] **Step 1: Write the failing test**

```go
package nn

import "testing"

func TestNewParamGradShapeMatchesValue(t *testing.T) {
	v, _ := NewTensorFromData([]float32{1, 2, 3}, []int{3})
	p := NewParam(v)
	if p.Grad.Size() != p.Value.Size() {
		t.Fatalf("Grad size = %d, want %d", p.Grad.Size(), p.Value.Size())
	}
}

func TestParamZeroGrad(t *testing.T) {
	v := NewTensor([]int{2})
	p := NewParam(v)
	p.Grad.Data[0], p.Grad.Data[1] = 5, -3
	p.ZeroGrad()
	for i, g := range p.Grad.Data {
		if g != 0 {
			t.Errorf("Grad[%d] = %v after ZeroGrad, want 0", i, g)
		}
	}
}

func TestNewRNGDeterministic(t *testing.T) {
	a := NewRNG(42)
	b := NewRNG(42)
	for i := 0; i < 5; i++ {
		if a.Float32() != b.Float32() {
			t.Fatal("NewRNG(42) produced different sequences across two instances")
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./nn/... -run "TestNewParam|TestParamZeroGrad|TestNewRNG" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
// nn/module.go
package nn

import "math/rand"

// Param is a trainable tensor pair an Optimizer can see and update.
type Param struct {
	Value *Tensor
	Grad  *Tensor
}

func NewParam(value *Tensor) *Param {
	return &Param{Value: value, Grad: NewTensor(value.Shape)}
}

func (p *Param) ZeroGrad() {
	for i := range p.Grad.Data {
		p.Grad.Data[i] = 0
	}
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
	RNG  *rand.Rand
}

type Module interface {
	Forward(ctx *Context, x *Tensor) (*Tensor, error)
	Backward(ctx *Context, gradOut *Tensor) (*Tensor, error)
	Params() []*Param
	OutputShape(inShape []int) ([]int, error)
}
```

```go
// nn/rng.go
package nn

import "math/rand"

func NewRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./nn/... -run "TestNewParam|TestParamZeroGrad|TestNewRNG" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn/module.go nn/rng.go nn/module_test.go
git commit -m "feat(nn): add Param, Context, Module interface, explicit RNG"
```

### Task 3: Initializers

**Files:**
- Create: `nn/init.go`
- Test: `nn/init_test.go`

**Interfaces:**
- Consumes: `nn.Tensor`, `nn.NewTensor`, `nn.NewRNG` (Tasks 1–2).
- Produces: `type Initializer func(rng *rand.Rand, shape []int) *Tensor`, `func XavierInit() Initializer`, `func HeInit() Initializer`, `func ZerosInit() Initializer`, `func UniformInit(low, high float32) Initializer`, `func NormalInit(mean, std float32) Initializer`.

- [ ] **Step 1: Write the failing test**

```go
package nn

import (
	"math"
	"testing"
)

func TestZerosInit(t *testing.T) {
	rng := NewRNG(1)
	tn := ZerosInit()(rng, []int{4, 4})
	for _, v := range tn.Data {
		if v != 0 {
			t.Fatalf("ZerosInit produced non-zero value %v", v)
		}
	}
}

func TestUniformInitBounds(t *testing.T) {
	rng := NewRNG(1)
	tn := UniformInit(-0.5, 0.5)(rng, []int{1000})
	for _, v := range tn.Data {
		if v < -0.5 || v >= 0.5 {
			t.Fatalf("UniformInit(-0.5,0.5) produced out-of-range value %v", v)
		}
	}
}

func TestHeInitMomentBounds(t *testing.T) {
	rng := NewRNG(1)
	fanIn := 256
	tn := HeInit()(rng, []int{fanIn, 64})
	var sumSq float64
	for _, v := range tn.Data {
		sumSq += float64(v) * float64(v)
	}
	variance := sumSq / float64(len(tn.Data))
	wantVariance := 2.0 / float64(fanIn)
	if math.Abs(variance-wantVariance)/wantVariance > 0.25 {
		t.Fatalf("He-initialized variance = %v, want close to %v", variance, wantVariance)
	}
}

func TestXavierInitConv4DShape(t *testing.T) {
	rng := NewRNG(1)
	// Conv weight convention: [outC, inC, kh, kw]
	tn := XavierInit()(rng, []int{8, 3, 3, 3})
	if tn.Size() != 8*3*3*3 {
		t.Fatalf("Size() = %d, want %d", tn.Size(), 8*3*3*3)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./nn/... -run "TestZerosInit|TestUniformInit|TestHeInit|TestXavierInit" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import (
	"math"
	"math/rand"
)

// Initializer fills a freshly-shaped Tensor with starting weights.
type Initializer func(rng *rand.Rand, shape []int) *Tensor

// fanInOut supports the two weight-tensor shapes used in this codebase:
// Linear weights are [in, out]; Conv2D kernels are [outC, inC, kh, kw].
func fanInOut(shape []int) (fanIn, fanOut int) {
	switch len(shape) {
	case 2:
		return shape[0], shape[1]
	case 4:
		receptive := shape[2] * shape[3]
		return shape[1] * receptive, shape[0] * receptive
	default:
		n := shapeSize(shape)
		return n, n
	}
}

func XavierInit() Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		fanIn, fanOut := fanInOut(shape)
		limit := float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
		t := NewTensor(shape)
		for i := range t.Data {
			t.Data[i] = (rng.Float32()*2 - 1) * limit
		}
		return t
	}
}

func HeInit() Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		fanIn, _ := fanInOut(shape)
		std := float32(math.Sqrt(2.0 / float64(fanIn)))
		t := NewTensor(shape)
		for i := range t.Data {
			t.Data[i] = float32(rng.NormFloat64()) * std
		}
		return t
	}
}

func ZerosInit() Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		return NewTensor(shape)
	}
}

func UniformInit(low, high float32) Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		t := NewTensor(shape)
		for i := range t.Data {
			t.Data[i] = low + rng.Float32()*(high-low)
		}
		return t
	}
}

func NormalInit(mean, std float32) Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		t := NewTensor(shape)
		for i := range t.Data {
			t.Data[i] = mean + float32(rng.NormFloat64())*std
		}
		return t
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./nn/... -run "TestZerosInit|TestUniformInit|TestHeInit|TestXavierInit" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn/init.go nn/init_test.go
git commit -m "feat(nn): add Xavier/He/Zeros/Uniform/Normal initializers"
```

### Task 4: Gradient-check helper + Linear layer

**Files:**
- Create: `nn/gradcheck_test.go`, `nn/linear.go`
- Test: `nn/linear_test.go`

**Interfaces:**
- Consumes: `nn.Tensor`, `nn.Param`, `nn.Context`, `nn.Module`, `nn.Initializer`, `nn.XavierInit` (Tasks 1–3).
- Produces: `func Linear(rng *rand.Rand, inFeatures, outFeatures int, init Initializer) *LinearLayer`, satisfying `nn.Module`. `gradcheck_test.go` produces two unexported test helpers, `checkInputGradient(t, m Module, ctx *Context, x *Tensor)` and `checkParamGradient(t *testing.T, forward func() (*Tensor, error), backward func(*Tensor) (*Tensor, error), p *Param)`, reused by every later gradient-checked module (Tasks 5, 12, 13, 15, 16).

- [ ] **Step 1: Write the gradient-check helper (test infrastructure, not itself a failing test)**

```go
// nn/gradcheck_test.go
package nn

import (
	"math"
	"testing"
)

const gradCheckEps = 1e-2
const gradCheckTol = 1e-2

func sumTensor(t *Tensor) float32 {
	var s float32
	for _, v := range t.Data {
		s += v
	}
	return s
}

// checkInputGradient verifies m.Backward's returned input-gradient against
// central finite differences of sum(m.Forward(x)), perturbing one element
// of x at a time. Use for modules with no learnable Params, or to check the
// input-gradient path of a module that also has Params (call
// checkParamGradient separately for those).
func checkInputGradient(t *testing.T, m Module, ctx *Context, x *Tensor) {
	t.Helper()
	y, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	gradOut := NewTensor(y.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradIn, err := m.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}
	for i := range x.Data {
		orig := x.Data[i]

		x.Data[i] = orig + gradCheckEps
		yPlus, _ := m.Forward(ctx, x)

		x.Data[i] = orig - gradCheckEps
		yMinus, _ := m.Forward(ctx, x)

		x.Data[i] = orig

		numGrad := (sumTensor(yPlus) - sumTensor(yMinus)) / (2 * gradCheckEps)
		if diff := math.Abs(float64(numGrad - gradIn.Data[i])); diff > gradCheckTol {
			t.Errorf("input gradient mismatch at index %d: analytic=%v numeric=%v", i, gradIn.Data[i], numGrad)
		}
	}
	// Restore module state (input cache etc.) to the unperturbed forward pass.
	m.Forward(ctx, x)
}

// checkParamGradient verifies backward's accumulated gradient on p against
// central finite differences of sum(forward()), perturbing one element of
// p.Value at a time. forward/backward must be closures over the module
// under test (e.g. `func() (*Tensor, error) { return layer.Forward(ctx, x) }`).
func checkParamGradient(t *testing.T, forward func() (*Tensor, error), backward func(*Tensor) (*Tensor, error), p *Param) {
	t.Helper()
	y, err := forward()
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	gradOut := NewTensor(y.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	if _, err := backward(gradOut); err != nil {
		t.Fatalf("backward: %v", err)
	}
	analytic := append([]float32(nil), p.Grad.Data...)

	for i := range p.Value.Data {
		orig := p.Value.Data[i]

		p.Value.Data[i] = orig + gradCheckEps
		yPlus, _ := forward()

		p.Value.Data[i] = orig - gradCheckEps
		yMinus, _ := forward()

		p.Value.Data[i] = orig

		numGrad := (sumTensor(yPlus) - sumTensor(yMinus)) / (2 * gradCheckEps)
		if diff := math.Abs(float64(numGrad - analytic[i])); diff > gradCheckTol {
			t.Errorf("param gradient mismatch at index %d: analytic=%v numeric=%v", i, analytic[i], numGrad)
		}
	}
	// Restore the module's cached forward state.
	forward()
}
```

- [ ] **Step 2: Write the failing test for Linear**

```go
// nn/linear_test.go
package nn

import "testing"

func TestLinearForwardShape(t *testing.T) {
	rng := NewRNG(1)
	l := Linear(rng, 3, 4, XavierInit())
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	ctx := &Context{Mode: Inference}
	y, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if y.Shape[0] != 2 || y.Shape[1] != 4 {
		t.Fatalf("output shape = %v, want [2 4]", y.Shape)
	}
}

func TestLinearOutputShapeInfersInFeatures(t *testing.T) {
	rng := NewRNG(1)
	l := Linear(rng, 0, 5, XavierInit())
	out, err := l.OutputShape([]int{8, 12})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	if out[0] != 8 || out[1] != 5 {
		t.Fatalf("OutputShape = %v, want [8 5]", out)
	}
	if len(l.Params()) != 2 || l.Params()[0].Value.Shape[0] != 12 {
		t.Fatalf("Linear did not build W with inferred inFeatures=12: %+v", l.Params()[0].Value.Shape)
	}
}

func TestLinearInputGradient(t *testing.T) {
	rng := NewRNG(2)
	l := Linear(rng, 3, 2, XavierInit())
	x, _ := NewTensorFromData([]float32{0.5, -1.2, 0.3, 1.1, 0.2, -0.7}, []int{2, 3})
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, l, ctx, x)
}

func TestLinearParamGradients(t *testing.T) {
	rng := NewRNG(3)
	l := Linear(rng, 3, 2, XavierInit())
	x, _ := NewTensorFromData([]float32{0.5, -1.2, 0.3, 1.1, 0.2, -0.7}, []int{2, 3})
	ctx := &Context{Mode: Inference}
	forward := func() (*Tensor, error) { return l.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return l.Backward(ctx, g) }
	checkParamGradient(t, forward, backward, l.Params()[0]) // W
	checkParamGradient(t, forward, backward, l.Params()[1]) // B
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `go test ./nn/... -run TestLinear -v`
Expected: FAIL — `Linear` undefined.

- [ ] **Step 4: Write minimal implementation**

```go
// nn/linear.go
package nn

import (
	"fmt"
	"math/rand"
)

type LinearLayer struct {
	inFeatures, outFeatures int
	W, B                    *Param
	init                    Initializer
	rng                     *rand.Rand
	input                   *Tensor
}

// Linear creates a dense layer. inFeatures == 0 defers weight allocation
// until OutputShape is called with the real preceding shape (see design
// decision #2 in the plan header).
func Linear(rng *rand.Rand, inFeatures, outFeatures int, init Initializer) *LinearLayer {
	if init == nil {
		init = XavierInit()
	}
	l := &LinearLayer{inFeatures: inFeatures, outFeatures: outFeatures, init: init, rng: rng}
	if inFeatures > 0 {
		l.build(inFeatures)
	}
	return l
}

func (l *LinearLayer) build(inFeatures int) {
	l.inFeatures = inFeatures
	l.W = NewParam(l.init(l.rng, []int{inFeatures, l.outFeatures}))
	l.B = NewParam(NewTensor([]int{l.outFeatures}))
}

func (l *LinearLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 2 {
		return nil, fmt.Errorf("nn: Linear expects input shape [batch, features], got %v", inShape)
	}
	in := inShape[1]
	if l.inFeatures == 0 {
		l.build(in)
	} else if l.inFeatures != in {
		return nil, fmt.Errorf("nn: Linear configured for %d input features, got %d", l.inFeatures, in)
	}
	return []int{inShape[0], l.outFeatures}, nil
}

func (l *LinearLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if l.W == nil {
		return nil, fmt.Errorf("nn: Linear not built — call OutputShape or construct via Sequential first")
	}
	if len(x.Shape) != 2 || x.Shape[1] != l.inFeatures {
		return nil, fmt.Errorf("nn: Linear expected input shape [batch, %d], got %v", l.inFeatures, x.Shape)
	}
	l.input = x
	batch := x.Shape[0]
	out := NewTensor([]int{batch, l.outFeatures})
	for b := 0; b < batch; b++ {
		for o := 0; o < l.outFeatures; o++ {
			sum := l.B.Value.Data[o]
			for i := 0; i < l.inFeatures; i++ {
				sum += x.Data[b*l.inFeatures+i] * l.W.Value.Data[i*l.outFeatures+o]
			}
			out.Data[b*l.outFeatures+o] = sum
		}
	}
	return out, nil
}

// Backward implements plain chain rule with no extra batch normalization:
// gradOut already carries whatever batch-scaling the Loss applied (see
// Task 7), so W.Grad/B.Grad are raw sums over the batch, exactly like
// gradIn — introducing an additional /batch here would silently shrink
// every parameter gradient by a factor of batch and fail the Task 4
// gradient-check tests.
func (l *LinearLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch := l.input.Shape[0]
	gradIn := NewTensor([]int{batch, l.inFeatures})
	l.W.ZeroGrad()
	l.B.ZeroGrad()
	for b := 0; b < batch; b++ {
		for o := 0; o < l.outFeatures; o++ {
			g := gradOut.Data[b*l.outFeatures+o]
			l.B.Grad.Data[o] += g
			for i := 0; i < l.inFeatures; i++ {
				l.W.Grad.Data[i*l.outFeatures+o] += g * l.input.Data[b*l.inFeatures+i]
				gradIn.Data[b*l.inFeatures+i] += g * l.W.Value.Data[i*l.outFeatures+o]
			}
		}
	}
	return gradIn, nil
}

func (l *LinearLayer) Params() []*Param {
	return []*Param{l.W, l.B}
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./nn/... -run TestLinear -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nn/gradcheck_test.go nn/linear.go nn/linear_test.go
git commit -m "feat(nn): add Linear layer with gradient-check test harness"
```

### Task 5: Activations (ReLU, Sigmoid, Tanh, LeakyReLU, GELU) + Softmax

**Files:**
- Create: `nn/activation.go`
- Test: `nn/activation_test.go`

**Interfaces:**
- Consumes: `nn.Module`, `nn.Tensor`, `checkInputGradient` (Tasks 1, 2, 4).
- Produces: `func ReLU() *ActivationModule`, `func Sigmoid() *ActivationModule`, `func Tanh() *ActivationModule`, `func LeakyReLU(alpha float32) *ActivationModule`, `func GELU() *ActivationModule`, `func Softmax() *SoftmaxModule` — all satisfy `nn.Module`.

- [ ] **Step 1: Write the failing tests**

```go
package nn

import (
	"math"
	"testing"
)

func TestReLUForward(t *testing.T) {
	x, _ := NewTensorFromData([]float32{-2, -0.5, 0, 1, 3}, []int{5})
	y, err := ReLU().Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []float32{0, 0, 0, 1, 3}
	for i := range want {
		if y.Data[i] != want[i] {
			t.Errorf("ReLU(%v) = %v, want %v", x.Data[i], y.Data[i], want[i])
		}
	}
}

func TestGELUExactFormula(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1.0}, []int{1})
	y, _ := GELU().Forward(&Context{}, x)
	want := float32(0.5 * 1.0 * (1 + math.Erf(1.0/math.Sqrt2)))
	if diff := math.Abs(float64(y.Data[0] - want)); diff > 1e-5 {
		t.Fatalf("GELU(1.0) = %v, want %v", y.Data[0], want)
	}
}

func TestActivationGradients(t *testing.T) {
	x, _ := NewTensorFromData([]float32{-1.5, -0.3, 0.4, 2.1}, []int{4})
	ctx := &Context{Mode: Inference}
	for _, tc := range []struct {
		name string
		m    Module
	}{
		{"relu", ReLU()},
		{"sigmoid", Sigmoid()},
		{"tanh", Tanh()},
		{"leaky_relu", LeakyReLU(0.01)},
		{"gelu", GELU()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			xc := x.Clone()
			checkInputGradient(t, tc.m, ctx, xc)
		})
	}
}

func TestSoftmaxRowsSumToOne(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 0, 0, 0}, []int{2, 3})
	y, err := Softmax().Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for b := 0; b < 2; b++ {
		var sum float32
		for c := 0; c < 3; c++ {
			sum += y.Data[b*3+c]
		}
		if diff := sum - 1; diff > 1e-5 || diff < -1e-5 {
			t.Errorf("row %d sums to %v, want 1", b, sum)
		}
	}
}

func TestSoftmaxGradient(t *testing.T) {
	x, _ := NewTensorFromData([]float32{0.2, 1.5, -0.3, 2.0, 0.1, -1.0}, []int{2, 3})
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, Softmax(), ctx, x)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run "TestReLU|TestGELU|TestActivationGradients|TestSoftmax" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import (
	"fmt"
	"math"
)

type activationFn struct {
	apply func(float32) float32
	deriv func(float32) float32 // derivative w.r.t. the pre-activation input x
}

// ActivationModule applies an elementwise activation and its exact
// derivative. deriv is always evaluated at the cached pre-activation input,
// never the output — required for GELU, applied uniformly for consistency.
type ActivationModule struct {
	fn    activationFn
	input *Tensor
}

func newActivation(fn activationFn) *ActivationModule {
	return &ActivationModule{fn: fn}
}

func ReLU() *ActivationModule {
	return newActivation(activationFn{
		apply: func(x float32) float32 {
			if x > 0 {
				return x
			}
			return 0
		},
		deriv: func(x float32) float32 {
			if x > 0 {
				return 1
			}
			return 0
		},
	})
}

func Sigmoid() *ActivationModule {
	sig := func(x float32) float32 { return float32(1 / (1 + math.Exp(float64(-x)))) }
	return newActivation(activationFn{
		apply: sig,
		deriv: func(x float32) float32 { s := sig(x); return s * (1 - s) },
	})
}

func Tanh() *ActivationModule {
	return newActivation(activationFn{
		apply: func(x float32) float32 { return float32(math.Tanh(float64(x))) },
		deriv: func(x float32) float32 {
			t := float32(math.Tanh(float64(x)))
			return 1 - t*t
		},
	})
}

func LeakyReLU(alpha float32) *ActivationModule {
	return newActivation(activationFn{
		apply: func(x float32) float32 {
			if x > 0 {
				return x
			}
			return alpha * x
		},
		deriv: func(x float32) float32 {
			if x > 0 {
				return 1
			}
			return alpha
		},
	})
}

// GELU uses the exact formula 0.5*x*(1+erf(x/sqrt(2))), not the SiLU
// approximation the old Network/nnx.go mislabeled as GELU.
func GELU() *ActivationModule {
	return newActivation(activationFn{
		apply: func(x float32) float32 {
			return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
		},
		deriv: func(x float32) float32 {
			cdf := 0.5 * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
			pdf := float32(math.Exp(-float64(x)*float64(x)/2)) / float32(math.Sqrt(2*math.Pi))
			return cdf + x*pdf
		},
	})
}

func (a *ActivationModule) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	a.input = x
	out := NewTensor(x.Shape)
	for i, v := range x.Data {
		out.Data[i] = a.fn.apply(v)
	}
	return out, nil
}

func (a *ActivationModule) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(a.input.Shape)
	for i, v := range a.input.Data {
		gradIn.Data[i] = gradOut.Data[i] * a.fn.deriv(v)
	}
	return gradIn, nil
}

func (a *ActivationModule) Params() []*Param { return nil }

func (a *ActivationModule) OutputShape(inShape []int) ([]int, error) { return inShape, nil }

// SoftmaxModule normalizes each row of a [batch, classes] tensor.
type SoftmaxModule struct {
	output *Tensor
}

func Softmax() *SoftmaxModule { return &SoftmaxModule{} }

func (s *SoftmaxModule) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 2 {
		return nil, fmt.Errorf("nn: Softmax expects input shape [batch, classes], got %v", x.Shape)
	}
	batch, classes := x.Shape[0], x.Shape[1]
	out := NewTensor(x.Shape)
	for b := 0; b < batch; b++ {
		maxV := x.Data[b*classes]
		for c := 1; c < classes; c++ {
			if v := x.Data[b*classes+c]; v > maxV {
				maxV = v
			}
		}
		var sum float32
		for c := 0; c < classes; c++ {
			e := float32(math.Exp(float64(x.Data[b*classes+c] - maxV)))
			out.Data[b*classes+c] = e
			sum += e
		}
		for c := 0; c < classes; c++ {
			out.Data[b*classes+c] /= sum
		}
	}
	s.output = out
	return out, nil
}

func (s *SoftmaxModule) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, classes := s.output.Shape[0], s.output.Shape[1]
	gradIn := NewTensor(s.output.Shape)
	for b := 0; b < batch; b++ {
		var dot float32
		for c := 0; c < classes; c++ {
			dot += gradOut.Data[b*classes+c] * s.output.Data[b*classes+c]
		}
		for c := 0; c < classes; c++ {
			y := s.output.Data[b*classes+c]
			gradIn.Data[b*classes+c] = y * (gradOut.Data[b*classes+c] - dot)
		}
	}
	return gradIn, nil
}

func (s *SoftmaxModule) Params() []*Param { return nil }

func (s *SoftmaxModule) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 2 {
		return nil, fmt.Errorf("nn: Softmax expects input shape [batch, classes], got %v", inShape)
	}
	return inShape, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run "TestReLU|TestGELU|TestActivationGradients|TestSoftmax" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn/activation.go nn/activation_test.go
git commit -m "feat(nn): add activations with real Softmax and exact GELU"
```

### Task 6: Sequential

**Files:**
- Create: `nn/sequential.go`
- Test: `nn/sequential_test.go`

**Interfaces:**
- Consumes: `nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.Softmax` (Tasks 2, 4, 5).
- Produces: `func Sequential(inputShape []int, modules ...Module) (*SequentialModel, error)` — `*SequentialModel` satisfies `nn.Module` and additionally exposes `func (s *SequentialModel) Modules() []Module`.

- [ ] **Step 1: Write the failing tests**

```go
package nn

import "testing"

func TestSequentialValidChain(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{4, 3},
		Linear(rng, 0, 5, XavierInit()),
		ReLU(),
		Linear(rng, 0, 2, XavierInit()),
		Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{4, 3})
	y, err := model.Forward(&Context{Mode: Inference}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if y.Shape[0] != 4 || y.Shape[1] != 2 {
		t.Fatalf("output shape = %v, want [4 2]", y.Shape)
	}
}

func TestSequentialRejectsMismatchedChain(t *testing.T) {
	rng := NewRNG(1)
	_, err := Sequential([]int{4, 3},
		Linear(rng, 5, 5, XavierInit()), // configured for 5 input features, gets 3
		ReLU(),
	)
	if err == nil {
		t.Fatal("expected error for mismatched Linear input size, got nil")
	}
}

func TestSequentialBackwardMatchesParamCount(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 3}, Linear(rng, 0, 4, XavierInit()), ReLU())
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	ctx := &Context{Mode: Inference}
	x := NewTensor([]int{2, 3})
	y, _ := model.Forward(ctx, x)
	gradOut := NewTensor(y.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradIn, err := model.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if gradIn.Shape[0] != 2 || gradIn.Shape[1] != 3 {
		t.Fatalf("input gradient shape = %v, want [2 3]", gradIn.Shape)
	}
	if len(model.Params()) != 2 { // W, B of the one Linear
		t.Fatalf("Params() len = %d, want 2", len(model.Params()))
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run TestSequential -v`
Expected: FAIL — `Sequential` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import "fmt"

type SequentialModel struct {
	modules []Module
}

// Sequential validates the whole chain via OutputShape starting from
// inputShape, returning an error naming the offending module index — no
// runtime shape surprises. This is also where lazily-built modules
// (e.g. Linear(rng, 0, ...)) learn their real input size and allocate
// their Params.
func Sequential(inputShape []int, modules ...Module) (*SequentialModel, error) {
	shape := inputShape
	for i, m := range modules {
		out, err := m.OutputShape(shape)
		if err != nil {
			return nil, fmt.Errorf("nn: Sequential module %d: %w", i, err)
		}
		shape = out
	}
	return &SequentialModel{modules: modules}, nil
}

func (s *SequentialModel) Modules() []Module { return s.modules }

func (s *SequentialModel) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	out := x
	for i, m := range s.modules {
		var err error
		out, err = m.Forward(ctx, out)
		if err != nil {
			return nil, fmt.Errorf("nn: Sequential module %d: %w", i, err)
		}
	}
	return out, nil
}

func (s *SequentialModel) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	grad := gradOut
	for i := len(s.modules) - 1; i >= 0; i-- {
		var err error
		grad, err = s.modules[i].Backward(ctx, grad)
		if err != nil {
			return nil, fmt.Errorf("nn: Sequential module %d: %w", i, err)
		}
	}
	return grad, nil
}

func (s *SequentialModel) Params() []*Param {
	var params []*Param
	for _, m := range s.modules {
		params = append(params, m.Params()...)
	}
	return params
}

func (s *SequentialModel) OutputShape(inShape []int) ([]int, error) {
	shape := inShape
	for i, m := range s.modules {
		out, err := m.OutputShape(shape)
		if err != nil {
			return nil, fmt.Errorf("nn: Sequential module %d: %w", i, err)
		}
		shape = out
	}
	return shape, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run TestSequential -v`
Expected: PASS

- [ ] **Step 5: Run the full nn package test suite so far**

Run: `go test ./nn/... -v`
Expected: PASS (all tests from Tasks 1–6)

- [ ] **Step 6: Commit**

```bash
git add nn/sequential.go nn/sequential_test.go
git commit -m "feat(nn): add Sequential with build-time shape validation"
```

---

## Phase 2: train core

Losses (incl. fused softmax+CCE), Metrics, SGD/Momentum/Adam/RMSprop + ClipNorm, Trainer (shuffle, batching, Predict, Evaluate), callbacks, schedulers as callbacks. XOR/wine end-to-end tests green.

Continued in the following tasks (7–13). See design decision #3 above for the exact fused-softmax contract Tasks 7 and 13 implement together.

### Task 7: Losses

**Files:**
- Create: `train/loss.go`
- Test: `train/loss_test.go`

**Interfaces:**
- Consumes: `nn.Tensor`, `nn.NewTensor`, `nn.NewTensorFromData` (Task 1).
- Produces: `type Loss interface { Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) }`, `func MSELoss() *MeanSquaredError`, `func BCELoss() *BinaryCrossEntropy`, `func CrossEntropy() *CrossEntropyLoss` with `func (l *CrossEntropyLoss) SetFused(fused bool)`, `func MAELoss() *MeanAbsoluteError`.

- [ ] **Step 1: Write the failing tests**

```go
package train

import (
	"math"
	"neugo/nn"
	"testing"
)

func TestMSELossValueAndGradient(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	target, _ := nn.NewTensorFromData([]float32{1, 1, 3, 6}, []int{2, 2})
	loss, grad, err := MSELoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// (0^2 + 1^2 + 0^2 + 2^2) / 4 = 5/4
	if diff := math.Abs(float64(loss - 1.25)); diff > 1e-5 {
		t.Fatalf("loss = %v, want 1.25", loss)
	}
	// d/dp mean((p-t)^2) = 2*(p-t)/N
	want := []float32{0, 0.5, 0, -1}
	for i := range want {
		if diff := math.Abs(float64(grad.Data[i] - want[i])); diff > 1e-5 {
			t.Errorf("grad[%d] = %v, want %v", i, grad.Data[i], want[i])
		}
	}
}

func TestBCELossClipsAndMatchesFiniteDifference(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.9, 0.1}, []int{2, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0}, []int{2, 1})
	_, grad, err := BCELoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	const eps = 1e-3
	for i := range pred.Data {
		p2 := pred.Clone()
		p2.Data[i] += eps
		lp, _, _ := BCELoss().Loss(p2, target)
		p3 := pred.Clone()
		p3.Data[i] -= eps
		lm, _, _ := BCELoss().Loss(p3, target)
		numGrad := (lp - lm) / (2 * eps)
		if diff := math.Abs(float64(numGrad - grad.Data[i])); diff > 1e-2 {
			t.Errorf("grad[%d] = %v, numeric = %v", i, grad.Data[i], numGrad)
		}
	}
}

func TestCrossEntropyNonFusedComposesSoftmaxJacobian(t *testing.T) {
	logits, _ := nn.NewTensorFromData([]float32{2, 1, 0.1}, []int{1, 3})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 0}, []int{1, 3})
	ce := CrossEntropy() // fused defaults to false
	_, grad, err := ce.Loss(logits, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// gradient must equal softmax(logits)-target, composed internally
	sm := nn.Softmax()
	probs, _ := sm.Forward(&nn.Context{}, logits)
	for i := range probs.Data {
		want := probs.Data[i] - target.Data[i]
		if diff := math.Abs(float64(grad.Data[i] - want)); diff > 1e-5 {
			t.Errorf("grad[%d] = %v, want %v", i, grad.Data[i], want)
		}
	}
}

func TestCrossEntropyFusedUsesPredDirectly(t *testing.T) {
	probs, _ := nn.NewTensorFromData([]float32{0.7, 0.2, 0.1}, []int{1, 3})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 0}, []int{1, 3})
	ce := CrossEntropy()
	ce.SetFused(true)
	_, grad, err := ce.Loss(probs, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	for i := range probs.Data {
		want := probs.Data[i] - target.Data[i]
		if diff := math.Abs(float64(grad.Data[i] - want)); diff > 1e-5 {
			t.Errorf("grad[%d] = %v, want %v", i, grad.Data[i], want)
		}
	}
}

func TestMAELossGradientSign(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{5, 1}, []int{2, 1})
	target, _ := nn.NewTensorFromData([]float32{3, 4}, []int{2, 1})
	_, grad, err := MAELoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if grad.Data[0] <= 0 {
		t.Errorf("grad[0] = %v, want > 0 (pred > target)", grad.Data[0])
	}
	if grad.Data[1] >= 0 {
		t.Errorf("grad[1] = %v, want < 0 (pred < target)", grad.Data[1])
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./train/... -run "TestMSELoss|TestBCELoss|TestCrossEntropy|TestMAELoss" -v`
Expected: FAIL — `train` package / symbols undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package train

import (
	"fmt"
	"math"
	"neugo/nn"
)

// Loss returns scalar loss (batch-mean) and dLoss/dPred.
type Loss interface {
	Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error)
}

const lossEpsilon = 1e-7

func clip32(p float32) float32 {
	if p < lossEpsilon {
		return lossEpsilon
	}
	if p > 1-lossEpsilon {
		return 1 - lossEpsilon
	}
	return p
}

func sameShape(a, b *nn.Tensor) error {
	if len(a.Shape) != len(b.Shape) {
		return fmt.Errorf("train: shape mismatch %v vs %v", a.Shape, b.Shape)
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return fmt.Errorf("train: shape mismatch %v vs %v", a.Shape, b.Shape)
		}
	}
	return nil
}

type MeanSquaredError struct{}

func MSELoss() *MeanSquaredError { return &MeanSquaredError{} }

func (l *MeanSquaredError) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(len(pred.Data))
	var sum float32
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		diff := pred.Data[i] - target.Data[i]
		sum += diff * diff
		grad.Data[i] = 2 * diff / n
	}
	return sum / n, grad, nil
}

type BinaryCrossEntropy struct{}

func BCELoss() *BinaryCrossEntropy { return &BinaryCrossEntropy{} }

func (l *BinaryCrossEntropy) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(len(pred.Data))
	var sum float64
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		p := clip32(pred.Data[i])
		y := target.Data[i]
		sum += -(float64(y)*math.Log(float64(p)) + float64(1-y)*math.Log(float64(1-p)))
		grad.Data[i] = (-(y/p) + (1-y)/(1-p)) / n
	}
	return float32(sum) / n, grad, nil
}

// CrossEntropyLoss computes categorical cross-entropy. When fused is true,
// pred is assumed to already be softmax probabilities (the model's last
// module is a *nn.SoftmaxModule, detected and set by train.New) and the
// gradient (probs-target)/batch is meant to bypass that module's own
// Backward. When fused is false, pred is raw logits and CrossEntropyLoss
// applies softmax internally before computing the identical gradient
// formula — see design decision #3 in the plan header.
type CrossEntropyLoss struct {
	fused bool
}

func CrossEntropy() *CrossEntropyLoss { return &CrossEntropyLoss{} }

func (l *CrossEntropyLoss) SetFused(fused bool) { l.fused = fused }
func (l *CrossEntropyLoss) Fused() bool         { return l.fused }

func (l *CrossEntropyLoss) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	if len(pred.Shape) != 2 {
		return 0, nil, fmt.Errorf("train: CrossEntropy expects [batch, classes], got %v", pred.Shape)
	}
	batch, classes := pred.Shape[0], pred.Shape[1]

	probs := pred.Data
	if !l.fused {
		sm := nn.Softmax()
		out, err := sm.Forward(&nn.Context{}, pred)
		if err != nil {
			return 0, nil, err
		}
		probs = out.Data
	}

	var sum float64
	for i, p := range probs {
		if target.Data[i] > 0 {
			sum += float64(target.Data[i]) * math.Log(float64(clip32(p)))
		}
	}
	loss := float32(-sum / float64(batch))

	grad := nn.NewTensor(pred.Shape)
	invBatch := 1.0 / float32(batch)
	for i := range probs {
		grad.Data[i] = (probs[i] - target.Data[i]) * invBatch
	}
	_ = classes
	return loss, grad, nil
}

type MeanAbsoluteError struct{}

func MAELoss() *MeanAbsoluteError { return &MeanAbsoluteError{} }

func (l *MeanAbsoluteError) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(len(pred.Data))
	var sum float32
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		diff := pred.Data[i] - target.Data[i]
		sum += float32(math.Abs(float64(diff)))
		switch {
		case diff > 0:
			grad.Data[i] = 1 / n
		case diff < 0:
			grad.Data[i] = -1 / n
		default:
			grad.Data[i] = 0
		}
	}
	return sum / n, grad, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./train/... -run "TestMSELoss|TestBCELoss|TestCrossEntropy|TestMAELoss" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train/loss.go train/loss_test.go
git commit -m "feat(train): add MSE/BCE/CrossEntropy(fused)/MAE losses"
```

### Task 8: Metrics

**Files:**
- Create: `train/metrics.go`
- Test: `train/metrics_test.go`

**Interfaces:**
- Consumes: `nn.Tensor` (Task 1).
- Produces: `type Metrics struct { Loss, Accuracy, Precision, Recall, F1Score float32; ConfusionMatrix [][]int }`, `func computeMetrics(loss float32, pred, target *nn.Tensor) (Metrics, error)` (unexported — consumed by `Trainer.Evaluate` in Task 12 and `crossval.go` in Task 18, both in-package/same-module callers).

- [ ] **Step 1: Write the failing tests**

```go
package train

import (
	"neugo/nn"
	"testing"
)

func TestComputeMetricsBinaryHandComputed(t *testing.T) {
	// 4 samples: predictions [0.9, 0.2, 0.6, 0.1], labels [1, 0, 1, 1]
	// predicted classes (>=0.5): 1,0,1,0 -> correct: samples 0,1,2 (3/4), sample 3 wrong
	pred, _ := nn.NewTensorFromData([]float32{0.9, 0.2, 0.6, 0.1}, []int{4, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 1, 1}, []int{4, 1})
	m, err := computeMetrics(0.1, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	if m.Accuracy != 75 {
		t.Errorf("Accuracy = %v, want 75", m.Accuracy)
	}
	// tp=2 (0,2), fp=0, tn=1 (1), fn=1 (3)
	wantCM := [][]int{{1, 0}, {1, 2}}
	for i := range wantCM {
		for j := range wantCM[i] {
			if m.ConfusionMatrix[i][j] != wantCM[i][j] {
				t.Errorf("ConfusionMatrix[%d][%d] = %d, want %d", i, j, m.ConfusionMatrix[i][j], wantCM[i][j])
			}
		}
	}
}

func TestComputeMetricsMulticlassHandComputed(t *testing.T) {
	// 3 samples, 3 classes, one-hot targets and argmax predictions:
	// sample0: pred class 0, actual class 0 (correct)
	// sample1: pred class 1, actual class 2 (wrong)
	// sample2: pred class 2, actual class 2 (correct)
	pred, _ := nn.NewTensorFromData([]float32{
		0.8, 0.1, 0.1,
		0.2, 0.7, 0.1,
		0.1, 0.2, 0.7,
	}, []int{3, 3})
	target, _ := nn.NewTensorFromData([]float32{
		1, 0, 0,
		0, 0, 1,
		0, 0, 1,
	}, []int{3, 3})
	m, err := computeMetrics(0.2, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	wantAcc := float32(2) / 3 * 100
	if diff := m.Accuracy - wantAcc; diff > 1e-4 || diff < -1e-4 {
		t.Errorf("Accuracy = %v, want %v", m.Accuracy, wantAcc)
	}
	if m.ConfusionMatrix[2][1] != 1 {
		t.Errorf("ConfusionMatrix[2][1] = %d, want 1 (actual=2 predicted=1)", m.ConfusionMatrix[2][1])
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./train/... -run TestComputeMetrics -v`
Expected: FAIL — `computeMetrics` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package train

import (
	"fmt"
	"neugo/nn"
)

// Metrics holds evaluation results, macro-averaged for multiclass.
type Metrics struct {
	Loss            float32
	Accuracy        float32
	Precision       float32
	Recall          float32
	F1Score         float32
	ConfusionMatrix [][]int
}

func argmaxRow(row []float32) int {
	maxIdx := 0
	maxVal := row[0]
	for i := 1; i < len(row); i++ {
		if row[i] > maxVal {
			maxVal = row[i]
			maxIdx = i
		}
	}
	return maxIdx
}

func computeMetrics(loss float32, pred, target *nn.Tensor) (Metrics, error) {
	if err := sameShape(pred, target); err != nil {
		return Metrics{}, err
	}
	if len(pred.Shape) != 2 {
		return Metrics{}, fmt.Errorf("train: computeMetrics expects [batch, classes], got %v", pred.Shape)
	}
	batch, classes := pred.Shape[0], pred.Shape[1]
	correct := 0

	if classes == 1 {
		var tp, fp, tn, fn int
		for b := 0; b < batch; b++ {
			predictedClass := 0
			if pred.Data[b] >= 0.5 {
				predictedClass = 1
			}
			actualClass := 0
			if target.Data[b] >= 0.5 {
				actualClass = 1
			}
			if predictedClass == actualClass {
				correct++
			}
			switch {
			case actualClass == 1 && predictedClass == 1:
				tp++
			case actualClass == 0 && predictedClass == 1:
				fp++
			case actualClass == 0 && predictedClass == 0:
				tn++
			case actualClass == 1 && predictedClass == 0:
				fn++
			}
		}
		var precision, recall, f1 float32
		if tp+fp > 0 {
			precision = float32(tp) / float32(tp+fp)
		}
		if tp+fn > 0 {
			recall = float32(tp) / float32(tp+fn)
		}
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}
		return Metrics{
			Loss:            loss,
			Accuracy:        float32(correct) / float32(batch) * 100,
			Precision:       precision,
			Recall:          recall,
			F1Score:         f1,
			ConfusionMatrix: [][]int{{tn, fp}, {fn, tp}},
		}, nil
	}

	confusion := make([][]int, classes)
	for i := range confusion {
		confusion[i] = make([]int, classes)
	}
	for b := 0; b < batch; b++ {
		predClass := argmaxRow(pred.Data[b*classes : (b+1)*classes])
		actualClass := argmaxRow(target.Data[b*classes : (b+1)*classes])
		if predClass == actualClass {
			correct++
		}
		confusion[actualClass][predClass]++
	}

	var totalPrecision, totalRecall float32
	numClasses := 0
	for c := 0; c < classes; c++ {
		tp := confusion[c][c]
		var fp, fn int
		for i := 0; i < classes; i++ {
			if i != c {
				fp += confusion[i][c]
				fn += confusion[c][i]
			}
		}
		if tp+fp > 0 {
			totalPrecision += float32(tp) / float32(tp+fp)
			numClasses++
		}
		if tp+fn > 0 {
			totalRecall += float32(tp) / float32(tp+fn)
		}
	}
	var precision, recall, f1 float32
	if numClasses > 0 {
		precision = totalPrecision / float32(numClasses)
		recall = totalRecall / float32(numClasses)
	}
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}
	return Metrics{
		Loss:            loss,
		Accuracy:        float32(correct) / float32(batch) * 100,
		Precision:       precision,
		Recall:          recall,
		F1Score:         f1,
		ConfusionMatrix: confusion,
	}, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./train/... -run TestComputeMetrics -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train/metrics.go train/metrics_test.go
git commit -m "feat(train): add binary/multiclass Metrics with macro-averaging"
```

### Task 9: Optimizers

**Files:**
- Create: `train/optimizer.go`
- Test: `train/optimizer_test.go`

**Interfaces:**
- Consumes: `nn.Param` (Task 2).
- Produces: `type Optimizer interface { Step(params []*nn.Param); SetLR(lr float32); GetLR() float32 }` (the `SetLR`/`GetLR` pair exists so Task 11's schedulers can adjust the learning rate of whichever optimizer the `Trainer` holds, without a type switch), `func SGD(lr float32) *SGDOptimizer`, `func Momentum(lr, beta float32) *MomentumOptimizer`, `func Adam(lr, beta1, beta2, eps float32) *AdamOptimizer`, `func RMSprop(lr, rho, eps float32) *RMSpropOptimizer`, `func ClipNorm(inner Optimizer, maxNorm float32) *ClipNormOptimizer`.

- [ ] **Step 1: Write the failing tests**

```go
package train

import (
	"math"
	"neugo/nn"
	"testing"
)

func newTestParam(value, grad []float32) *nn.Param {
	v, _ := nn.NewTensorFromData(append([]float32(nil), value...), []int{len(value)})
	p := nn.NewParam(v)
	copy(p.Grad.Data, grad)
	return p
}

func TestSGDStep(t *testing.T) {
	p := newTestParam([]float32{1, 2}, []float32{0.5, -1})
	SGD(0.1).Step([]*nn.Param{p})
	want := []float32{1 - 0.1*0.5, 2 - 0.1*-1}
	for i := range want {
		if diff := math.Abs(float64(p.Value.Data[i] - want[i])); diff > 1e-6 {
			t.Errorf("Value[%d] = %v, want %v", i, p.Value.Data[i], want[i])
		}
	}
}

func TestMomentumAccumulatesVelocity(t *testing.T) {
	p := newTestParam([]float32{0}, []float32{1})
	opt := Momentum(0.1, 0.9)
	opt.Step([]*nn.Param{p}) // v = 0.9*0 + 0.1*1 = 0.1; value = 0 - 0.1 = -0.1
	if diff := math.Abs(float64(p.Value.Data[0] - -0.1)); diff > 1e-6 {
		t.Fatalf("after step 1, Value = %v, want -0.1", p.Value.Data[0])
	}
	p.Grad.Data[0] = 1
	opt.Step([]*nn.Param{p}) // v = 0.9*0.1 + 0.1*1 = 0.19; value = -0.1 - 0.19 = -0.29
	if diff := math.Abs(float64(p.Value.Data[0] - -0.29)); diff > 1e-6 {
		t.Fatalf("after step 2, Value = %v, want -0.29", p.Value.Data[0])
	}
}

func TestAdamFirstStepBiasCorrection(t *testing.T) {
	p := newTestParam([]float32{0}, []float32{1})
	opt := Adam(0.001, 0.9, 0.999, 1e-8)
	opt.Step([]*nn.Param{p})
	// m=0.1, v=0.001, mHat=1, vHat=1 -> update = lr*1/(1+eps) ~= lr
	want := float32(-0.001)
	if diff := math.Abs(float64(p.Value.Data[0] - want)); diff > 1e-5 {
		t.Fatalf("Value after 1 Adam step = %v, want ~%v", p.Value.Data[0], want)
	}
}

func TestAdamDecreasesQuadraticLoss(t *testing.T) {
	p := newTestParam([]float32{10}, []float32{0})
	opt := Adam(0.5, 0.9, 0.999, 1e-8)
	for i := 0; i < 50; i++ {
		p.Grad.Data[0] = 2 * p.Value.Data[0] // d/dx x^2
		opt.Step([]*nn.Param{p})
	}
	if math.Abs(float64(p.Value.Data[0])) > 1 {
		t.Fatalf("Adam did not converge x toward 0: got %v", p.Value.Data[0])
	}
}

func TestRMSpropDecreasesQuadraticLoss(t *testing.T) {
	p := newTestParam([]float32{10}, []float32{0})
	opt := RMSprop(0.1, 0.9, 1e-8)
	for i := 0; i < 200; i++ {
		p.Grad.Data[0] = 2 * p.Value.Data[0]
		opt.Step([]*nn.Param{p})
	}
	if math.Abs(float64(p.Value.Data[0])) > 1 {
		t.Fatalf("RMSprop did not converge x toward 0: got %v", p.Value.Data[0])
	}
}

func TestClipNormRescalesLargeGradients(t *testing.T) {
	p := newTestParam([]float32{0, 0}, []float32{3, 4}) // norm = 5
	clipped := ClipNorm(SGD(1.0), 1.0)
	clipped.Step([]*nn.Param{p})
	// after clipping to norm 1: grad ~= [0.6, 0.8]; SGD lr=1 -> value = -grad
	if diff := math.Abs(float64(p.Value.Data[0] - -0.6)); diff > 1e-4 {
		t.Errorf("Value[0] = %v, want -0.6", p.Value.Data[0])
	}
	if diff := math.Abs(float64(p.Value.Data[1] - -0.8)); diff > 1e-4 {
		t.Errorf("Value[1] = %v, want -0.8", p.Value.Data[1])
	}
}

func TestSetLRGetLRRoundTrip(t *testing.T) {
	for name, opt := range map[string]Optimizer{
		"sgd":      SGD(0.1),
		"momentum": Momentum(0.1, 0.9),
		"adam":     Adam(0.1, 0.9, 0.999, 1e-8),
		"rmsprop":  RMSprop(0.1, 0.9, 1e-8),
		"clipnorm": ClipNorm(SGD(0.1), 1.0),
	} {
		t.Run(name, func(t *testing.T) {
			opt.SetLR(0.42)
			if diff := math.Abs(float64(opt.GetLR() - 0.42)); diff > 1e-6 {
				t.Fatalf("GetLR() = %v after SetLR(0.42), want 0.42", opt.GetLR())
			}
		})
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./train/... -run "TestSGDStep|TestMomentum|TestAdam|TestRMSprop|TestClipNorm|TestSetLRGetLR" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
package train

import (
	"math"
	"neugo/nn"
)

// Optimizer reads params[i].Grad and mutates params[i].Value.
type Optimizer interface {
	Step(params []*nn.Param)
}

type SGDOptimizer struct {
	LR float32
}

func SGD(lr float32) *SGDOptimizer { return &SGDOptimizer{LR: lr} }

func (o *SGDOptimizer) Step(params []*nn.Param) {
	for _, p := range params {
		for i := range p.Value.Data {
			p.Value.Data[i] -= o.LR * p.Grad.Data[i]
		}
	}
}

func (o *SGDOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *SGDOptimizer) GetLR() float32   { return o.LR }

type MomentumOptimizer struct {
	LR, Beta float32
	velocity map[*nn.Param][]float32
}

func Momentum(lr, beta float32) *MomentumOptimizer {
	return &MomentumOptimizer{LR: lr, Beta: beta, velocity: map[*nn.Param][]float32{}}
}

func (o *MomentumOptimizer) Step(params []*nn.Param) {
	for _, p := range params {
		v, ok := o.velocity[p]
		if !ok {
			v = make([]float32, len(p.Value.Data))
			o.velocity[p] = v
		}
		for i := range p.Value.Data {
			v[i] = o.Beta*v[i] + o.LR*p.Grad.Data[i]
			p.Value.Data[i] -= v[i]
		}
	}
}

func (o *MomentumOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *MomentumOptimizer) GetLR() float32   { return o.LR }

type AdamOptimizer struct {
	LR, Beta1, Beta2, Eps float32
	t                     int
	m, v                  map[*nn.Param][]float32
}

func Adam(lr, beta1, beta2, eps float32) *AdamOptimizer {
	return &AdamOptimizer{
		LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps,
		m: map[*nn.Param][]float32{}, v: map[*nn.Param][]float32{},
	}
}

func (o *AdamOptimizer) Step(params []*nn.Param) {
	o.t++
	b1t := float32(math.Pow(float64(o.Beta1), float64(o.t)))
	b2t := float32(math.Pow(float64(o.Beta2), float64(o.t)))
	for _, p := range params {
		m, ok := o.m[p]
		if !ok {
			m = make([]float32, len(p.Value.Data))
			o.m[p] = m
		}
		v, ok := o.v[p]
		if !ok {
			v = make([]float32, len(p.Value.Data))
			o.v[p] = v
		}
		for i := range p.Value.Data {
			g := p.Grad.Data[i]
			m[i] = o.Beta1*m[i] + (1-o.Beta1)*g
			v[i] = o.Beta2*v[i] + (1-o.Beta2)*g*g
			mHat := m[i] / (1 - b1t)
			vHat := v[i] / (1 - b2t)
			p.Value.Data[i] -= o.LR * mHat / (float32(math.Sqrt(float64(vHat))) + o.Eps)
		}
	}
}

func (o *AdamOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *AdamOptimizer) GetLR() float32   { return o.LR }

type RMSpropOptimizer struct {
	LR, Rho, Eps float32
	sq           map[*nn.Param][]float32
}

func RMSprop(lr, rho, eps float32) *RMSpropOptimizer {
	return &RMSpropOptimizer{LR: lr, Rho: rho, Eps: eps, sq: map[*nn.Param][]float32{}}
}

func (o *RMSpropOptimizer) Step(params []*nn.Param) {
	for _, p := range params {
		sq, ok := o.sq[p]
		if !ok {
			sq = make([]float32, len(p.Value.Data))
			o.sq[p] = sq
		}
		for i := range p.Value.Data {
			g := p.Grad.Data[i]
			sq[i] = o.Rho*sq[i] + (1-o.Rho)*g*g
			p.Value.Data[i] -= o.LR * g / (float32(math.Sqrt(float64(sq[i]))) + o.Eps)
		}
	}
}

func (o *RMSpropOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *RMSpropOptimizer) GetLR() float32   { return o.LR }

// ClipNormOptimizer rescales gradients by their global L2 norm before
// delegating to the wrapped Optimizer, if that norm exceeds maxNorm.
type ClipNormOptimizer struct {
	inner   Optimizer
	maxNorm float32
}

func ClipNorm(inner Optimizer, maxNorm float32) *ClipNormOptimizer {
	return &ClipNormOptimizer{inner: inner, maxNorm: maxNorm}
}

func (o *ClipNormOptimizer) Step(params []*nn.Param) {
	var sumSq float64
	for _, p := range params {
		for _, g := range p.Grad.Data {
			sumSq += float64(g) * float64(g)
		}
	}
	norm := float32(math.Sqrt(sumSq))
	if norm > o.maxNorm {
		scale := o.maxNorm / norm
		for _, p := range params {
			for i := range p.Grad.Data {
				p.Grad.Data[i] *= scale
			}
		}
	}
	o.inner.Step(params)
}

func (o *ClipNormOptimizer) SetLR(lr float32) { o.inner.SetLR(lr) }
func (o *ClipNormOptimizer) GetLR() float32   { return o.inner.GetLR() }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./train/... -run "TestSGDStep|TestMomentum|TestAdam|TestRMSprop|TestClipNorm|TestSetLRGetLR" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train/optimizer.go train/optimizer_test.go
git commit -m "feat(train): add SGD/Momentum/Adam/RMSprop optimizers and ClipNorm"
```

### Task 10: Callbacks

**Files:**
- Create: `train/callback.go`
- Test: `train/callback_test.go`

**Interfaces:**
- Consumes: `nn.Param` (Task 2), `train.Metrics` (Task 8).
- Produces: `type Callback interface { OnTrainBegin(); OnTrainEnd(); OnEpochBegin(epoch int); OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param); OnBatchEnd(batch int, loss float32) }`, `type BaseCallback struct{}` (no-op defaults, embed to implement a subset), `func NewHistory() *History`, `func EarlyStopping(patience int) *EarlyStoppingCallback` with `func (es *EarlyStoppingCallback) RestoreBestWeights(params []*nn.Param)`, `func ModelCheckpoint(filepath, monitor, mode string, saveBestOnly bool) *ModelCheckpointCallback` (its `Save func(path string) error` field is wired by `Trainer.Fit` in Task 12, not set here), `func ProgressBar(totalEpochs, printEvery int) *ProgressBarCallback`, `func NewCallbackList(cbs ...Callback) *CallbackList`.

Note: the interface intentionally does **not** take a `*Trainer` — that would create a file-ordering dependency on Task 12 and let the old bug back in (a callback holding a reference to shared, mutable trainer state). Every hook receives exactly the data it needs as plain values/slices, decoupling callbacks from the trainer's internals by construction.

- [ ] **Step 1: Write the failing tests**

```go
package train

import (
	"errors"
	"neugo/nn"
	"testing"
)

func TestEarlyStoppingTriggersAfterPatience(t *testing.T) {
	es := EarlyStopping(2)
	p := newTestParam([]float32{1}, []float32{0})
	params := []*nn.Param{p}

	es.OnEpochEnd(0, 1.0, nil, params) // improves (inf -> 1.0)
	if es.ShouldStop {
		t.Fatal("ShouldStop after epoch 0, want false")
	}
	es.OnEpochEnd(1, 1.0, nil, params) // no improvement, counter=1
	if es.ShouldStop {
		t.Fatal("ShouldStop after epoch 1, want false")
	}
	es.OnEpochEnd(2, 1.0, nil, params) // no improvement, counter=2 >= patience
	if !es.ShouldStop {
		t.Fatal("ShouldStop after epoch 2, want true")
	}
}

func TestEarlyStoppingRestoresBestWeights(t *testing.T) {
	es := EarlyStopping(1)
	p := newTestParam([]float32{1}, []float32{0})
	params := []*nn.Param{p}

	es.OnEpochEnd(0, 1.0, nil, params) // best = 1.0, snapshot value=[1]
	p.Value.Data[0] = 999             // simulate further training moving weights
	es.OnEpochEnd(1, 2.0, nil, params) // worse, no new snapshot

	es.RestoreBestWeights(params)
	if p.Value.Data[0] != 1 {
		t.Fatalf("Value[0] after RestoreBestWeights = %v, want 1", p.Value.Data[0])
	}
}

func TestModelCheckpointSavesOnlyOnImprovement(t *testing.T) {
	mc := ModelCheckpoint("model.json", "loss", "min", true)
	saves := 0
	mc.Save = func(path string) error { saves++; return nil }

	mc.OnEpochEnd(0, 0, &Metrics{Loss: 1.0}, nil) // improves -> save
	mc.OnEpochEnd(1, 0, &Metrics{Loss: 1.5}, nil) // worse -> no save
	mc.OnEpochEnd(2, 0, &Metrics{Loss: 0.5}, nil) // improves -> save

	if saves != 2 {
		t.Fatalf("saves = %d, want 2", saves)
	}
}

func TestModelCheckpointRecordsSaveError(t *testing.T) {
	mc := ModelCheckpoint("model.json", "loss", "min", true)
	wantErr := errors.New("disk full")
	mc.Save = func(path string) error { return wantErr }
	mc.OnEpochEnd(0, 0, &Metrics{Loss: 1.0}, nil)
	if mc.LastError != wantErr {
		t.Fatalf("LastError = %v, want %v", mc.LastError, wantErr)
	}
}

func TestHistoryRecordsLossesInOrder(t *testing.T) {
	h := NewHistory()
	h.OnEpochEnd(0, 0.9, &Metrics{Loss: 0.8, Accuracy: 50}, nil)
	h.OnEpochEnd(1, 0.5, &Metrics{Loss: 0.4, Accuracy: 70}, nil)
	wantTrain := []float32{0.9, 0.5}
	wantVal := []float32{0.8, 0.4}
	for i := range wantTrain {
		if h.TrainLoss[i] != wantTrain[i] {
			t.Errorf("TrainLoss[%d] = %v, want %v", i, h.TrainLoss[i], wantTrain[i])
		}
		if h.ValLoss[i] != wantVal[i] {
			t.Errorf("ValLoss[%d] = %v, want %v", i, h.ValLoss[i], wantVal[i])
		}
	}
}

func TestCallbackListFanOut(t *testing.T) {
	h1, h2 := NewHistory(), NewHistory()
	cl := NewCallbackList(h1, h2)
	cl.OnEpochEnd(0, 1.0, nil, nil)
	if len(h1.TrainLoss) != 1 || len(h2.TrainLoss) != 1 {
		t.Fatalf("expected both callbacks to observe the epoch end, got h1=%v h2=%v", h1.TrainLoss, h2.TrainLoss)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./train/... -run "TestEarlyStopping|TestModelCheckpoint|TestHistory|TestCallbackList" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
package train

import (
	"math"
	"neugo/nn"
	"time"
)

// Callback observes training events. Hooks receive plain data (never a
// *Trainer reference) so callbacks can't accidentally hold onto shared,
// mutable trainer state across Fit calls — see Task 10 note in the plan.
type Callback interface {
	OnTrainBegin()
	OnTrainEnd()
	OnEpochBegin(epoch int)
	OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param)
	OnBatchEnd(batch int, loss float32)
}

// BaseCallback provides no-op defaults; embed it to implement only the
// hooks you need.
type BaseCallback struct{}

func (BaseCallback) OnTrainBegin()                                                              {}
func (BaseCallback) OnTrainEnd()                                                                {}
func (BaseCallback) OnEpochBegin(epoch int)                                                     {}
func (BaseCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {}
func (BaseCallback) OnBatchEnd(batch int, loss float32)                                         {}

// History accumulates per-epoch losses/metrics. Trainer.Fit constructs and
// returns a fresh *History on every call — it is never something a caller
// hands into train.Callbacks(...) and never stored on the Trainer itself,
// which is what fixes the old "History accumulates across Train calls" bug
// by construction rather than by caller discipline.
type History struct {
	BaseCallback
	TrainLoss           []float32
	ValLoss, ValAcc, ValF1 []float32
	StartTime, EndTime   time.Time
}

func NewHistory() *History { return &History{} }

func (h *History) OnTrainBegin() { h.StartTime = time.Now() }
func (h *History) OnTrainEnd()   { h.EndTime = time.Now() }

func (h *History) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	h.TrainLoss = append(h.TrainLoss, trainLoss)
	if valMetrics != nil {
		h.ValLoss = append(h.ValLoss, valMetrics.Loss)
		h.ValAcc = append(h.ValAcc, valMetrics.Accuracy)
		h.ValF1 = append(h.ValF1, valMetrics.F1Score)
	}
}

func (h *History) Duration() time.Duration { return h.EndTime.Sub(h.StartTime) }

// EarlyStoppingCallback stops training after Patience epochs without a
// Loss (train loss, or validation loss when validation data is supplied)
// improvement of at least MinDelta, and can restore the best in-memory
// weight snapshot afterward.
type EarlyStoppingCallback struct {
	BaseCallback
	Patience   int
	MinDelta   float32
	ShouldStop bool

	bestLoss   float32
	counter    int
	bestParams [][]float32
}

func EarlyStopping(patience int) *EarlyStoppingCallback {
	return &EarlyStoppingCallback{Patience: patience, bestLoss: float32(math.Inf(1))}
}

func (es *EarlyStoppingCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	monitor := trainLoss
	if valMetrics != nil {
		monitor = valMetrics.Loss
	}
	if monitor < es.bestLoss-es.MinDelta {
		es.bestLoss = monitor
		es.counter = 0
		es.bestParams = snapshotParamValues(params)
	} else {
		es.counter++
		if es.counter >= es.Patience {
			es.ShouldStop = true
		}
	}
}

func (es *EarlyStoppingCallback) RestoreBestWeights(params []*nn.Param) {
	if es.bestParams == nil {
		return
	}
	for i, p := range params {
		copy(p.Value.Data, es.bestParams[i])
	}
}

func snapshotParamValues(params []*nn.Param) [][]float32 {
	snap := make([][]float32, len(params))
	for i, p := range params {
		snap[i] = append([]float32(nil), p.Value.Data...)
	}
	return snap
}

// ModelCheckpointCallback saves the model when Monitor improves (or every
// epoch, if SaveBestOnly is false). Save is left nil here — Trainer.Fit
// wires it to nn.Save(model, path) once nn.Save exists (Task 21); until
// then a ModelCheckpointCallback with a nil Save is a documented no-op.
// Failures are recorded in LastError rather than printed, keeping stdout
// output limited to ProgressBar/Summary per the Global Constraints.
type ModelCheckpointCallback struct {
	BaseCallback
	Filepath     string
	Monitor      string // "loss", "accuracy", "f1"
	Mode         string // "min" or "max"
	SaveBestOnly bool
	Save         func(path string) error
	LastError    error

	bestValue float32
}

func ModelCheckpoint(filepath, monitor, mode string, saveBestOnly bool) *ModelCheckpointCallback {
	best := float32(math.Inf(1))
	if mode == "max" {
		best = float32(math.Inf(-1))
	}
	return &ModelCheckpointCallback{Filepath: filepath, Monitor: monitor, Mode: mode, SaveBestOnly: saveBestOnly, bestValue: best}
}

func (mc *ModelCheckpointCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	if mc.Save == nil || valMetrics == nil {
		return
	}
	var current float32
	switch mc.Monitor {
	case "accuracy":
		current = valMetrics.Accuracy
	case "f1":
		current = valMetrics.F1Score
	default:
		current = valMetrics.Loss
	}
	improved := current < mc.bestValue
	if mc.Mode == "max" {
		improved = current > mc.bestValue
	}
	if improved {
		mc.bestValue = current
	}
	if improved || !mc.SaveBestOnly {
		mc.LastError = mc.Save(mc.Filepath)
	}
}

// ProgressBarCallback is one of the two callbacks permitted to write to
// stdout (the other is nn.Summary).
type ProgressBarCallback struct {
	BaseCallback
	TotalEpochs int
	PrintEvery  int
}

func ProgressBar(totalEpochs, printEvery int) *ProgressBarCallback {
	return &ProgressBarCallback{TotalEpochs: totalEpochs, PrintEvery: printEvery}
}

func (pb *ProgressBarCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	if pb.PrintEvery <= 0 || (epoch%pb.PrintEvery != 0 && epoch != pb.TotalEpochs-1) {
		return
	}
	if valMetrics != nil {
		fmt.Printf("Epoch %d/%d - loss: %.4f - val_loss: %.4f - val_acc: %.2f%%\n",
			epoch+1, pb.TotalEpochs, trainLoss, valMetrics.Loss, valMetrics.Accuracy)
	} else {
		fmt.Printf("Epoch %d/%d - loss: %.4f\n", epoch+1, pb.TotalEpochs, trainLoss)
	}
}

// CallbackList fans every hook out to each registered Callback in order.
type CallbackList struct {
	callbacks []Callback
}

func NewCallbackList(cbs ...Callback) *CallbackList { return &CallbackList{callbacks: cbs} }

func (cl *CallbackList) Add(cb Callback) { cl.callbacks = append(cl.callbacks, cb) }

func (cl *CallbackList) OnTrainBegin() {
	for _, cb := range cl.callbacks {
		cb.OnTrainBegin()
	}
}

func (cl *CallbackList) OnTrainEnd() {
	for _, cb := range cl.callbacks {
		cb.OnTrainEnd()
	}
}

func (cl *CallbackList) OnEpochBegin(epoch int) {
	for _, cb := range cl.callbacks {
		cb.OnEpochBegin(epoch)
	}
}

func (cl *CallbackList) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	for _, cb := range cl.callbacks {
		cb.OnEpochEnd(epoch, trainLoss, valMetrics, params)
	}
}

func (cl *CallbackList) OnBatchEnd(batch int, loss float32) {
	for _, cb := range cl.callbacks {
		cb.OnBatchEnd(batch, loss)
	}
}
```

Add `"fmt"` to the import block (used by `ProgressBarCallback.OnEpochEnd`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./train/... -run "TestEarlyStopping|TestModelCheckpoint|TestHistory|TestCallbackList" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train/callback.go train/callback_test.go
git commit -m "feat(train): add History/EarlyStopping/ModelCheckpoint/ProgressBar callbacks"
```

### Task 11: Schedulers as callbacks

**Files:**
- Create: `train/scheduler.go`
- Test: `train/scheduler_test.go`

**Interfaces:**
- Consumes: `train.Optimizer`, `train.Callback`, `train.BaseCallback` (Tasks 9, 10).
- Produces: `func StepDecay(opt Optimizer, decayRate float32, decaySteps int) *StepDecayScheduler`, `func ExponentialDecay(opt Optimizer, decayRate float32) *ExponentialDecayScheduler`, `func CosineAnnealing(opt Optimizer, minLR float32, maxEpochs int) *CosineAnnealingScheduler`, `func Warmup(opt Optimizer, targetLR float32, warmupSteps int) *WarmupScheduler`, `func ReduceLROnPlateau(opt Optimizer, factor float32, patience int, minLR float32, mode string) *ReduceLROnPlateauScheduler` — all satisfy `train.Callback` via embedded `BaseCallback` plus one overridden hook.

Note: schedulers take the `Optimizer` they control directly at construction (mirroring design decision #1's RNG-threading rationale — the design doc's `train.StepDecay(0.5, 10)` sample elides this the same way it elides RNG and error handling). This is what lets schedulers mutate learning rate without the `Callback` interface needing an `Optimizer`/`Trainer` reference — `Trainer.Fit` (Task 12) just needs to call `callbacks.OnEpochBegin(epoch)` before each epoch's batches and `callbacks.OnEpochEnd(...)` after, and every scheduler here already holds what it needs.

- [ ] **Step 1: Write the failing tests**

```go
package train

import (
	"math"
	"testing"
)

func TestStepDecayReducesAtBoundaries(t *testing.T) {
	opt := SGD(1.0)
	s := StepDecay(opt, 0.5, 10)
	s.OnEpochBegin(0)
	if opt.GetLR() != 1.0 {
		t.Fatalf("LR at epoch 0 = %v, want 1.0", opt.GetLR())
	}
	s.OnEpochBegin(10)
	if opt.GetLR() != 0.5 {
		t.Fatalf("LR at epoch 10 = %v, want 0.5", opt.GetLR())
	}
	s.OnEpochBegin(20)
	if opt.GetLR() != 0.25 {
		t.Fatalf("LR at epoch 20 = %v, want 0.25", opt.GetLR())
	}
}

func TestExponentialDecayFormula(t *testing.T) {
	opt := SGD(1.0)
	s := ExponentialDecay(opt, 0.9)
	s.OnEpochBegin(5)
	want := float32(math.Pow(0.9, 5))
	if diff := math.Abs(float64(opt.GetLR() - want)); diff > 1e-5 {
		t.Fatalf("LR at epoch 5 = %v, want %v", opt.GetLR(), want)
	}
}

func TestCosineAnnealingEndpoints(t *testing.T) {
	opt := SGD(1.0)
	s := CosineAnnealing(opt, 0.0, 100)
	s.OnEpochBegin(0)
	if diff := math.Abs(float64(opt.GetLR() - 1.0)); diff > 1e-4 {
		t.Fatalf("LR at epoch 0 = %v, want 1.0", opt.GetLR())
	}
	s.OnEpochBegin(100)
	if diff := math.Abs(float64(opt.GetLR() - 0.0)); diff > 1e-4 {
		t.Fatalf("LR at epoch 100 = %v, want 0.0", opt.GetLR())
	}
}

func TestWarmupLinearRamp(t *testing.T) {
	opt := SGD(0.0)
	s := Warmup(opt, 1.0, 10)
	s.OnEpochBegin(5)
	if diff := math.Abs(float64(opt.GetLR() - 0.5)); diff > 1e-4 {
		t.Fatalf("LR at epoch 5 = %v, want 0.5", opt.GetLR())
	}
	s.OnEpochBegin(10)
	if opt.GetLR() != 1.0 {
		t.Fatalf("LR at epoch 10 = %v, want 1.0", opt.GetLR())
	}
}

func TestReduceLROnPlateauReducesAfterPatience(t *testing.T) {
	opt := SGD(1.0)
	s := ReduceLROnPlateau(opt, 0.5, 2, 0.01, "min")
	s.OnEpochEnd(0, 1.0, nil, nil) // improves (inf -> 1.0)
	s.OnEpochEnd(1, 1.0, nil, nil) // no improvement, counter=1
	if opt.GetLR() != 1.0 {
		t.Fatalf("LR after 1 stagnant epoch = %v, want unchanged 1.0", opt.GetLR())
	}
	s.OnEpochEnd(2, 1.0, nil, nil) // no improvement, counter=2 >= patience -> reduce
	if opt.GetLR() != 0.5 {
		t.Fatalf("LR after patience exceeded = %v, want 0.5", opt.GetLR())
	}
}

func TestReduceLROnPlateauRespectsMinLR(t *testing.T) {
	opt := SGD(0.02)
	s := ReduceLROnPlateau(opt, 0.5, 1, 0.01, "min")
	s.OnEpochEnd(0, 1.0, nil, nil)
	s.OnEpochEnd(1, 1.0, nil, nil) // would reduce to 0.01, exactly MinLR -> stays at 0.02 (not > MinLR)
	if opt.GetLR() != 0.02 {
		t.Fatalf("LR = %v, want unchanged 0.02 (reduction would not exceed MinLR)", opt.GetLR())
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./train/... -run "TestStepDecay|TestExponentialDecay|TestCosineAnnealing|TestWarmup|TestReduceLROnPlateau" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
package train

import (
	"math"
	"neugo/nn"
)

type StepDecayScheduler struct {
	BaseCallback
	opt                  Optimizer
	initialLR, decayRate float32
	decaySteps           int
}

func StepDecay(opt Optimizer, decayRate float32, decaySteps int) *StepDecayScheduler {
	return &StepDecayScheduler{opt: opt, initialLR: opt.GetLR(), decayRate: decayRate, decaySteps: decaySteps}
}

func (s *StepDecayScheduler) OnEpochBegin(epoch int) {
	numDecays := epoch / s.decaySteps
	s.opt.SetLR(s.initialLR * float32(math.Pow(float64(s.decayRate), float64(numDecays))))
}

type ExponentialDecayScheduler struct {
	BaseCallback
	opt                  Optimizer
	initialLR, decayRate float32
}

func ExponentialDecay(opt Optimizer, decayRate float32) *ExponentialDecayScheduler {
	return &ExponentialDecayScheduler{opt: opt, initialLR: opt.GetLR(), decayRate: decayRate}
}

func (s *ExponentialDecayScheduler) OnEpochBegin(epoch int) {
	s.opt.SetLR(s.initialLR * float32(math.Pow(float64(s.decayRate), float64(epoch))))
}

type CosineAnnealingScheduler struct {
	BaseCallback
	opt                 Optimizer
	initialLR, minLR    float32
	maxEpochs           int
}

func CosineAnnealing(opt Optimizer, minLR float32, maxEpochs int) *CosineAnnealingScheduler {
	return &CosineAnnealingScheduler{opt: opt, initialLR: opt.GetLR(), minLR: minLR, maxEpochs: maxEpochs}
}

func (s *CosineAnnealingScheduler) OnEpochBegin(epoch int) {
	if epoch >= s.maxEpochs {
		s.opt.SetLR(s.minLR)
		return
	}
	progress := float64(epoch) / float64(s.maxEpochs)
	cosineDecay := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	s.opt.SetLR(s.minLR + (s.initialLR-s.minLR)*float32(cosineDecay))
}

type WarmupScheduler struct {
	BaseCallback
	opt                    Optimizer
	initialLR, targetLR    float32
	warmupSteps            int
}

func Warmup(opt Optimizer, targetLR float32, warmupSteps int) *WarmupScheduler {
	return &WarmupScheduler{opt: opt, initialLR: opt.GetLR(), targetLR: targetLR, warmupSteps: warmupSteps}
}

func (s *WarmupScheduler) OnEpochBegin(epoch int) {
	if epoch >= s.warmupSteps {
		s.opt.SetLR(s.targetLR)
		return
	}
	progress := float32(epoch) / float32(s.warmupSteps)
	s.opt.SetLR(s.initialLR + (s.targetLR-s.initialLR)*progress)
}

type ReduceLROnPlateauScheduler struct {
	BaseCallback
	opt        Optimizer
	Factor     float32
	Patience   int
	MinLR      float32
	Mode       string // "min" or "max"
	bestMetric float32
	counter    int
}

func ReduceLROnPlateau(opt Optimizer, factor float32, patience int, minLR float32, mode string) *ReduceLROnPlateauScheduler {
	best := float32(math.Inf(1))
	if mode == "max" {
		best = float32(math.Inf(-1))
	}
	return &ReduceLROnPlateauScheduler{opt: opt, Factor: factor, Patience: patience, MinLR: minLR, Mode: mode, bestMetric: best}
}

func (s *ReduceLROnPlateauScheduler) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	metric := trainLoss
	if valMetrics != nil {
		metric = valMetrics.Loss
	}
	improved := metric < s.bestMetric
	if s.Mode == "max" {
		improved = metric > s.bestMetric
	}
	if improved {
		s.bestMetric = metric
		s.counter = 0
		return
	}
	s.counter++
	if s.counter >= s.Patience {
		if newLR := s.opt.GetLR() * s.Factor; newLR > s.MinLR {
			s.opt.SetLR(newLR)
		}
		s.counter = 0
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./train/... -run "TestStepDecay|TestExponentialDecay|TestCosineAnnealing|TestWarmup|TestReduceLROnPlateau" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train/scheduler.go train/scheduler_test.go
git commit -m "feat(train): add LR schedulers as Callbacks (Step/Exponential/Cosine/Warmup/Plateau)"
```

### Task 12: Trainer

**Files:**
- Create: `train/trainer.go`
- Test: `train/trainer_test.go`

**Interfaces:**
- Consumes: `nn.SequentialModel`, `nn.Context`, `nn.Train`/`nn.Inference`, `nn.NewRNG` (Phase 1); `train.Loss`, `train.CrossEntropyLoss`, `train.Optimizer`, `train.ClipNorm`, `train.CallbackList`, `train.History`, `train.EarlyStoppingCallback`, `train.ModelCheckpointCallback`, `train.Metrics`, `computeMetrics` (Tasks 7–11).
- Produces: `func New(model *nn.SequentialModel, opt Optimizer, loss Loss) *Trainer`, `func (t *Trainer) Fit(x, y *nn.Tensor, opts ...FitOption) (*History, error)`, `func (t *Trainer) Predict(x *nn.Tensor) (*nn.Tensor, error)`, `func (t *Trainer) Evaluate(x, y *nn.Tensor) (Metrics, error)`, `FitOption`s: `Epochs(n int)`, `BatchSize(n int)`, `Shuffle(enabled bool)`, `Seed(seed int64)`, `Validation(x, y *nn.Tensor)`, `ClipGrad(maxNorm float32)`, `Callbacks(cbs ...Callback)`, `WithSaveFunc(fn func(*nn.SequentialModel, string) error)`.

This is the task where design decision #3 (fused softmax+CCE gradient routing) and the `ModelCheckpoint`/`nn.Save` decoupling from Task 10 both get wired together, and where the one true epoch loop described in §4.3 of the design doc is implemented.

- [ ] **Step 1: Write the failing tests**

```go
package train

import (
	"neugo/nn"
	"testing"
)

func TestFitRequiresEpochs(t *testing.T) {
	rng := nn.NewRNG(1)
	model, _ := nn.Sequential([]int{1, 2}, nn.Linear(rng, 2, 1, nn.XavierInit()))
	trainer := New(model, SGD(0.1), MSELoss())
	x := nn.NewTensor([]int{1, 2})
	y := nn.NewTensor([]int{1, 1})
	_, err := trainer.Fit(x, y) // no Epochs(...) option
	if err == nil {
		t.Fatal("expected error when Epochs is not set, got nil")
	}
}

func TestXORConvergesWithAdam(t *testing.T) {
	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{4, 2},
		nn.Linear(rng, 2, 8, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 8, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	hist, err := trainer.Fit(x, y, Epochs(2000), BatchSize(4), Seed(1))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	finalLoss := hist.TrainLoss[len(hist.TrainLoss)-1]
	if finalLoss >= 0.05 {
		t.Fatalf("final train loss = %v after 2000 Adam epochs, want < 0.05", finalLoss)
	}
}

func TestXORConvergesWithSGD(t *testing.T) {
	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{4, 2},
		nn.Linear(rng, 2, 8, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 8, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := New(model, SGD(0.5), BCELoss())
	hist, err := trainer.Fit(x, y, Epochs(5000), BatchSize(4), Seed(1))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	finalLoss := hist.TrainLoss[len(hist.TrainLoss)-1]
	if finalLoss >= 0.05 {
		t.Fatalf("final train loss = %v after 5000 SGD epochs, want < 0.05", finalLoss)
	}
}

func TestMulticlassConvergesWithFusedSoftmax(t *testing.T) {
	rng := nn.NewRNG(2)
	model, err := nn.Sequential([]int{6, 2},
		nn.Linear(rng, 2, 8, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 8, 3, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	// Three well-separated 2D clusters, two samples each.
	x, _ := nn.NewTensorFromData([]float32{
		-2, -2, -1.8, -2.1,
		0, 2, 0.1, 1.9,
		2, -2, 1.9, -2.1,
	}, []int{6, 2})
	y, _ := nn.NewTensorFromData([]float32{
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
	}, []int{6, 3})

	ce := CrossEntropy()
	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), ce)
	if !ce.Fused() {
		t.Fatal("New did not detect trailing Softmax and enable fused CrossEntropy")
	}
	if _, err := trainer.Fit(x, y, Epochs(500), BatchSize(6), Seed(2)); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if metrics.Accuracy < 100 {
		t.Fatalf("Accuracy = %v after training on separable clusters, want 100", metrics.Accuracy)
	}
}

func TestPredictMatchesForwardInInferenceMode(t *testing.T) {
	rng := nn.NewRNG(3)
	model, err := nn.Sequential([]int{2, 3},
		nn.Linear(rng, 3, 4, nn.XavierInit()),
		nn.ReLU(),
		Dropout(0.5), // will exist by Task 16; TestPredictMatchesForwardInInferenceMode is re-run then.
		nn.Linear(rng, 4, 1, nn.XavierInit()),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{2, 3})
	trainer := New(model, SGD(0.1), MSELoss())
	p1, err := trainer.Predict(x)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	p2, _ := trainer.Predict(x)
	for i := range p1.Data {
		if p1.Data[i] != p2.Data[i] {
			t.Fatalf("Predict is non-deterministic in inference mode at index %d: %v vs %v", i, p1.Data[i], p2.Data[i])
		}
	}
}

func TestEvaluateReturnsPopulatedMetrics(t *testing.T) {
	rng := nn.NewRNG(4)
	model, _ := nn.Sequential([]int{4, 2}, nn.Linear(rng, 2, 1, nn.XavierInit()), nn.Sigmoid())
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})
	trainer := New(model, SGD(0.1), BCELoss())
	m, err := trainer.Evaluate(x, y)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if m.ConfusionMatrix == nil {
		t.Fatal("Evaluate returned nil ConfusionMatrix")
	}
}
```

Note: `TestPredictMatchesForwardInInferenceMode` references `Dropout(0.5)`, which does not exist until Task 16 (Phase 3). Skip that one test (`t.Skip("Dropout not implemented until Task 16")` as its first line) when running this task's tests, then delete the `t.Skip` line as part of Task 16's own step 1 (Task 16 lists this file among its edits) so the test actually exercises Dropout once it exists.

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./train/... -run "TestFitRequiresEpochs|TestXORConverges|TestMulticlassConverges|TestEvaluateReturnsPopulatedMetrics" -v`
Expected: FAIL — `Trainer`/`New`/`Fit` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package train

import (
	"fmt"
	"neugo/nn"
)

type Trainer struct {
	model *nn.SequentialModel
	opt   Optimizer
	loss  Loss
}

// New builds a Trainer and, if loss is a *CrossEntropyLoss and the model's
// last module is *nn.SoftmaxModule, enables the fused softmax+CCE gradient
// shortcut (design decision #3 in the plan header).
func New(model *nn.SequentialModel, opt Optimizer, loss Loss) *Trainer {
	if ce, ok := loss.(*CrossEntropyLoss); ok {
		modules := model.Modules()
		if n := len(modules); n > 0 {
			if _, isSoftmax := modules[n-1].(*nn.SoftmaxModule); isSoftmax {
				ce.SetFused(true)
			}
		}
	}
	return &Trainer{model: model, opt: opt, loss: loss}
}

type FitConfig struct {
	epochs          int
	batchSize       int
	shuffle         bool
	seed            int64
	valX, valY      *nn.Tensor
	hasVal          bool
	clipGradMaxNorm float32
	callbacks       *CallbackList
	saveFunc        func(*nn.SequentialModel, string) error
}

type FitOption func(*FitConfig)

func Epochs(n int) FitOption          { return func(c *FitConfig) { c.epochs = n } }
func BatchSize(n int) FitOption       { return func(c *FitConfig) { c.batchSize = n } }
func Shuffle(enabled bool) FitOption  { return func(c *FitConfig) { c.shuffle = enabled } }
func Seed(seed int64) FitOption       { return func(c *FitConfig) { c.seed = seed } }
func ClipGrad(maxNorm float32) FitOption {
	return func(c *FitConfig) { c.clipGradMaxNorm = maxNorm }
}
func Callbacks(cbs ...Callback) FitOption {
	return func(c *FitConfig) { c.callbacks = NewCallbackList(cbs...) }
}
func Validation(x, y *nn.Tensor) FitOption {
	return func(c *FitConfig) { c.valX, c.valY, c.hasVal = x, y, true }
}
func WithSaveFunc(fn func(*nn.SequentialModel, string) error) FitOption {
	return func(c *FitConfig) { c.saveFunc = fn }
}

func gatherRows(t *nn.Tensor, idx []int, rowShape []int) *nn.Tensor {
	rowSize := 1
	for _, d := range rowShape {
		rowSize *= d
	}
	out := nn.NewTensor(append([]int{len(idx)}, rowShape...))
	for i, src := range idx {
		copy(out.Data[i*rowSize:(i+1)*rowSize], t.Data[src*rowSize:(src+1)*rowSize])
	}
	return out
}

// Fit runs the one training loop in this codebase: per-epoch shuffle,
// batched forward/loss/backward/optimizer-step, then validation metrics,
// scheduler and early-stopping callbacks, then history — see §4.3 of the
// design doc.
func (t *Trainer) Fit(x, y *nn.Tensor, opts ...FitOption) (*History, error) {
	cfg := &FitConfig{seed: 42}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.epochs <= 0 {
		return nil, fmt.Errorf("train: Fit requires train.Epochs(n) with n > 0")
	}
	if cfg.batchSize <= 0 {
		cfg.batchSize = x.Shape[0]
	}
	if cfg.callbacks == nil {
		cfg.callbacks = NewCallbackList()
	}
	if cfg.saveFunc != nil {
		for _, cb := range cfg.callbacks.callbacks {
			if mc, ok := cb.(*ModelCheckpointCallback); ok {
				mc.Save = func(path string) error { return cfg.saveFunc(t.model, path) }
			}
		}
	}

	opt := t.opt
	if cfg.clipGradMaxNorm > 0 {
		opt = ClipNorm(t.opt, cfg.clipGradMaxNorm)
	}

	rng := nn.NewRNG(cfg.seed)
	ctx := &nn.Context{Mode: nn.Train, RNG: rng}

	numSamples := x.Shape[0]
	featShape := append([]int(nil), x.Shape[1:]...)
	labelShape := append([]int(nil), y.Shape[1:]...)

	modules := t.model.Modules()
	fused := false
	if ce, ok := t.loss.(*CrossEntropyLoss); ok {
		fused = ce.Fused()
	}

	hist := NewHistory()
	hist.OnTrainBegin()
	cfg.callbacks.OnTrainBegin()

	for epoch := 0; epoch < cfg.epochs; epoch++ {
		cfg.callbacks.OnEpochBegin(epoch)

		indices := make([]int, numSamples)
		for i := range indices {
			indices[i] = i
		}
		if cfg.shuffle {
			rng.Shuffle(numSamples, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		}

		var epochLoss float32
		numBatches := 0
		for start := 0; start < numSamples; start += cfg.batchSize {
			end := start + cfg.batchSize
			if end > numSamples {
				end = numSamples
			}
			batchIdx := indices[start:end]
			xb := gatherRows(x, batchIdx, featShape)
			yb := gatherRows(y, batchIdx, labelShape)

			pred, err := t.model.Forward(ctx, xb)
			if err != nil {
				return nil, fmt.Errorf("train: forward: %w", err)
			}
			lossVal, gradOut, err := t.loss.Loss(pred, yb)
			if err != nil {
				return nil, fmt.Errorf("train: loss: %w", err)
			}

			if fused && len(modules) > 0 {
				grad := gradOut
				for i := len(modules) - 2; i >= 0; i-- {
					if grad, err = modules[i].Backward(ctx, grad); err != nil {
						return nil, fmt.Errorf("train: backward: %w", err)
					}
				}
			} else if _, err := t.model.Backward(ctx, gradOut); err != nil {
				return nil, fmt.Errorf("train: backward: %w", err)
			}

			opt.Step(t.model.Params())

			epochLoss += lossVal
			numBatches++
			cfg.callbacks.OnBatchEnd(numBatches-1, lossVal)
		}
		avgTrainLoss := epochLoss / float32(numBatches)

		var valMetrics *Metrics
		if cfg.hasVal {
			m, err := t.Evaluate(cfg.valX, cfg.valY)
			if err != nil {
				return nil, fmt.Errorf("train: validation evaluate: %w", err)
			}
			valMetrics = &m
		}

		params := t.model.Params()
		hist.OnEpochEnd(epoch, avgTrainLoss, valMetrics, params)
		cfg.callbacks.OnEpochEnd(epoch, avgTrainLoss, valMetrics, params)

		stop := false
		for _, cb := range cfg.callbacks.callbacks {
			if es, ok := cb.(*EarlyStoppingCallback); ok && es.ShouldStop {
				es.RestoreBestWeights(params)
				stop = true
			}
		}
		if stop {
			break
		}
	}

	hist.OnTrainEnd()
	cfg.callbacks.OnTrainEnd()
	return hist, nil
}

// Predict runs the model in Inference mode (Dropout off, BatchNorm on
// running stats).
func (t *Trainer) Predict(x *nn.Tensor) (*nn.Tensor, error) {
	return t.model.Forward(&nn.Context{Mode: nn.Inference}, x)
}

// Evaluate runs the model in Inference mode and returns loss plus the full
// Metrics (accuracy/precision/recall/F1/confusion), macro-averaged for
// multiclass.
func (t *Trainer) Evaluate(x, y *nn.Tensor) (Metrics, error) {
	pred, err := t.Predict(x)
	if err != nil {
		return Metrics{}, err
	}
	lossVal, _, err := t.loss.Loss(pred, y)
	if err != nil {
		return Metrics{}, err
	}
	return computeMetrics(lossVal, pred, y)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./train/... -run "TestFitRequiresEpochs|TestXORConverges|TestMulticlassConverges|TestEvaluateReturnsPopulatedMetrics" -v`
Expected: PASS. `TestPredictMatchesForwardInInferenceMode` stays skipped until Task 16.

- [ ] **Step 5: Run the full train package suite so far**

Run: `go test ./train/... -v`
Expected: PASS (all tests from Tasks 7–12, one `Skip`)

- [ ] **Step 6: Commit**

```bash
git add train/trainer.go train/trainer_test.go
git commit -m "feat(train): add Trainer with fused softmax+CCE, callbacks, validation"
```

---

## Phase 3: Conv stack + regularization

Conv2D/MaxPool/AvgPool/Flatten modules, Dropout, BatchNorm, crossval port, model summary. Synthetic CNN convergence test green. `Network/` deleted at the end of this phase (`tensor/` stays until Task 22 — see the File Structure note above for why).

### Task 13: Conv2D

**Files:**
- Create: `nn/conv.go`
- Test: `nn/conv_test.go`

**Interfaces:**
- Consumes: `nn.Module`, `nn.Param`, `nn.Initializer`, `checkInputGradient`, `checkParamGradient` (Tasks 2–4).
- Produces: `func Conv2D(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer`, `func Conv2DSame(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer` — both satisfy `nn.Module` operating on `[batch, h, w, channels]` tensors, per design decision #6.

- [ ] **Step 1: Write the failing tests**

```go
package nn

import "testing"

func TestConv2DOutputShapeValid(t *testing.T) {
	rng := NewRNG(1)
	c := Conv2D(rng, 1, 4, 3, HeInit())
	out, err := c.OutputShape([]int{2, 8, 8, 1})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 6, 6, 4} // (8-3)/1+1 = 6
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestConv2DSamePreservesSpatialDims(t *testing.T) {
	rng := NewRNG(1)
	c := Conv2DSame(rng, 1, 4, 3, HeInit())
	out, err := c.OutputShape([]int{2, 8, 8, 1})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	if out[1] != 8 || out[2] != 8 {
		t.Fatalf("Conv2DSame output spatial dims = %v, want [.. 8 8 ..]", out)
	}
}

func TestConv2DInputGradient(t *testing.T) {
	rng := NewRNG(2)
	c := Conv2D(rng, 2, 3, 3, HeInit())
	if _, err := c.OutputShape([]int{1, 5, 5, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 5, 5, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%7) * 0.1 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
}

func TestConv2DParamGradients(t *testing.T) {
	rng := NewRNG(3)
	c := Conv2D(rng, 2, 3, 3, HeInit())
	if _, err := c.OutputShape([]int{1, 5, 5, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 5, 5, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.15 - 0.3
	}
	ctx := &Context{Mode: Inference}
	forward := func() (*Tensor, error) { return c.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return c.Backward(ctx, g) }
	for _, p := range c.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestConv2DSameGradients(t *testing.T) {
	rng := NewRNG(4)
	c := Conv2DSame(rng, 1, 2, 3, HeInit())
	if _, err := c.OutputShape([]int{1, 4, 4, 1}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 4, 4, 1})
	for i := range x.Data {
		x.Data[i] = float32(i%3)*0.2 - 0.2
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run TestConv2D -v`
Expected: FAIL — `Conv2D` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import (
	"fmt"
	"math/rand"
)

type Conv2DLayer struct {
	inChannels, outChannels, kernelSize, padding int
	W, B                                         *Param
	init                                         Initializer
	rng                                          *rand.Rand
	input                                        *Tensor
}

// Conv2D is stride-1, zero-padding ("valid").
func Conv2D(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer {
	return newConv2D(rng, inChannels, outChannels, kernelSize, 0, init)
}

// Conv2DSame is stride-1 with padding (kernelSize-1)/2 ("same"); kernelSize
// must be odd.
func Conv2DSame(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer {
	return newConv2D(rng, inChannels, outChannels, kernelSize, (kernelSize-1)/2, init)
}

func newConv2D(rng *rand.Rand, inChannels, outChannels, kernelSize, padding int, init Initializer) *Conv2DLayer {
	if init == nil {
		init = HeInit()
	}
	c := &Conv2DLayer{inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize, padding: padding, init: init, rng: rng}
	c.W = NewParam(init(rng, []int{outChannels, inChannels, kernelSize, kernelSize}))
	c.B = NewParam(NewTensor([]int{outChannels}))
	return c
}

func (c *Conv2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 || inShape[3] != c.inChannels {
		return nil, fmt.Errorf("nn: Conv2D expects input shape [batch, h, w, %d], got %v", c.inChannels, inShape)
	}
	h, w := inShape[1], inShape[2]
	outH := (h+2*c.padding-c.kernelSize)/1 + 1
	outW := (w+2*c.padding-c.kernelSize)/1 + 1
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("nn: Conv2D input %dx%d too small for kernel %d with padding %d", h, w, c.kernelSize, c.padding)
	}
	return []int{inShape[0], outH, outW, c.outChannels}, nil
}

func (c *Conv2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 4 || x.Shape[3] != c.inChannels {
		return nil, fmt.Errorf("nn: Conv2D expects input shape [batch, h, w, %d], got %v", c.inChannels, x.Shape)
	}
	c.input = x
	batch, h, w := x.Shape[0], x.Shape[1], x.Shape[2]
	k, pad := c.kernelSize, c.padding
	outH := (h+2*pad-k)/1 + 1
	outW := (w+2*pad-k)/1 + 1
	out := NewTensor([]int{batch, outH, outW, c.outChannels})

	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for oc := 0; oc < c.outChannels; oc++ {
					sum := c.B.Value.Data[oc]
					for ic := 0; ic < c.inChannels; ic++ {
						for kh := 0; kh < k; kh++ {
							ih := oh - pad + kh
							if ih < 0 || ih >= h {
								continue
							}
							for kw := 0; kw < k; kw++ {
								iw := ow - pad + kw
								if iw < 0 || iw >= w {
									continue
								}
								xVal := x.Data[((b*h+ih)*w+iw)*c.inChannels+ic]
								wVal := c.W.Value.Data[((oc*c.inChannels+ic)*k+kh)*k+kw]
								sum += xVal * wVal
							}
						}
					}
					out.Data[((b*outH+oh)*outW+ow)*c.outChannels+oc] = sum
				}
			}
		}
	}
	return out, nil
}

func (c *Conv2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, h, w := c.input.Shape[0], c.input.Shape[1], c.input.Shape[2]
	k, pad := c.kernelSize, c.padding
	outH, outW := gradOut.Shape[1], gradOut.Shape[2]

	// No batch-scaling on W.Grad/B.Grad here — see design decision #6.
	gradIn := NewTensor(c.input.Shape)
	c.W.ZeroGrad()
	c.B.ZeroGrad()

	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for oc := 0; oc < c.outChannels; oc++ {
					g := gradOut.Data[((b*outH+oh)*outW+ow)*c.outChannels+oc]
					c.B.Grad.Data[oc] += g
					for ic := 0; ic < c.inChannels; ic++ {
						for kh := 0; kh < k; kh++ {
							ih := oh - pad + kh
							if ih < 0 || ih >= h {
								continue
							}
							for kw := 0; kw < k; kw++ {
								iw := ow - pad + kw
								if iw < 0 || iw >= w {
									continue
								}
								xVal := c.input.Data[((b*h+ih)*w+iw)*c.inChannels+ic]
								wVal := c.W.Value.Data[((oc*c.inChannels+ic)*k+kh)*k+kw]
								c.W.Grad.Data[((oc*c.inChannels+ic)*k+kh)*k+kw] += g * xVal
								gradIn.Data[((b*h+ih)*w+iw)*c.inChannels+ic] += g * wVal
							}
						}
					}
				}
			}
		}
	}
	return gradIn, nil
}

func (c *Conv2DLayer) Params() []*Param { return []*Param{c.W, c.B} }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run TestConv2D -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn/conv.go nn/conv_test.go
git commit -m "feat(nn): add batched Conv2D (valid and same padding) with gradient checks"
```

### Task 14: MaxPool2D, AvgPool2D

**Files:**
- Create: `nn/pooling.go`
- Test: `nn/pooling_test.go`

**Interfaces:**
- Consumes: `nn.Module`, `checkInputGradient` (Tasks 2, 4).
- Produces: `func MaxPool2D(poolSize, stride int) *MaxPool2DLayer`, `func AvgPool2D(poolSize, stride int) *AvgPool2DLayer` — both satisfy `nn.Module` on `[batch, h, w, channels]`.

- [ ] **Step 1: Write the failing tests**

```go
package nn

import "testing"

func TestMaxPool2DForward(t *testing.T) {
	// 1x4x4x1, pool 2x2 stride 2 -> 1x2x2x1
	x, _ := NewTensorFromData([]float32{
		1, 2, 5, 6,
		3, 4, 7, 8,
		9, 10, 13, 14,
		11, 12, 15, 16,
	}, []int{1, 4, 4, 1})
	m := MaxPool2D(2, 2)
	y, err := m.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []float32{4, 8, 12, 16}
	for i := range want {
		if y.Data[i] != want[i] {
			t.Errorf("y.Data[%d] = %v, want %v", i, y.Data[i], want[i])
		}
	}
}

func TestMaxPool2DGradient(t *testing.T) {
	x := NewTensor([]int{1, 4, 4, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%11) * 0.13
	}
	checkInputGradient(t, MaxPool2D(2, 2), &Context{}, x)
}

func TestAvgPool2DForward(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{1, 2, 2, 1})
	a := AvgPool2D(2, 2)
	y, err := a.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if diff := y.Data[0] - 2.5; diff > 1e-5 || diff < -1e-5 {
		t.Fatalf("y.Data[0] = %v, want 2.5", y.Data[0])
	}
}

func TestAvgPool2DGradient(t *testing.T) {
	x := NewTensor([]int{1, 4, 4, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%9) * 0.11
	}
	checkInputGradient(t, AvgPool2D(2, 2), &Context{}, x)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run "TestMaxPool2D|TestAvgPool2D" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import "fmt"

type MaxPool2DLayer struct {
	poolSize, stride    int
	input                *Tensor
	outH, outW           int
	maxIdx                []int // flat input index chosen for each output element
}

func MaxPool2D(poolSize, stride int) *MaxPool2DLayer {
	return &MaxPool2DLayer{poolSize: poolSize, stride: stride}
}

func (m *MaxPool2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 {
		return nil, fmt.Errorf("nn: MaxPool2D expects input shape [batch, h, w, c], got %v", inShape)
	}
	outH := (inShape[1]-m.poolSize)/m.stride + 1
	outW := (inShape[2]-m.poolSize)/m.stride + 1
	return []int{inShape[0], outH, outW, inShape[3]}, nil
}

func (m *MaxPool2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 4 {
		return nil, fmt.Errorf("nn: MaxPool2D expects input shape [batch, h, w, c], got %v", x.Shape)
	}
	m.input = x
	batch, h, w, ch := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outH := (h-m.poolSize)/m.stride + 1
	outW := (w-m.poolSize)/m.stride + 1
	m.outH, m.outW = outH, outW
	out := NewTensor([]int{batch, outH, outW, ch})
	m.maxIdx = make([]int, batch*outH*outW*ch)

	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for c := 0; c < ch; c++ {
					best := float32(0)
					bestIdx := -1
					for ph := 0; ph < m.poolSize; ph++ {
						ih := oh*m.stride + ph
						for pw := 0; pw < m.poolSize; pw++ {
							iw := ow*m.stride + pw
							idx := ((b*h+ih)*w+iw)*ch + c
							v := x.Data[idx]
							if bestIdx == -1 || v > best {
								best = v
								bestIdx = idx
							}
						}
					}
					outIdx := ((b*outH+oh)*outW+ow)*ch + c
					out.Data[outIdx] = best
					m.maxIdx[outIdx] = bestIdx
				}
			}
		}
	}
	return out, nil
}

func (m *MaxPool2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(m.input.Shape)
	for outIdx, srcIdx := range m.maxIdx {
		gradIn.Data[srcIdx] += gradOut.Data[outIdx]
	}
	return gradIn, nil
}

func (m *MaxPool2DLayer) Params() []*Param { return nil }

type AvgPool2DLayer struct {
	poolSize, stride int
	inputShape        []int
	outH, outW         int
}

func AvgPool2D(poolSize, stride int) *AvgPool2DLayer {
	return &AvgPool2DLayer{poolSize: poolSize, stride: stride}
}

func (a *AvgPool2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 {
		return nil, fmt.Errorf("nn: AvgPool2D expects input shape [batch, h, w, c], got %v", inShape)
	}
	outH := (inShape[1]-a.poolSize)/a.stride + 1
	outW := (inShape[2]-a.poolSize)/a.stride + 1
	return []int{inShape[0], outH, outW, inShape[3]}, nil
}

func (a *AvgPool2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 4 {
		return nil, fmt.Errorf("nn: AvgPool2D expects input shape [batch, h, w, c], got %v", x.Shape)
	}
	a.inputShape = x.Shape
	batch, h, w, ch := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outH := (h-a.poolSize)/a.stride + 1
	outW := (w-a.poolSize)/a.stride + 1
	a.outH, a.outW = outH, outW
	out := NewTensor([]int{batch, outH, outW, ch})
	area := float32(a.poolSize * a.poolSize)

	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for c := 0; c < ch; c++ {
					var sum float32
					for ph := 0; ph < a.poolSize; ph++ {
						ih := oh*a.stride + ph
						for pw := 0; pw < a.poolSize; pw++ {
							iw := ow*a.stride + pw
							sum += x.Data[((b*h+ih)*w+iw)*ch+c]
						}
					}
					out.Data[((b*outH+oh)*outW+ow)*ch+c] = sum / area
				}
			}
		}
	}
	return out, nil
}

func (a *AvgPool2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(a.inputShape)
	h, w, ch := a.inputShape[1], a.inputShape[2], a.inputShape[3]
	area := float32(a.poolSize * a.poolSize)
	batch := a.inputShape[0]

	for b := 0; b < batch; b++ {
		for oh := 0; oh < a.outH; oh++ {
			for ow := 0; ow < a.outW; ow++ {
				for c := 0; c < ch; c++ {
					g := gradOut.Data[((b*a.outH+oh)*a.outW+ow)*ch+c] / area
					for ph := 0; ph < a.poolSize; ph++ {
						ih := oh*a.stride + ph
						for pw := 0; pw < a.poolSize; pw++ {
							iw := ow*a.stride + pw
							gradIn.Data[((b*h+ih)*w+iw)*ch+c] += g
						}
					}
				}
			}
		}
	}
	return gradIn, nil
}

func (a *AvgPool2DLayer) Params() []*Param { return nil }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run "TestMaxPool2D|TestAvgPool2D" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn/pooling.go nn/pooling_test.go
git commit -m "feat(nn): add MaxPool2D and AvgPool2D with gradient checks"
```

### Task 15: Flatten

**Files:**
- Create: `nn/flatten.go`
- Test: `nn/flatten_test.go`

**Interfaces:**
- Consumes: `nn.Module`, `checkInputGradient` (Tasks 2, 4).
- Produces: `func Flatten() *FlattenLayer` satisfying `nn.Module`, mapping `[batch, h, w, c]` to `[batch, h*w*c]`.

- [ ] **Step 1: Write the failing tests**

```go
package nn

import "testing"

func TestFlattenForwardPreservesOrder(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{1, 2, 2, 2})
	f := Flatten()
	y, err := f.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if y.Shape[0] != 1 || y.Shape[1] != 8 {
		t.Fatalf("Flatten shape = %v, want [1 8]", y.Shape)
	}
	for i := range x.Data {
		if y.Data[i] != x.Data[i] {
			t.Errorf("y.Data[%d] = %v, want %v", i, y.Data[i], x.Data[i])
		}
	}
}

func TestFlattenBackwardReshapes(t *testing.T) {
	x := NewTensor([]int{2, 2, 2, 3})
	f := Flatten()
	ctx := &Context{}
	if _, err := f.Forward(ctx, x); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gradOut := NewTensor([]int{2, 12})
	for i := range gradOut.Data {
		gradOut.Data[i] = float32(i)
	}
	gradIn, err := f.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if gradIn.Shape[0] != 2 || gradIn.Shape[1] != 2 || gradIn.Shape[2] != 2 || gradIn.Shape[3] != 3 {
		t.Fatalf("gradIn shape = %v, want [2 2 2 3]", gradIn.Shape)
	}
	for i := range gradOut.Data {
		if gradIn.Data[i] != gradOut.Data[i] {
			t.Errorf("gradIn.Data[%d] = %v, want %v", i, gradIn.Data[i], gradOut.Data[i])
		}
	}
}

func TestFlattenGradient(t *testing.T) {
	x := NewTensor([]int{2, 2, 2, 2})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.1
	}
	checkInputGradient(t, Flatten(), &Context{}, x)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run TestFlatten -v`
Expected: FAIL — `Flatten` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import "fmt"

type FlattenLayer struct {
	inputShape []int
}

func Flatten() *FlattenLayer { return &FlattenLayer{} }

func (f *FlattenLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) < 2 {
		return nil, fmt.Errorf("nn: Flatten expects at least [batch, ...], got %v", inShape)
	}
	size := 1
	for _, d := range inShape[1:] {
		size *= d
	}
	return []int{inShape[0], size}, nil
}

func (f *FlattenLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	f.inputShape = append([]int(nil), x.Shape...)
	out, err := f.OutputShape(x.Shape)
	if err != nil {
		return nil, err
	}
	data := append([]float32(nil), x.Data...)
	return &Tensor{Data: data, Shape: out}, nil
}

func (f *FlattenLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	data := append([]float32(nil), gradOut.Data...)
	return &Tensor{Data: data, Shape: append([]int(nil), f.inputShape...)}, nil
}

func (f *FlattenLayer) Params() []*Param { return nil }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run TestFlatten -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn/flatten.go nn/flatten_test.go
git commit -m "feat(nn): add Flatten module"
```

### Task 16: Dropout

**Files:**
- Create: `nn/dropout.go`
- Modify: `train/trainer_test.go` — remove the `t.Skip(...)` line from `TestPredictMatchesForwardInInferenceMode` (added in Task 12) now that `Dropout` exists.
- Test: `nn/dropout_test.go`

**Interfaces:**
- Consumes: `nn.Module`, `nn.Context`, `nn.Train`/`nn.Inference` (Task 2).
- Produces: `func Dropout(rate float32) *DropoutLayer` satisfying `nn.Module` — inverted dropout in `Train` mode using `ctx.RNG`, identity in `Inference`.

- [ ] **Step 1: Write the failing tests**

```go
package nn

import "testing"

func TestDropoutIdentityInInferenceMode(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{4})
	d := Dropout(0.9) // even at a high rate, inference must be identity
	y, err := d.Forward(&Context{Mode: Inference}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := range x.Data {
		if y.Data[i] != x.Data[i] {
			t.Errorf("y.Data[%d] = %v, want %v (identity)", i, y.Data[i], x.Data[i])
		}
	}
}

func TestDropoutIdentityWhenRateZero(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{4})
	d := Dropout(0)
	y, err := d.Forward(&Context{Mode: Train, RNG: NewRNG(1)}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := range x.Data {
		if y.Data[i] != x.Data[i] {
			t.Errorf("y.Data[%d] = %v, want %v (identity at rate 0)", i, y.Data[i], x.Data[i])
		}
	}
}

func TestDropoutApproximatesRateStatistically(t *testing.T) {
	rate := float32(0.3)
	x := NewTensor([]int{10000})
	for i := range x.Data {
		x.Data[i] = 1
	}
	d := Dropout(rate)
	y, err := d.Forward(&Context{Mode: Train, RNG: NewRNG(9)}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	zeroed := 0
	for _, v := range y.Data {
		if v == 0 {
			zeroed++
		}
	}
	frac := float64(zeroed) / float64(len(y.Data))
	if frac < 0.25 || frac > 0.35 {
		t.Fatalf("dropped fraction = %v, want close to %v", frac, rate)
	}
}

func TestDropoutBackwardScalesByRecordedMask(t *testing.T) {
	d := Dropout(0.5)
	x := NewTensor([]int{20})
	for i := range x.Data {
		x.Data[i] = 1
	}
	ctx := &Context{Mode: Train, RNG: NewRNG(7)}
	y, err := d.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gradOut := NewTensor([]int{20})
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradIn, err := d.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	for i := range gradIn.Data {
		if y.Data[i] == 0 {
			if gradIn.Data[i] != 0 {
				t.Errorf("gradIn[%d] = %v, want 0 (element was dropped)", i, gradIn.Data[i])
			}
		} else if diff := gradIn.Data[i] - y.Data[i]; diff > 1e-5 || diff < -1e-5 {
			// x[i]==1, so y[i] IS the scale factor applied; gradIn should match it exactly.
			t.Errorf("gradIn[%d] = %v, want %v", i, gradIn.Data[i], y.Data[i])
		}
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run TestDropout -v`
Expected: FAIL — `Dropout` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

type DropoutLayer struct {
	rate float32
	mask []float32 // per-element scale factor recorded by the last Train-mode Forward (0 or 1/(1-rate)); nil after an Inference Forward
}

func Dropout(rate float32) *DropoutLayer { return &DropoutLayer{rate: rate} }

func (d *DropoutLayer) OutputShape(inShape []int) ([]int, error) { return inShape, nil }

func (d *DropoutLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if ctx.Mode != Train || d.rate == 0 {
		d.mask = nil
		return x.Clone(), nil
	}
	scale := 1.0 / (1.0 - d.rate)
	out := NewTensor(x.Shape)
	d.mask = make([]float32, len(x.Data))
	for i, v := range x.Data {
		if ctx.RNG.Float32() > d.rate {
			out.Data[i] = v * scale
			d.mask[i] = scale
		}
	}
	return out, nil
}

func (d *DropoutLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(gradOut.Shape)
	if d.mask == nil {
		copy(gradIn.Data, gradOut.Data)
		return gradIn, nil
	}
	for i := range gradOut.Data {
		gradIn.Data[i] = gradOut.Data[i] * d.mask[i]
	}
	return gradIn, nil
}

func (d *DropoutLayer) Params() []*Param { return nil }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run TestDropout -v`
Expected: PASS

- [ ] **Step 5: Un-skip and run the Task 12 Predict/inference test**

Edit `train/trainer_test.go`: delete the `t.Skip(...)` line (if one was added) at the top of `TestPredictMatchesForwardInInferenceMode`.

Run: `go test ./train/... -run TestPredictMatchesForwardInInferenceMode -v`
Expected: PASS — `Dropout` in the model is now a real, deterministic identity in `Inference` mode.

- [ ] **Step 6: Commit**

```bash
git add nn/dropout.go nn/dropout_test.go train/trainer_test.go
git commit -m "feat(nn): add Dropout module; un-skip Predict inference-mode test"
```

### Task 17: BatchNorm

**Files:**
- Create: `nn/norm.go`
- Test: `nn/norm_test.go`

**Interfaces:**
- Consumes: `nn.Module`, `nn.Param`, `checkInputGradient`, `checkParamGradient` (Tasks 2, 4).
- Produces: `func BatchNorm(channels int) *BatchNormLayer` satisfying `nn.Module` — trainable `Gamma`/`Beta` (`Params()`), running mean/variance updated in `Train`, used in `Inference`, over the last (channel) dimension for both 2D and 4D inputs (design decision #5).

- [ ] **Step 1: Write the failing tests**

```go
package nn

import (
	"math"
	"testing"
)

func TestBatchNormNormalizesTrainBatch(t *testing.T) {
	bn := BatchNorm(2)
	// channel 0: [1,3,5,7] mean=4 var=5; channel 1: [10,10,10,10] mean=10 var=0
	x, _ := NewTensorFromData([]float32{1, 10, 3, 10, 5, 10, 7, 10}, []int{4, 2})
	y, err := bn.Forward(&Context{Mode: Train}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// gamma=1, beta=0 initially, so y should equal xhat.
	var mean0 float32
	for i := 0; i < 4; i++ {
		mean0 += y.Data[i*2]
	}
	mean0 /= 4
	if math.Abs(float64(mean0)) > 1e-4 {
		t.Fatalf("channel 0 normalized mean = %v, want ~0", mean0)
	}
	for i := 0; i < 4; i++ {
		if math.Abs(float64(y.Data[i*2+1])) > 1e-2 {
			t.Fatalf("channel 1 (zero variance) normalized value = %v, want ~0", y.Data[i*2+1])
		}
	}
}

func TestBatchNormUsesRunningStatsInInference(t *testing.T) {
	bn := BatchNorm(1)
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{4, 1})
	if _, err := bn.Forward(&Context{Mode: Train}, x); err != nil {
		t.Fatalf("Forward (train): %v", err)
	}
	// A single-sample inference pass can't compute its own batch stats;
	// it must reuse the running stats recorded during the Train pass above.
	single, _ := NewTensorFromData([]float32{100}, []int{1, 1})
	y, err := bn.Forward(&Context{Mode: Inference}, single)
	if err != nil {
		t.Fatalf("Forward (inference): %v", err)
	}
	// With running mean far below 100 and small running variance, the
	// normalized output should be large and positive, not exactly 0.
	if y.Data[0] < 5 {
		t.Fatalf("inference output = %v, want a large positive value reflecting the running stats", y.Data[0])
	}
}

func TestBatchNormInputGradientDense(t *testing.T) {
	bn := BatchNorm(3)
	x := NewTensor([]int{5, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.3 - 1.0
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, bn, ctx, x)
}

func TestBatchNormInputGradientConv(t *testing.T) {
	bn := BatchNorm(2)
	x := NewTensor([]int{2, 3, 3, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.2 - 0.4
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, bn, ctx, x)
}

func TestBatchNormParamGradients(t *testing.T) {
	bn := BatchNorm(3)
	x := NewTensor([]int{5, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.25 - 0.8
	}
	ctx := &Context{Mode: Train}
	forward := func() (*Tensor, error) { return bn.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return bn.Backward(ctx, g) }
	for _, p := range bn.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run TestBatchNorm -v`
Expected: FAIL — `BatchNorm` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import (
	"fmt"
	"math"
)

// BatchNormLayer normalizes over the last (channel) dimension. Because
// channels are always the fastest-varying axis in this codebase's tensor
// layout (design decision #5), one implementation serves both dense
// [batch, features] and conv [batch, h, w, channels] inputs: a flat index
// idx belongs to channel idx % channels, and its statistics are computed
// over the N = size/channels elements sharing that channel.
type BatchNormLayer struct {
	channels               int
	Gamma, Beta            *Param
	eps, momentum          float32
	runningMean, runningVar []float32

	input               *Tensor
	normalized          []float32
	batchMean, batchVar []float32
}

func BatchNorm(channels int) *BatchNormLayer {
	gamma := NewTensor([]int{channels})
	for i := range gamma.Data {
		gamma.Data[i] = 1
	}
	runningVar := make([]float32, channels)
	for i := range runningVar {
		runningVar[i] = 1
	}
	return &BatchNormLayer{
		channels:    channels,
		Gamma:       NewParam(gamma),
		Beta:        NewParam(NewTensor([]int{channels})),
		eps:         1e-5,
		momentum:    0.9,
		runningMean: make([]float32, channels),
		runningVar:  runningVar,
	}
}

func (bn *BatchNormLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) == 0 || inShape[len(inShape)-1] != bn.channels {
		return nil, fmt.Errorf("nn: BatchNorm configured for %d channels, got shape %v", bn.channels, inShape)
	}
	return inShape, nil
}

func (bn *BatchNormLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := bn.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	out := NewTensor(x.Shape)

	if ctx.Mode != Train {
		for i, v := range x.Data {
			c := i % bn.channels
			xhat := (v - bn.runningMean[c]) / float32(math.Sqrt(float64(bn.runningVar[c]+bn.eps)))
			out.Data[i] = bn.Gamma.Value.Data[c]*xhat + bn.Beta.Value.Data[c]
		}
		return out, nil
	}

	n := len(x.Data) / bn.channels
	mean := make([]float32, bn.channels)
	for i, v := range x.Data {
		mean[i%bn.channels] += v
	}
	for c := range mean {
		mean[c] /= float32(n)
	}
	variance := make([]float32, bn.channels)
	for i, v := range x.Data {
		d := v - mean[i%bn.channels]
		variance[i%bn.channels] += d * d
	}
	for c := range variance {
		variance[c] /= float32(n)
	}

	bn.input = x
	bn.batchMean = mean
	bn.batchVar = variance
	bn.normalized = make([]float32, len(x.Data))
	for i, v := range x.Data {
		c := i % bn.channels
		xhat := (v - mean[c]) / float32(math.Sqrt(float64(variance[c]+bn.eps)))
		bn.normalized[i] = xhat
		out.Data[i] = bn.Gamma.Value.Data[c]*xhat + bn.Beta.Value.Data[c]
	}
	for c := 0; c < bn.channels; c++ {
		bn.runningMean[c] = bn.momentum*bn.runningMean[c] + (1-bn.momentum)*mean[c]
		bn.runningVar[c] = bn.momentum*bn.runningVar[c] + (1-bn.momentum)*variance[c]
	}
	return out, nil
}

// Backward is the standard batchnorm backward derivation; see design
// decision #6 for why no additional batch-scaling is applied to
// Gamma.Grad/Beta.Grad.
func (bn *BatchNormLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	n := len(bn.input.Data) / bn.channels
	nf := float32(n)
	gradIn := NewTensor(bn.input.Shape)
	bn.Gamma.ZeroGrad()
	bn.Beta.ZeroGrad()

	dxhat := make([]float32, len(bn.input.Data))
	for i, dy := range gradOut.Data {
		c := i % bn.channels
		bn.Gamma.Grad.Data[c] += dy * bn.normalized[i]
		bn.Beta.Grad.Data[c] += dy
		dxhat[i] = dy * bn.Gamma.Value.Data[c]
	}

	invStd := make([]float32, bn.channels)
	for c := 0; c < bn.channels; c++ {
		invStd[c] = 1.0 / float32(math.Sqrt(float64(bn.batchVar[c]+bn.eps)))
	}

	dvar := make([]float32, bn.channels)
	for i, dxh := range dxhat {
		c := i % bn.channels
		centered := bn.input.Data[i] - bn.batchMean[c]
		dvar[c] += dxh * centered * -0.5 * invStd[c] * invStd[c] * invStd[c]
	}

	dmean := make([]float32, bn.channels)
	for i, dxh := range dxhat {
		c := i % bn.channels
		dmean[c] += dxh * -invStd[c]
	}

	for i := range bn.input.Data {
		c := i % bn.channels
		centered := bn.input.Data[i] - bn.batchMean[c]
		gradIn.Data[i] = dxhat[i]*invStd[c] + dvar[c]*2*centered/nf + dmean[c]/nf
	}
	return gradIn, nil
}

func (bn *BatchNormLayer) Params() []*Param { return []*Param{bn.Gamma, bn.Beta} }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run TestBatchNorm -v`
Expected: PASS

- [ ] **Step 5: Run the full nn package suite so far**

Run: `go test ./nn/... -v`
Expected: PASS (all tests from Tasks 1–17)

- [ ] **Step 6: Commit**

```bash
git add nn/norm.go nn/norm_test.go
git commit -m "feat(nn): add BatchNorm with running stats, generic over dense/conv shapes"
```

### Task 18: Cross-validation

**Files:**
- Create: `train/crossval.go`
- Test: `train/crossval_test.go`

**Interfaces:**
- Consumes: `train.Metrics` (Task 8).
- Produces: `type Fold struct { TrainX, TrainY, TestX, TestY [][]float32 }`, `func KFoldSplits(rng *rand.Rand, x, y [][]float32, k int, shuffle bool) []Fold`, `func StratifiedKFoldSplits(rng *rand.Rand, x, y [][]float32, k int) []Fold` (binary stratification on `y[i][0] > 0.5`), `type CrossValResult struct { FoldMetrics []Metrics; MeanAccuracy, StdAccuracy, MeanF1, StdF1, MeanLoss, StdLoss float32; BestFold, WorstFold int }`, `func CrossValidate(folds []Fold, trainFold func(fold Fold) (Metrics, error)) (CrossValResult, error)`.

Note: `CrossValidate` takes a `trainFold` closure instead of a model-construction callback tied to `*Trainer` — this is what keeps cross-validation decoupled from exactly how a caller builds and trains a model per fold (dense or conv, whichever optimizer/loss), matching how the old `Network/crossval.go` was tightly coupled to `Network.Train`/`Network.Evaluate` and is exactly the kind of coupling this restructure is meant to remove.

- [ ] **Step 1: Write the failing tests**

```go
package train

import (
	"fmt"
	"testing"
)

func makeXY(n int) ([][]float32, [][]float32) {
	x := make([][]float32, n)
	y := make([][]float32, n)
	for i := 0; i < n; i++ {
		x[i] = []float32{float32(i)}
		class := float32(0)
		if i%2 == 0 {
			class = 1
		}
		y[i] = []float32{class}
	}
	return x, y
}

func TestKFoldSplitsSizesAndNoOverlap(t *testing.T) {
	x, y := makeXY(10)
	folds := KFoldSplits(newTestRNG(1), x, y, 5, false)
	if len(folds) != 5 {
		t.Fatalf("len(folds) = %d, want 5", len(folds))
	}
	seenAsTest := map[float32]int{}
	for _, f := range folds {
		if len(f.TestX)+len(f.TrainX) != 10 {
			t.Fatalf("fold train+test = %d, want 10", len(f.TestX)+len(f.TrainX))
		}
		for _, row := range f.TestX {
			seenAsTest[row[0]]++
		}
	}
	if len(seenAsTest) != 10 {
		t.Fatalf("distinct samples seen as test = %d, want 10", len(seenAsTest))
	}
	for v, count := range seenAsTest {
		if count != 1 {
			t.Errorf("sample %v appeared as test in %d folds, want exactly 1", v, count)
		}
	}
}

func TestStratifiedKFoldPreservesClassRatioPerFold(t *testing.T) {
	x, y := makeXY(20) // exactly 10 class-1, 10 class-0
	folds := StratifiedKFoldSplits(newTestRNG(2), x, y, 4)
	for i, f := range folds {
		var ones int
		for _, row := range f.TestY {
			if row[0] > 0.5 {
				ones++
			}
		}
		if ones != len(f.TestY)/2 {
			t.Errorf("fold %d: %d/%d test samples are class 1, want exactly half", i, ones, len(f.TestY))
		}
	}
}

func TestCrossValidateAggregatesMeanAndStd(t *testing.T) {
	folds := []Fold{{}, {}, {}} // trainFold below ignores fold contents
	accuracies := []float32{60, 70, 80}
	i := 0
	trainFold := func(f Fold) (Metrics, error) {
		m := Metrics{Accuracy: accuracies[i], F1Score: accuracies[i] / 100, Loss: 1 - accuracies[i]/100}
		i++
		return m, nil
	}
	result, err := CrossValidate(folds, trainFold)
	if err != nil {
		t.Fatalf("CrossValidate: %v", err)
	}
	if result.MeanAccuracy != 70 {
		t.Fatalf("MeanAccuracy = %v, want 70", result.MeanAccuracy)
	}
	if result.BestFold != 2 || result.WorstFold != 0 {
		t.Fatalf("BestFold=%d WorstFold=%d, want 2 and 0", result.BestFold, result.WorstFold)
	}
}

func TestCrossValidatePropagatesFoldError(t *testing.T) {
	folds := []Fold{{}}
	_, err := CrossValidate(folds, func(f Fold) (Metrics, error) {
		return Metrics{}, fmt.Errorf("boom")
	})
	if err == nil {
		t.Fatal("expected error to propagate from a failing fold, got nil")
	}
}
```

`newTestRNG` is a tiny helper added in this task's test file (`train/crossval_test.go`) so tests read `newTestRNG(1)` instead of importing `math/rand` directly in every test:

```go
func newTestRNG(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }
```

Add `"math/rand"` to this test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./train/... -run "TestKFoldSplits|TestStratifiedKFold|TestCrossValidate" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
package train

import (
	"fmt"
	"math"
	"math/rand"
)

type Fold struct {
	TrainX, TrainY, TestX, TestY [][]float32
}

func KFoldSplits(rng *rand.Rand, x, y [][]float32, k int, shuffle bool) []Fold {
	n := len(x)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	if shuffle {
		rng.Shuffle(n, func(i, j int) { order[i], order[j] = order[j], order[i] })
	}

	foldSize := n / k
	folds := make([]Fold, k)
	for f := 0; f < k; f++ {
		start := f * foldSize
		end := start + foldSize
		if f == k-1 {
			end = n
		}
		var fold Fold
		for i, idx := range order {
			if i >= start && i < end {
				fold.TestX = append(fold.TestX, x[idx])
				fold.TestY = append(fold.TestY, y[idx])
			} else {
				fold.TrainX = append(fold.TrainX, x[idx])
				fold.TrainY = append(fold.TrainY, y[idx])
			}
		}
		folds[f] = fold
	}
	return folds
}

// StratifiedKFoldSplits assumes binary labels (y[i][0] > 0.5 == class 1)
// and preserves the class ratio in every fold's test split.
func StratifiedKFoldSplits(rng *rand.Rand, x, y [][]float32, k int) []Fold {
	var x0, y0, x1, y1 [][]float32
	for i := range y {
		if y[i][0] > 0.5 {
			x1, y1 = append(x1, x[i]), append(y1, y[i])
		} else {
			x0, y0 = append(x0, x[i]), append(y0, y[i])
		}
	}
	folds0 := KFoldSplits(rng, x0, y0, k, true)
	folds1 := KFoldSplits(rng, x1, y1, k, true)

	folds := make([]Fold, k)
	for f := 0; f < k; f++ {
		folds[f] = Fold{
			TrainX: append(append([][]float32{}, folds0[f].TrainX...), folds1[f].TrainX...),
			TrainY: append(append([][]float32{}, folds0[f].TrainY...), folds1[f].TrainY...),
			TestX:  append(append([][]float32{}, folds0[f].TestX...), folds1[f].TestX...),
			TestY:  append(append([][]float32{}, folds0[f].TestY...), folds1[f].TestY...),
		}
	}
	return folds
}

type CrossValResult struct {
	FoldMetrics                []Metrics
	MeanAccuracy, StdAccuracy  float32
	MeanF1, StdF1              float32
	MeanLoss, StdLoss          float32
	BestFold, WorstFold        int
}

// CrossValidate calls trainFold once per fold — trainFold is expected to
// build a fresh model, train it on fold.TrainX/TrainY, and return the
// Metrics from evaluating it on fold.TestX/TestY.
func CrossValidate(folds []Fold, trainFold func(fold Fold) (Metrics, error)) (CrossValResult, error) {
	foldMetrics := make([]Metrics, len(folds))
	for i, f := range folds {
		m, err := trainFold(f)
		if err != nil {
			return CrossValResult{}, fmt.Errorf("train: fold %d: %w", i, err)
		}
		foldMetrics[i] = m
	}
	return summarizeFolds(foldMetrics), nil
}

func summarizeFolds(metrics []Metrics) CrossValResult {
	k := len(metrics)
	result := CrossValResult{FoldMetrics: metrics}
	bestAcc := float32(math.Inf(-1))
	worstAcc := float32(math.Inf(1))
	var sumAcc, sumF1, sumLoss float32
	for i, m := range metrics {
		sumAcc += m.Accuracy
		sumF1 += m.F1Score
		sumLoss += m.Loss
		if m.Accuracy > bestAcc {
			bestAcc = m.Accuracy
			result.BestFold = i
		}
		if m.Accuracy < worstAcc {
			worstAcc = m.Accuracy
			result.WorstFold = i
		}
	}
	result.MeanAccuracy = sumAcc / float32(k)
	result.MeanF1 = sumF1 / float32(k)
	result.MeanLoss = sumLoss / float32(k)

	var varAcc, varF1, varLoss float32
	for _, m := range metrics {
		varAcc += (m.Accuracy - result.MeanAccuracy) * (m.Accuracy - result.MeanAccuracy)
		varF1 += (m.F1Score - result.MeanF1) * (m.F1Score - result.MeanF1)
		varLoss += (m.Loss - result.MeanLoss) * (m.Loss - result.MeanLoss)
	}
	result.StdAccuracy = float32(math.Sqrt(float64(varAcc / float32(k))))
	result.StdF1 = float32(math.Sqrt(float64(varF1 / float32(k))))
	result.StdLoss = float32(math.Sqrt(float64(varLoss / float32(k))))
	return result
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./train/... -run "TestKFoldSplits|TestStratifiedKFold|TestCrossValidate" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train/crossval.go train/crossval_test.go
git commit -m "feat(train): add k-fold and stratified k-fold cross-validation"
```

### Task 19: Model summary

**Files:**
- Create: `nn/summary.go`
- Test: `nn/summary_test.go`

**Interfaces:**
- Consumes: `nn.SequentialModel`, `nn.Module`, `nn.Param` (Task 6).
- Produces: `func Summary(model *SequentialModel, inputShape []int) (string, error)`, `func ParamCount(model *SequentialModel) int`.

- [ ] **Step 1: Write the failing tests**

```go
package nn

import (
	"strings"
	"testing"
)

func TestParamCountMatchesSumOfParams(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 4},
		Linear(rng, 4, 5, XavierInit()), // 4*5 + 5 = 25
		ReLU(),
		Linear(rng, 5, 3, XavierInit()), // 5*3 + 3 = 18
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	if got, want := ParamCount(model), 43; got != want {
		t.Fatalf("ParamCount = %d, want %d", got, want)
	}
}

func TestSummaryListsEachLayerAndTotal(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 4},
		Linear(rng, 4, 5, XavierInit()),
		ReLU(),
		Linear(rng, 5, 3, XavierInit()),
		Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	out, err := Summary(model, []int{2, 4})
	if err != nil {
		t.Fatalf("Summary: %v", err)
	}
	for _, want := range []string{"Linear", "ReLU", "Softmax", "Total params: 43"} {
		if !strings.Contains(out, want) {
			t.Errorf("Summary output missing %q:\n%s", want, out)
		}
	}
}

func TestSummaryPropagatesShapeError(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 4}, Linear(rng, 4, 5, XavierInit()))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	if _, err := Summary(model, []int{2, 999}); err == nil {
		t.Fatal("expected error for a Summary inputShape that mismatches the built model, got nil")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run "TestParamCount|TestSummary" -v`
Expected: FAIL — undefined symbols.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import (
	"fmt"
	"strings"
)

func moduleTypeName(m Module) string {
	switch m.(type) {
	case *LinearLayer:
		return "Linear"
	case *Conv2DLayer:
		return "Conv2D"
	case *MaxPool2DLayer:
		return "MaxPool2D"
	case *AvgPool2DLayer:
		return "AvgPool2D"
	case *FlattenLayer:
		return "Flatten"
	case *DropoutLayer:
		return "Dropout"
	case *BatchNormLayer:
		return "BatchNorm"
	case *SoftmaxModule:
		return "Softmax"
	case *SequentialModel:
		return "Sequential"
	case *ActivationModule:
		return "ReLU" // refined below once ActivationModule tags its own name
	default:
		return fmt.Sprintf("%T", m)
	}
}

func paramCountOf(m Module) int {
	count := 0
	for _, p := range m.Params() {
		count += p.Value.Size()
	}
	return count
}

// ParamCount returns the total number of trainable scalars across every
// module in model.
func ParamCount(model *SequentialModel) int {
	return paramCountOf(model)
}

// Summary re-runs the same OutputShape chain Sequential used at
// construction (safe — OutputShape is idempotent for already-built
// modules) to print each layer's output shape and parameter count plus a
// grand total. This and ProgressBar are the only stdout writers permitted
// by the Global Constraints.
func Summary(model *SequentialModel, inputShape []int) (string, error) {
	var b strings.Builder
	fmt.Fprintf(&b, "%-4s %-12s %-24s %10s\n", "#", "Layer", "Output Shape", "Params")
	fmt.Fprintln(&b, strings.Repeat("-", 54))

	shape := inputShape
	total := 0
	for i, m := range model.Modules() {
		out, err := m.OutputShape(shape)
		if err != nil {
			return "", fmt.Errorf("nn: Summary module %d: %w", i, err)
		}
		shape = out
		n := paramCountOf(m)
		total += n
		fmt.Fprintf(&b, "%-4d %-12s %-24v %10d\n", i, moduleTypeName(m), shape, n)
	}
	fmt.Fprintln(&b, strings.Repeat("-", 54))
	fmt.Fprintf(&b, "Total params: %d\n", total)
	return b.String(), nil
}
```

The `moduleTypeName` switch's `*ActivationModule` case returning a hardcoded `"ReLU"` is a known rough edge: `ActivationModule` doesn't currently expose which activation it wraps (Task 5 only stores an `activationFn` closure). Fix it in this task, not later: add `name string` and `alpha float32` fields to `ActivationModule` in `nn/activation.go` (`name` set by every constructor — `"relu"`, `"sigmoid"`, `"tanh"`, `"leaky_relu"`, `"gelu"`; `alpha` set only by `LeakyReLU(alpha)`, left at its zero value otherwise), add exported accessors `func (a *ActivationModule) Name() string { return a.name }` and `func (a *ActivationModule) Alpha() float32 { return a.alpha }`, and change the switch case here to `case *ActivationModule: return m.(*ActivationModule).Name()`. These accessors are also exactly what Task 21 (serialization) needs to encode which activation — and which `LeakyReLU` slope — a saved `ActivationModule` is, so doing it now avoids touching `activation.go` a third time.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run "TestParamCount|TestSummary" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nn/summary.go nn/summary_test.go nn/activation.go
git commit -m "feat(nn): add model Summary and ParamCount; tag ActivationModule with its Name()"
```

### Task 20: Synthetic CNN convergence test; delete `Network/`

**Files:**
- Create: `train/cnn_e2e_test.go`
- Delete: `Network/` (whole directory, all 23 files)

**Interfaces:**
- Consumes: `nn.Conv2D`, `nn.MaxPool2D`, `nn.Flatten`, `nn.Linear`, `nn.Sigmoid`, `nn.Sequential` (Phase 1, Tasks 13–15); `train.New`, `train.Adam`, `train.BCELoss` (Tasks 9, 12).

This is the Phase 3 exit gate from design doc §5: "Synthetic CNN convergence test green. `Network/` deleted at the end of this phase." Nothing here is deleted until the test in Step 4 passes.

- [ ] **Step 1: Write the failing test**

```go
package train

import (
	"math/rand"
	"neugo/nn"
	"testing"
)

// syntheticImages builds n 8x8 single-channel images alternating between
// two easily-separable classes (bright left half vs. bright right half)
// with small per-pixel jitter, deterministic given rng.
func syntheticImages(rng *rand.Rand, n int) (*nn.Tensor, *nn.Tensor) {
	x := nn.NewTensor([]int{n, 8, 8, 1})
	y := nn.NewTensor([]int{n, 1})
	for i := 0; i < n; i++ {
		class := i % 2
		for h := 0; h < 8; h++ {
			for w := 0; w < 8; w++ {
				base := float32(0.1)
				if (w < 4) == (class == 0) {
					base = 0.9
				}
				jitter := (rng.Float32() - 0.5) * 0.05
				x.Data[(i*8+h)*8+w] = base + jitter
			}
		}
		y.Data[i] = float32(class)
	}
	return x, y
}

func TestSyntheticCNNConverges(t *testing.T) {
	dataRNG := nn.NewRNG(5)
	x, y := syntheticImages(dataRNG, 40)

	modelRNG := nn.NewRNG(6)
	model, err := nn.Sequential([]int{40, 8, 8, 1},
		nn.Conv2D(modelRNG, 1, 4, 3, nn.HeInit()),
		nn.ReLU(),
		nn.MaxPool2D(2, 2),
		nn.Flatten(),
		nn.Linear(modelRNG, 0, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}

	trainer := New(model, Adam(0.01, 0.9, 0.999, 1e-8), BCELoss())
	if _, err := trainer.Fit(x, y, Epochs(300), BatchSize(8), Shuffle(true), Seed(7)); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if metrics.Accuracy < 90 {
		t.Fatalf("Accuracy = %v after training on separable synthetic images, want >= 90", metrics.Accuracy)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./train/... -run TestSyntheticCNNConverges -v`
Expected: FAIL only if any Phase 1–3 module has a wiring bug not caught by its own unit/gradient-check tests — this test exercises the full Conv2D→ReLU→MaxPool2D→Flatten→Linear→Sigmoid chain end-to-end through `Trainer.Fit`, something no single earlier task's tests do together. If it fails, treat it as a real bug (likely a shape or indexing mismatch between two adjacent modules) — do not weaken the assertion.

- [ ] **Step 3: There is no separate "implementation" step**

Every module this test exercises already exists from Phase 1–3. If the test in Step 2 didn't fail, skip straight to Step 4. If it did fail, fix the offending module (with its own gradient-check test as your guide) before proceeding — do not modify this test's tolerance to paper over a real bug.

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./train/... -run TestSyntheticCNNConverges -v`
Expected: PASS

- [ ] **Step 5: Run the full nn and train suites**

Run: `go build ./... && go vet ./... && go test ./nn/... ./train/... -v`
Expected: everything from Tasks 1–20 green.

- [ ] **Step 6: Delete `Network/` and commit**

`Network/` is now fully superseded: `nn` + `train` cover every capability it had (plus the fixes and new features from design doc §1/§3), proven by the test suite that just passed. `data/` does not yet compile independently of this deletion (Task 22 fixes it) and does not import `Network`, so this deletion is safe on its own.

```bash
git rm -r Network/
go build ./...
```

Expected: `go build ./...` still succeeds (the `data` package's pre-existing compile errors, fixed in Task 22, are unrelated to `Network/` and already present before this deletion — confirm with `go build ./nn/... ./train/...` if `go build ./...` still fails only inside `data/`).

```bash
git add -A
git commit -m "feat: delete Network/, fully superseded by nn/ and train/"
```

---

## Phase 4: Serialization + data + repo

`serialize.go`, `data` package fixes + tests, examples rewrite, docs consolidation, `.gitignore`, artifact cleanup. Full `go build ./... && go vet ./... && go test ./...` green.

### Task 21: Serialization

**Files:**
- Create: `nn/serialize.go`
- Test: `nn/serialize_test.go`

**Interfaces:**
- Consumes: every `nn.Module` concrete type from Phase 1 and Phase 3 (`LinearLayer`, `Conv2DLayer`, `MaxPool2DLayer`, `AvgPool2DLayer`, `FlattenLayer`, `DropoutLayer`, `BatchNormLayer`, `ActivationModule` with its `Name()`/`Alpha()` from Task 19, `SoftmaxModule`, `SequentialModel`).
- Produces: `func Save(model *SequentialModel, path string) error`, `func Load(path string) (*SequentialModel, error)`.

A JSON document is a tree of `{type, config, params, modules}` nodes, one node per module, with `"sequential"` nodes nesting children under `modules` — this is what makes save/load work identically for a flat dense model and a `Conv2D`-containing one, per design doc §4.4. RNG seed and optimizer state are never serialized (training-resume is explicitly out of scope, design doc §4.4); `BatchNorm`'s running mean/variance **are** serialized (config-level, not `Params()`) because omitting them would make a loaded model's `Inference`-mode output silently wrong — the design doc doesn't call this out explicitly, but it follows directly from §4.2's requirement that running stats are "used in Inference."

- [ ] **Step 1: Write the failing tests**

```go
package nn

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadDenseModelRoundTrip(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 3},
		Linear(rng, 3, 4, XavierInit()),
		ReLU(),
		BatchNorm(4),
		Dropout(0.2),
		Linear(rng, 4, 2, XavierInit()),
		Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	// Run one Train-mode forward so BatchNorm accumulates non-trivial running stats.
	ctx := &Context{Mode: Train, RNG: NewRNG(2)}
	x := NewTensor([]int{2, 3})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.3
	}
	if _, err := model.Forward(ctx, x); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "model.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	infCtx := &Context{Mode: Inference}
	want, err := model.Forward(infCtx, x)
	if err != nil {
		t.Fatalf("Forward original: %v", err)
	}
	got, err := loaded.Forward(infCtx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v (loaded model diverges from original)", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadConvModelRoundTrip(t *testing.T) {
	rng := NewRNG(3)
	model, err := Sequential([]int{1, 6, 6, 1},
		Conv2D(rng, 1, 2, 3, HeInit()),
		ReLU(),
		MaxPool2D(2, 2),
		Flatten(),
		Linear(rng, 0, 1, XavierInit()),
		Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 6, 6, 1})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "cnn.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestLoadMissingFileReturnsError(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "does-not-exist.json")); err == nil {
		t.Fatal("expected error loading a missing file, got nil")
	}
}

func TestLoadRejectsMalformedJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("not json"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading malformed JSON, got nil")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./nn/... -run "TestSaveLoad|TestLoadMissing|TestLoadRejects" -v`
Expected: FAIL — `Save`/`Load` undefined.

- [ ] **Step 3: Write minimal implementation**

```go
package nn

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
)

type paramDoc struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

func toParamDoc(p *Param) paramDoc { return paramDoc{Shape: p.Value.Shape, Data: p.Value.Data} }

type moduleDoc struct {
	Type    string               `json:"type"`
	Config  json.RawMessage      `json:"config,omitempty"`
	Params  map[string]paramDoc  `json:"params,omitempty"`
	Modules []moduleDoc          `json:"modules,omitempty"`
}

type linearConfig struct {
	InFeatures  int `json:"in_features"`
	OutFeatures int `json:"out_features"`
}

type conv2DConfig struct {
	InChannels  int `json:"in_channels"`
	OutChannels int `json:"out_channels"`
	KernelSize  int `json:"kernel_size"`
	Padding     int `json:"padding"`
}

type poolConfig struct {
	PoolSize int `json:"pool_size"`
	Stride   int `json:"stride"`
}

type dropoutConfig struct {
	Rate float32 `json:"rate"`
}

type batchNormConfig struct {
	Channels    int       `json:"channels"`
	RunningMean []float32 `json:"running_mean"`
	RunningVar  []float32 `json:"running_var"`
}

type leakyReLUConfig struct {
	Alpha float32 `json:"alpha"`
}

func mustMarshal(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

func encodeModule(m Module) (moduleDoc, error) {
	switch v := m.(type) {
	case *LinearLayer:
		return moduleDoc{
			Type:   "linear",
			Config: mustMarshal(linearConfig{InFeatures: v.inFeatures, OutFeatures: v.outFeatures}),
			Params: map[string]paramDoc{"W": toParamDoc(v.W), "B": toParamDoc(v.B)},
		}, nil
	case *Conv2DLayer:
		return moduleDoc{
			Type:   "conv2d",
			Config: mustMarshal(conv2DConfig{InChannels: v.inChannels, OutChannels: v.outChannels, KernelSize: v.kernelSize, Padding: v.padding}),
			Params: map[string]paramDoc{"W": toParamDoc(v.W), "B": toParamDoc(v.B)},
		}, nil
	case *MaxPool2DLayer:
		return moduleDoc{Type: "maxpool2d", Config: mustMarshal(poolConfig{PoolSize: v.poolSize, Stride: v.stride})}, nil
	case *AvgPool2DLayer:
		return moduleDoc{Type: "avgpool2d", Config: mustMarshal(poolConfig{PoolSize: v.poolSize, Stride: v.stride})}, nil
	case *FlattenLayer:
		return moduleDoc{Type: "flatten"}, nil
	case *DropoutLayer:
		return moduleDoc{Type: "dropout", Config: mustMarshal(dropoutConfig{Rate: v.rate})}, nil
	case *BatchNormLayer:
		return moduleDoc{
			Type:   "batchnorm",
			Config: mustMarshal(batchNormConfig{Channels: v.channels, RunningMean: v.runningMean, RunningVar: v.runningVar}),
			Params: map[string]paramDoc{"gamma": toParamDoc(v.Gamma), "beta": toParamDoc(v.Beta)},
		}, nil
	case *SoftmaxModule:
		return moduleDoc{Type: "softmax"}, nil
	case *ActivationModule:
		if v.Name() == "leaky_relu" {
			return moduleDoc{Type: "leaky_relu", Config: mustMarshal(leakyReLUConfig{Alpha: v.Alpha()})}, nil
		}
		return moduleDoc{Type: v.Name()}, nil
	case *SequentialModel:
		children := make([]moduleDoc, len(v.modules))
		for i, cm := range v.modules {
			cd, err := encodeModule(cm)
			if err != nil {
				return moduleDoc{}, err
			}
			children[i] = cd
		}
		return moduleDoc{Type: "sequential", Modules: children}, nil
	default:
		return moduleDoc{}, fmt.Errorf("nn: Save: unsupported module type %T", m)
	}
}

// Save writes model as a JSON document — a tree of {type, config, params,
// modules} nodes readable by Load. RNG seed and optimizer state are never
// included (training-resume is out of scope).
func Save(model *SequentialModel, path string) error {
	doc, err := encodeModule(model)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return fmt.Errorf("nn: Save: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("nn: Save: %w", err)
	}
	return nil
}

func paramOrErr(doc moduleDoc, key, moduleType string) (paramDoc, error) {
	p, ok := doc.Params[key]
	if !ok {
		return paramDoc{}, fmt.Errorf("nn: Load: %s module missing %q param", moduleType, key)
	}
	return p, nil
}

func decodeModule(doc moduleDoc, rng *rand.Rand) (Module, error) {
	switch doc.Type {
	case "linear":
		var cfg linearConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: linear config: %w", err)
		}
		l := Linear(rng, cfg.InFeatures, cfg.OutFeatures, ZerosInit())
		w, err := paramOrErr(doc, "W", "linear")
		if err != nil {
			return nil, err
		}
		b, err := paramOrErr(doc, "B", "linear")
		if err != nil {
			return nil, err
		}
		copy(l.W.Value.Data, w.Data)
		copy(l.B.Value.Data, b.Data)
		return l, nil

	case "conv2d":
		var cfg conv2DConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: conv2d config: %w", err)
		}
		c := newConv2D(rng, cfg.InChannels, cfg.OutChannels, cfg.KernelSize, cfg.Padding, ZerosInit())
		w, err := paramOrErr(doc, "W", "conv2d")
		if err != nil {
			return nil, err
		}
		b, err := paramOrErr(doc, "B", "conv2d")
		if err != nil {
			return nil, err
		}
		copy(c.W.Value.Data, w.Data)
		copy(c.B.Value.Data, b.Data)
		return c, nil

	case "maxpool2d":
		var cfg poolConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: maxpool2d config: %w", err)
		}
		return MaxPool2D(cfg.PoolSize, cfg.Stride), nil

	case "avgpool2d":
		var cfg poolConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: avgpool2d config: %w", err)
		}
		return AvgPool2D(cfg.PoolSize, cfg.Stride), nil

	case "flatten":
		return Flatten(), nil

	case "dropout":
		var cfg dropoutConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: dropout config: %w", err)
		}
		return Dropout(cfg.Rate), nil

	case "batchnorm":
		var cfg batchNormConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: batchnorm config: %w", err)
		}
		bn := BatchNorm(cfg.Channels)
		copy(bn.runningMean, cfg.RunningMean)
		copy(bn.runningVar, cfg.RunningVar)
		g, err := paramOrErr(doc, "gamma", "batchnorm")
		if err != nil {
			return nil, err
		}
		beta, err := paramOrErr(doc, "beta", "batchnorm")
		if err != nil {
			return nil, err
		}
		copy(bn.Gamma.Value.Data, g.Data)
		copy(bn.Beta.Value.Data, beta.Data)
		return bn, nil

	case "softmax":
		return Softmax(), nil
	case "relu":
		return ReLU(), nil
	case "sigmoid":
		return Sigmoid(), nil
	case "tanh":
		return Tanh(), nil
	case "gelu":
		return GELU(), nil
	case "leaky_relu":
		var cfg leakyReLUConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: leaky_relu config: %w", err)
		}
		return LeakyReLU(cfg.Alpha), nil

	case "sequential":
		children := make([]Module, len(doc.Modules))
		for i, cd := range doc.Modules {
			cm, err := decodeModule(cd, rng)
			if err != nil {
				return nil, err
			}
			children[i] = cm
		}
		return &SequentialModel{modules: children}, nil

	default:
		return nil, fmt.Errorf("nn: Load: unknown module type %q", doc.Type)
	}
}

// Load reads a JSON document written by Save and reconstructs the module
// tree with its trained weights. The weight-init RNG passed to
// constructors during reconstruction is never actually used for
// randomness — every Param is immediately overwritten from the saved
// data — so a fixed throwaway seed is fine here.
func Load(path string) (*SequentialModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("nn: Load: %w", err)
	}
	var doc moduleDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("nn: Load: %w", err)
	}
	m, err := decodeModule(doc, NewRNG(0))
	if err != nil {
		return nil, err
	}
	seq, ok := m.(*SequentialModel)
	if !ok {
		return nil, fmt.Errorf("nn: Load: root module has type %q, want \"sequential\"", doc.Type)
	}
	return seq, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./nn/... -run "TestSaveLoad|TestLoadMissing|TestLoadRejects" -v`
Expected: PASS

- [ ] **Step 5: Run the full nn package suite**

Run: `go test ./nn/... -v`
Expected: PASS (all tests from Tasks 1–21)

- [ ] **Step 6: Wire `ModelCheckpoint` in the examples that use it**

Task 10 left `ModelCheckpointCallback.Save` unset by default; any example that wants real checkpointing (Task 23's `callbacks` example) passes `train.WithSaveFunc(nn.Save)` to `Fit`. No further code change is needed here — this step is just the reminder that Task 21 is what makes that wiring meaningful.

- [ ] **Step 7: Commit**

```bash
git add nn/serialize.go nn/serialize_test.go
git commit -m "feat(nn): add Save/Load with a type-registry JSON format for any module tree"
```

### Task 22: Fix the `data` package; delete `tensor/`

**Files:**
- Modify: `data/image.go`, `data/cifar10.go`, `data/preprocessing.go`, `data/balancing.go`, `data/doc.go`
- Delete: `tensor/` (whole directory)
- Test: `data/image_test.go`, `data/preprocessing_test.go`, `data/balancing_test.go` (new)

**Interfaces:**
- Produces: `type Image struct { Data [][][]float32; Height, Width, Channels int }` and `func NewImage(height, width, channels int) *Image` in `data/image.go`, replacing every use of the deleted `neugo/tensor` package's `*tensor.Tensor3D`. Signature changes: `func ShuffleData(rng *rand.Rand, features, labels [][]float32) ([][]float32, [][]float32)`, `func SplitData(rng *rand.Rand, features, labels [][]float32, config SplitConfig) Split`, `func OversampleMinorityClass(rng *rand.Rand, features, labels [][]float32, config OversampleConfig) ([][]float32, [][]float32)`, `func UndersampleMajorityClass(rng *rand.Rand, features, labels [][]float32, config UndersampleConfig) ([][]float32, [][]float32)`, `func BalanceDataset(rng *rand.Rand, features, labels [][]float32, targetRatio float64, preferOversample bool) ([][]float32, [][]float32)`, `func SplitImageData(rng *rand.Rand, images []*Image, labels [][]float32, config SplitConfig) ImageSplit`.

Four separate problems, all fixed in this one task since they're all in the same small package and block each other from compiling:

1. **Compile errors** (design doc §1): unused `hasHeader` var in `LoadMNISTFromCSV`/`LoadBinaryImageFromCSV`; a `float64 >= float32` comparison in `LoadBinaryImageFromCSV`; undefined `NewRNG` in `shuffleImageData`; unused `"encoding/binary"` and `"fmt"` imports in `cifar10.go`.
2. **Hand-rolled `sqrt`** in `cifar10.go` (`NormalizeImages`'s helper) instead of `math.Sqrt`.
3. **Global `rand.Seed`** (deprecated) in `preprocessing.go` and `balancing.go` — replaced by explicit `*rand.Rand` parameters per design decision #1's project-wide "no global RNG state" rule, applied here exactly as design doc §4.5 specifies.
4. **`neugo/tensor` dependency**: `data` cannot import `nn` (design doc §4.1: "`data` is standalone, no dependency on `nn`/`train`"), so replacing `*tensor.Tensor3D` with `*nn.Tensor` is not an option — `tensor/` is being deleted as fully superseded by `nn/tensor.go`, but `data` still needs *some* image representation. The fix is a small package-local `Image` type in `data/image.go`, laid out `[height][width][channel]` (channel-last) specifically so any caller that also imports `nn` — i.e. the `examples/` package, never `data` itself — can trivially stack a `[]*Image` into a batched `*nn.Tensor` of shape `[n, h, w, c]` (Task 23 writes that glue).

- [ ] **Step 1: Write the failing tests**

```go
// data/image_test.go
package data

import "testing"

func TestNewImageShape(t *testing.T) {
	img := NewImage(4, 5, 3)
	if img.Height != 4 || img.Width != 5 || img.Channels != 3 {
		t.Fatalf("NewImage shape = (%d,%d,%d), want (4,5,3)", img.Height, img.Width, img.Channels)
	}
	if len(img.Data) != 4 || len(img.Data[0]) != 5 || len(img.Data[0][0]) != 3 {
		t.Fatalf("Data dims = (%d,%d,%d), want (4,5,3)", len(img.Data), len(img.Data[0]), len(img.Data[0][0]))
	}
}

func TestSplitImageDataRatios(t *testing.T) {
	images := make([]*Image, 10)
	labels := make([][]float32, 10)
	for i := range images {
		images[i] = NewImage(2, 2, 1)
		labels[i] = []float32{float32(i)}
	}
	rng := newTestRNG(1)
	split := SplitImageData(rng, images, labels, SplitConfig{TrainRatio: 0.6, ValRatio: 0.2, TestRatio: 0.2, Shuffle: true})
	if len(split.TrainX) != 6 || len(split.ValX) != 2 || len(split.TestX) != 2 {
		t.Fatalf("split sizes = (%d,%d,%d), want (6,2,2)", len(split.TrainX), len(split.ValX), len(split.TestX))
	}
}
```

```go
// data/preprocessing_test.go
package data

import (
	"math/rand"
	"testing"
)

func newTestRNG(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

func TestShuffleDataIsDeterministicPerRNG(t *testing.T) {
	x := [][]float32{{1}, {2}, {3}, {4}, {5}}
	y := [][]float32{{10}, {20}, {30}, {40}, {50}}
	sx1, sy1 := ShuffleData(newTestRNG(7), x, y)
	sx2, sy2 := ShuffleData(newTestRNG(7), x, y)
	for i := range sx1 {
		if sx1[i][0] != sx2[i][0] || sy1[i][0] != sy2[i][0] {
			t.Fatalf("ShuffleData with the same seed produced different orders at index %d", i)
		}
	}
}

func TestSplitDataRatios(t *testing.T) {
	n := 100
	x := make([][]float32, n)
	y := make([][]float32, n)
	for i := 0; i < n; i++ {
		x[i] = []float32{float32(i)}
		y[i] = []float32{float32(i)}
	}
	split := SplitData(newTestRNG(1), x, y, SplitConfig{TrainRatio: 0.7, ValRatio: 0.15, TestRatio: 0.15, Shuffle: true})
	if len(split.TrainX) != 70 || len(split.ValX) != 15 || len(split.TestX) != 15 {
		t.Fatalf("split sizes = (%d,%d,%d), want (70,15,15)", len(split.TrainX), len(split.ValX), len(split.TestX))
	}
}
```

```go
// data/balancing_test.go
package data

import "testing"

func TestBalanceDatasetOversampleIncreasesMinorityCount(t *testing.T) {
	// 9 majority (label 0), 1 minority (label 1)
	features := make([][]float32, 10)
	labels := make([][]float32, 10)
	for i := 0; i < 10; i++ {
		features[i] = []float32{float32(i)}
		if i == 0 {
			labels[i] = []float32{1}
		} else {
			labels[i] = []float32{0}
		}
	}
	bx, by := BalanceDataset(newTestRNG(1), features, labels, 0.4, true)
	if len(bx) != len(by) {
		t.Fatalf("balanced features/labels length mismatch: %d vs %d", len(bx), len(by))
	}
	var minority int
	for _, l := range by {
		if l[0] > 0.5 {
			minority++
		}
	}
	dist := AnalyzeClassDistribution(by, 0.5)
	if !dist.IsBalanced && minority <= 1 {
		t.Fatalf("BalanceDataset did not increase minority count: got %d minority samples out of %d", minority, len(by))
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go build ./data/... && go test ./data/... -v`
Expected: FAIL — the package does not currently compile (this is the pre-existing state the design doc's audit describes), so both the build and the new tests fail.

- [ ] **Step 3: Fix `data/image.go`**

Replace the `tensor` import and `ImageDataset.Images` type, and delete the unused `hasHeader` bookkeeping:

```go
package data

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

// Image is a package-local, channel-last (matching nn.Tensor's [batch, h,
// w, c] convention) image representation. data cannot import nn (design
// doc §4.1), so this exists instead of the deleted tensor.Tensor3D —
// callers that also import nn (i.e. examples/, never data itself) stack a
// []*Image into a batched *nn.Tensor.
type Image struct {
	Data                     [][][]float32 // [height][width][channel]
	Height, Width, Channels  int
}

func NewImage(height, width, channels int) *Image {
	data := make([][][]float32, height)
	for h := range data {
		data[h] = make([][]float32, width)
		for w := range data[h] {
			data[h][w] = make([]float32, channels)
		}
	}
	return &Image{Data: data, Height: height, Width: width, Channels: channels}
}

type ImageDataset struct {
	Images   []*Image
	Labels   [][]float32
	Height   int
	Width    int
	Channels int
}

func LoadMNISTFromCSV(filepath string) (*ImageDataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	startIdx := 0
	if _, err := strconv.ParseFloat(records[0][0], 32); err != nil {
		startIdx = 1
	}

	numSamples := len(records) - startIdx
	images := make([]*Image, numSamples)
	labels := make([][]float32, numSamples)
	height, width, channels := 28, 28, 1

	for i := startIdx; i < len(records); i++ {
		record := records[i]
		label, err := strconv.ParseFloat(record[0], 32)
		if err != nil {
			return nil, fmt.Errorf("invalid label at row %d: %v", i, err)
		}
		labels[i-startIdx] = make([]float32, 10)
		labels[i-startIdx][int(label)] = 1.0

		img := NewImage(height, width, channels)
		for pixelIdx := 1; pixelIdx < len(record) && pixelIdx <= height*width; pixelIdx++ {
			pixelValue, err := strconv.ParseFloat(record[pixelIdx], 32)
			if err != nil {
				return nil, fmt.Errorf("invalid pixel at row %d, col %d: %v", i, pixelIdx, err)
			}
			h := (pixelIdx - 1) / width
			w := (pixelIdx - 1) % width
			img.Data[h][w][0] = float32(pixelValue) / 255.0
		}
		images[i-startIdx] = img
	}

	return &ImageDataset{Images: images, Labels: labels, Height: height, Width: width, Channels: channels}, nil
}

func LoadBinaryImageFromCSV(filepath string, threshold float32) (*ImageDataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	startIdx := 0
	if _, err := strconv.ParseFloat(records[0][0], 32); err != nil {
		startIdx = 1
	}

	numSamples := len(records) - startIdx
	images := make([]*Image, numSamples)
	labels := make([][]float32, numSamples)
	height, width, channels := 28, 28, 1

	for i := startIdx; i < len(records); i++ {
		record := records[i]
		label, err := strconv.ParseFloat(record[0], 32)
		if err != nil {
			return nil, fmt.Errorf("invalid label at row %d: %v", i, err)
		}
		binaryLabel := float32(0)
		if float32(label) >= threshold { // was `label >= threshold`: float64 vs float32, would not compile
			binaryLabel = 1
		}
		labels[i-startIdx] = []float32{binaryLabel}

		img := NewImage(height, width, channels)
		for pixelIdx := 1; pixelIdx < len(record) && pixelIdx <= height*width; pixelIdx++ {
			pixelValue, err := strconv.ParseFloat(record[pixelIdx], 32)
			if err != nil {
				return nil, fmt.Errorf("invalid pixel at row %d, col %d: %v", i, pixelIdx, err)
			}
			h := (pixelIdx - 1) / width
			w := (pixelIdx - 1) % width
			img.Data[h][w][0] = float32(pixelValue) / 255.0
		}
		images[i-startIdx] = img
	}

	return &ImageDataset{Images: images, Labels: labels, Height: height, Width: width, Channels: channels}, nil
}

type ImageSplit struct {
	TrainX []*Image
	TrainY [][]float32
	ValX   []*Image
	ValY   [][]float32
	TestX  []*Image
	TestY  [][]float32
}

func SplitImageData(rng *rand.Rand, images []*Image, labels [][]float32, config SplitConfig) ImageSplit {
	numSamples := len(images)
	if config.Shuffle {
		images, labels = shuffleImageData(rng, images, labels)
	}
	trainEnd := int(float64(numSamples) * config.TrainRatio)
	valEnd := trainEnd + int(float64(numSamples)*config.ValRatio)
	return ImageSplit{
		TrainX: images[:trainEnd], TrainY: labels[:trainEnd],
		ValX: images[trainEnd:valEnd], ValY: labels[trainEnd:valEnd],
		TestX: images[valEnd:], TestY: labels[valEnd:],
	}
}

func shuffleImageData(rng *rand.Rand, images []*Image, labels [][]float32) ([]*Image, [][]float32) {
	n := len(images)
	shuffledImages := make([]*Image, n)
	shuffledLabels := make([][]float32, n)
	for i, idx := range rng.Perm(n) {
		shuffledImages[i] = images[idx]
		shuffledLabels[i] = labels[idx]
	}
	return shuffledImages, shuffledLabels
}
```

`SplitConfig.Seed` (used elsewhere by `SplitData`) is removed in Step 4 below — `SplitImageData` above never referenced it even before this fix, so no further change is needed here.

- [ ] **Step 4: Fix `data/preprocessing.go`**

Replace `ShuffleData` and `SplitData`, and drop the now-unused `Seed` field:

```go
// ShuffleData shuffles features and labels in unison using rng.
func ShuffleData(rng *rand.Rand, features [][]float32, labels [][]float32) ([][]float32, [][]float32) {
	n := len(features)
	indices := rng.Perm(n)
	shuffledX := make([][]float32, n)
	shuffledY := make([][]float32, n)
	for i, idx := range indices {
		shuffledX[i] = features[idx]
		shuffledY[i] = labels[idx]
	}
	return shuffledX, shuffledY
}

// SplitConfig holds configuration for data splitting. Seeding is explicit
// via SplitData's rng parameter, not a field here.
type SplitConfig struct {
	TrainRatio float64
	ValRatio   float64
	TestRatio  float64
	Shuffle    bool
}

// SplitData splits data into train/validation/test sets.
func SplitData(rng *rand.Rand, features [][]float32, labels [][]float32, config SplitConfig) Split {
	total := config.TrainRatio + config.ValRatio + config.TestRatio
	if math.Abs(total-1.0) > 0.01 && config.TestRatio == 0 {
		config.TestRatio = 1.0 - config.TrainRatio - config.ValRatio
	}
	if config.Shuffle {
		features, labels = ShuffleData(rng, features, labels)
	}
	n := len(features)
	trainSize := int(float64(n) * config.TrainRatio)
	valSize := int(float64(n) * config.ValRatio)
	return Split{
		TrainX: features[:trainSize], TrainY: labels[:trainSize],
		ValX: features[trainSize : trainSize+valSize], ValY: labels[trainSize : trainSize+valSize],
		TestX: features[trainSize+valSize:], TestY: labels[trainSize+valSize:],
	}
}
```

Remove the `"time"` import (no longer needed — it was only used by the deleted `rand.Seed(time.Now().UnixNano())` fallback) and add `"math/rand"` in its place; keep the existing `"math"` import for `math.Abs`.

- [ ] **Step 5: Fix `data/balancing.go`**

Replace `OversampleMinorityClass`, `UndersampleMajorityClass`, `BalanceDataset`, and drop the now-unused `Seed` fields from `OversampleConfig`/`UndersampleConfig`:

```go
type OversampleConfig struct {
	TargetRatio float64
	Strategy    string // "duplicate" or "random"
}

func OversampleMinorityClass(rng *rand.Rand, features [][]float32, labels [][]float32, config OversampleConfig) ([][]float32, [][]float32) {
	balancedX := append([][]float32{}, features...)
	balancedY := append([][]float32{}, labels...)

	minorityIndices := make([]int, 0)
	for i := range labels {
		if labels[i][0] > 0.5 {
			minorityIndices = append(minorityIndices, i)
		}
	}
	targetMinorityCount := int(float64(len(labels)) / (1 - config.TargetRatio) * config.TargetRatio)
	samplesToAdd := targetMinorityCount - len(minorityIndices)

	if samplesToAdd > 0 && len(minorityIndices) > 0 {
		if config.Strategy == "random" {
			for added := 0; added < samplesToAdd; added++ {
				idx := minorityIndices[rng.Intn(len(minorityIndices))]
				balancedX = append(balancedX, features[idx])
				balancedY = append(balancedY, labels[idx])
			}
		} else {
			for added := 0; added < samplesToAdd; added++ {
				idx := minorityIndices[added%len(minorityIndices)]
				balancedX = append(balancedX, features[idx])
				balancedY = append(balancedY, labels[idx])
			}
		}
	}
	return balancedX, balancedY
}

type UndersampleConfig struct {
	TargetRatio float64
	Strategy    string // "random" or "systematic"
}

func UndersampleMajorityClass(rng *rand.Rand, features [][]float32, labels [][]float32, config UndersampleConfig) ([][]float32, [][]float32) {
	var minorityX, minorityY, majorityX, majorityY [][]float32
	for i := range labels {
		if labels[i][0] > 0.5 {
			minorityX = append(minorityX, features[i])
			minorityY = append(minorityY, labels[i])
		} else {
			majorityX = append(majorityX, features[i])
			majorityY = append(majorityY, labels[i])
		}
	}
	targetMajorityCount := int(float64(len(minorityY)) * (1 - config.TargetRatio) / config.TargetRatio)

	if targetMajorityCount < len(majorityY) {
		if config.Strategy == "random" {
			indices := rng.Perm(len(majorityY))[:targetMajorityCount]
			sampledX := make([][]float32, targetMajorityCount)
			sampledY := make([][]float32, targetMajorityCount)
			for i, idx := range indices {
				sampledX[i] = majorityX[idx]
				sampledY[i] = majorityY[idx]
			}
			majorityX, majorityY = sampledX, sampledY
		} else {
			step := len(majorityY) / targetMajorityCount
			var sampledX, sampledY [][]float32
			for i := 0; i < len(majorityY) && len(sampledY) < targetMajorityCount; i += step {
				sampledX = append(sampledX, majorityX[i])
				sampledY = append(sampledY, majorityY[i])
			}
			majorityX, majorityY = sampledX, sampledY
		}
	}
	return append(minorityX, majorityX...), append(minorityY, majorityY...)
}

func BalanceDataset(rng *rand.Rand, features [][]float32, labels [][]float32, targetRatio float64, preferOversample bool) ([][]float32, [][]float32) {
	dist := AnalyzeClassDistribution(labels, 0.5)
	if dist.IsBalanced {
		return features, labels
	}
	if preferOversample {
		return OversampleMinorityClass(rng, features, labels, OversampleConfig{TargetRatio: targetRatio, Strategy: "duplicate"})
	}
	return UndersampleMajorityClass(rng, features, labels, UndersampleConfig{TargetRatio: targetRatio, Strategy: "random"})
}
```

Remove the `"time"` import from this file too (same reason as Step 4).

- [ ] **Step 6: Fix `data/cifar10.go`**

Remove the unused `"encoding/binary"` and `"fmt"` imports, replace `*tensor.Tensor3D`/`tensor.NewTensor3D` with `*Image`/`NewImage` (channel-last: `img.Data[h][w][c]` instead of the old `img.Data[c][h][w]`), and replace the hand-rolled `sqrt` with `math.Sqrt`:

```go
package data

import (
	"io"
	"math"
	"os"
)

type CIFAR10Dataset struct {
	Images     []*Image
	Labels     [][]float32
	ClassNames []string
}
```

In `LoadCIFAR10Binary`, `LoadCIFAR10BinaryBatch`, and `LoadCIFAR10BinaryClassSubset`, replace every `img := tensor.NewTensor3D(32, 32, 3)` with `img := NewImage(32, 32, 3)` and every `img.Data[c][h][w] = ...` with `img.Data[h][w][c] = ...` (same value expression, just channel-last indexing — the pixel-unpacking arithmetic `idx := 1 + c*1024 + h*32 + w` is unchanged since the *source* CIFAR-10 binary format is channel-first regardless of how this codebase stores the result).

`NormalizeImages` becomes:

```go
func NormalizeImages(images []*Image) []*Image {
	if len(images) == 0 {
		return images
	}
	numChannels, height, width := images[0].Channels, images[0].Height, images[0].Width
	means := make([]float32, numChannels)
	stds := make([]float32, numChannels)

	for c := 0; c < numChannels; c++ {
		var sum float32
		count := 0
		for _, img := range images {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					sum += img.Data[h][w][c]
					count++
				}
			}
		}
		means[c] = sum / float32(count)

		var variance float32
		for _, img := range images {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					diff := img.Data[h][w][c] - means[c]
					variance += diff * diff
				}
			}
		}
		stds[c] = float32(math.Sqrt(float64(variance / float32(count))))
	}

	normalized := make([]*Image, len(images))
	for i, img := range images {
		normalized[i] = NewImage(height, width, numChannels)
		for c := 0; c < numChannels; c++ {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					if stds[c] > 0 {
						normalized[i].Data[h][w][c] = (img.Data[h][w][c] - means[c]) / stds[c]
					} else {
						normalized[i].Data[h][w][c] = img.Data[h][w][c] - means[c]
					}
				}
			}
		}
	}
	return normalized
}
```

Delete the file-local `sqrt(x float64) float64` function entirely — `math.Sqrt` replaces it.

- [ ] **Step 7: Update `data/doc.go`**

Update every code sample that calls `ShuffleData`, `SplitData`, `OversampleMinorityClass`, `UndersampleMajorityClass`, or `BalanceDataset` to pass an explicit `rng` first argument (e.g. `rng := data_examplePkgRand()` — in the real doc comment, show `rng := rand.New(rand.NewSource(42))` and thread it through each call), and drop every `Seed: 42` config literal field now that the field no longer exists. Delete the stale "Complete Example" section at the bottom that references the deleted `neugo/Network` package entirely — Task 23 provides real, buildable, up-to-date examples instead of a doc-comment sketch of one.

- [ ] **Step 8: Run tests to verify they pass**

Run: `go build ./data/... && go vet ./data/... && go test ./data/... -v`
Expected: PASS — this is the first time `data` has compiled at all in this codebase's history.

- [ ] **Step 9: Delete `tensor/`**

`data` no longer imports it (Step 3), and `Network/` (its only other consumer) was deleted in Task 20.

```bash
git rm -r tensor/
go build ./...
```

Expected: `go build ./...` succeeds across the whole repository for the first time in this plan.

- [ ] **Step 10: Commit**

```bash
git add data/ -A
git commit -m "fix(data): fix compile errors, explicit RNG, drop tensor/ dependency; delete tensor/"
```

### Task 23: Rewrite examples as six buildable directories

**Files:**
- Create: `examples/xor/main.go`, `examples/wine_quality/main.go`, `examples/fashion_mnist/main.go`, `examples/cifar10_cnn/main.go`, `examples/callbacks/main.go`, `examples/crossval/main.go`
- Delete: every existing flat file in `examples/` (`train.go`, `test_activations.go`, `test_losses.go`, `example_usage.go`, `demo_phase1_phase2.go`, `showcase.go`, `fraud_detection.go`, `wine_quality.go`, `wine_quality_clean.go`, `phase4_demo.go`, `debug_training.go`, `complete_showcase.go`, `check_predictions.go`, `cnn_demo.go`, `cnn_showcase.go`, `cifar10_demo.go`, `fashion_mnist_demo.go`, `cats_vs_dogs_synthetic.go`, `clean_api_demo.go`, `functional_demo.go`, `nnx_demo.go`) — 21 files, all `package main`, not buildable as one directory, several importing the old broken `Network`/`data` API.
- Delete: root `main.go`.

**Interfaces:**
- Consumes: the complete `nn`/`train`/`data` public API from every prior task.

Real datasets already exist in this repo at `dataset/wine_quality/winequality-red.csv` (1599 rows, `;`-delimited, 11 features + integer `quality` label) — used directly by `wine_quality`. No Fashion-MNIST CSV or CIFAR-10 binary batches are committed (deliberately — repo hygiene, design doc §4.8), so `fashion_mnist` and `cifar10_cnn` try a conventional path first and fall back to a small deterministic synthetic dataset of the right shape, printing which one they used — this keeps every example runnable with zero setup while still exercising the real `data.LoadMNISTFromCSV`/`data.LoadCIFAR10Binary` code paths whenever real data is present.

- [ ] **Step 1: `examples/xor/main.go`**

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
		fmt.Println("build model:", err)
		return
	}
	summary, _ := nn.Summary(model, []int{4, 2})
	fmt.Print(summary)

	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := train.New(model, train.Adam(0.05, 0.9, 0.999, 1e-8), train.BCELoss())
	hist, err := trainer.Fit(x, y, train.Epochs(2000), train.BatchSize(4), train.Seed(1))
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("final train loss: %.4f\n", hist.TrainLoss[len(hist.TrainLoss)-1])

	preds, err := trainer.Predict(x)
	if err != nil {
		fmt.Println("predict:", err)
		return
	}
	inputs := [][2]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for i, in := range inputs {
		fmt.Printf("XOR(%v, %v) = %.4f\n", in[0], in[1], preds.Data[i])
	}
}
```

- [ ] **Step 2: `examples/wine_quality/main.go`**

```go
package main

import (
	"fmt"
	"neugo/data"
	"neugo/nn"
	"neugo/train"
)

func toTensor(rows [][]float32) *nn.Tensor {
	cols := len(rows[0])
	flat := make([]float32, len(rows)*cols)
	for i, row := range rows {
		copy(flat[i*cols:(i+1)*cols], row)
	}
	t, _ := nn.NewTensorFromData(flat, []int{len(rows), cols})
	return t
}

func main() {
	dataset, err := data.LoadCSV("dataset/wine_quality/winequality-red.csv", data.CSVConfig{
		Delimiter:       ';',
		HasHeader:       true,
		LabelColumn:     -1,
		LabelType:       "binary",
		BinaryThreshold: 6.0,
	})
	if err != nil {
		fmt.Println("load csv:", err)
		return
	}

	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	rng := nn.NewRNG(1)
	dataRNG := nn.NewRNG(2)
	split := data.SplitData(dataRNG, normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7, ValRatio: 0.15, TestRatio: 0.15, Shuffle: true,
	})

	model, err := nn.Sequential([]int{1, dataset.NumFeatures},
		nn.Linear(rng, dataset.NumFeatures, 16, nn.HeInit()),
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(rng, 16, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}

	trainer := train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.BCELoss())
	hist, err := trainer.Fit(
		toTensor(split.TrainX), toTensor(split.TrainY),
		train.Epochs(100), train.BatchSize(32), train.Shuffle(true), train.Seed(3),
		train.Validation(toTensor(split.ValX), toTensor(split.ValY)),
		train.Callbacks(train.EarlyStopping(10)),
	)
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("trained %d epochs, final train loss %.4f\n", len(hist.TrainLoss), hist.TrainLoss[len(hist.TrainLoss)-1])

	metrics, err := trainer.Evaluate(toTensor(split.TestX), toTensor(split.TestY))
	if err != nil {
		fmt.Println("evaluate:", err)
		return
	}
	fmt.Printf("test accuracy: %.2f%%  f1: %.4f\n", metrics.Accuracy, metrics.F1Score)

	if err := nn.Save(model, "wine_quality_model.json"); err != nil {
		fmt.Println("save:", err)
	}
}
```

- [ ] **Step 3: `examples/fashion_mnist/main.go`**

```go
package main

import (
	"fmt"
	"neugo/data"
	"neugo/nn"
	"neugo/train"
)

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

// syntheticFashionMNIST stands in for real data when no CSV is present —
// see the note at the top of Task 23.
func syntheticFashionMNIST(n int) *data.ImageDataset {
	rng := nn.NewRNG(11)
	images := make([]*data.Image, n)
	labels := make([][]float32, n)
	for i := 0; i < n; i++ {
		class := i % 10
		img := data.NewImage(28, 28, 1)
		for h := 0; h < 28; h++ {
			for w := 0; w < 28; w++ {
				v := float32(0.1)
				if (h+w)%10 == class {
					v = 0.9
				}
				img.Data[h][w][0] = v + (rng.Float32()-0.5)*0.05
			}
		}
		images[i] = img
		label := make([]float32, 10)
		label[class] = 1
		labels[i] = label
	}
	return &data.ImageDataset{Images: images, Labels: labels, Height: 28, Width: 28, Channels: 1}
}

func main() {
	dataset, err := data.LoadMNISTFromCSV("dataset/fashion_mnist/fashion-mnist_train.csv")
	if err != nil {
		fmt.Println("no Fashion-MNIST CSV found, using synthetic data:", err)
		dataset = syntheticFashionMNIST(200)
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
		nn.Linear(rng, 0, 64, nn.HeInit()),
		nn.ReLU(),
		nn.Dropout(0.3),
		nn.Linear(rng, 64, 10, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}

	x := imagesToTensor(dataset.Images)
	y := labelsToTensor(dataset.Labels)

	trainer := train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.CrossEntropy())
	if _, err := trainer.Fit(x, y, train.Epochs(20), train.BatchSize(16), train.Shuffle(true), train.Seed(2)); err != nil {
		fmt.Println("fit:", err)
		return
	}
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		fmt.Println("evaluate:", err)
		return
	}
	fmt.Printf("train-set accuracy: %.2f%%\n", metrics.Accuracy)
}
```

- [ ] **Step 4: `examples/cifar10_cnn/main.go`**

```go
package main

import (
	"fmt"
	"neugo/data"
	"neugo/nn"
	"neugo/train"
)

func cifarImagesToTensor(images []*data.Image) *nn.Tensor {
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

func cifarLabelsToTensor(labels [][]float32) *nn.Tensor {
	cols := len(labels[0])
	flat := make([]float32, len(labels)*cols)
	for i, row := range labels {
		copy(flat[i*cols:(i+1)*cols], row)
	}
	t, _ := nn.NewTensorFromData(flat, []int{len(labels), cols})
	return t
}

func syntheticCIFAR10(n int) *data.CIFAR10Dataset {
	rng := nn.NewRNG(21)
	classNames := []string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}
	images := make([]*data.Image, n)
	labels := make([][]float32, n)
	for i := 0; i < n; i++ {
		class := i % 10
		img := data.NewImage(32, 32, 3)
		for h := 0; h < 32; h++ {
			for w := 0; w < 32; w++ {
				for c := 0; c < 3; c++ {
					v := float32(0.1)
					if (h+w+c)%10 == class {
						v = 0.9
					}
					img.Data[h][w][c] = v + (rng.Float32()-0.5)*0.05
				}
			}
		}
		images[i] = img
		label := make([]float32, 10)
		label[class] = 1
		labels[i] = label
	}
	return &data.CIFAR10Dataset{Images: images, Labels: labels, ClassNames: classNames}
}

func main() {
	dataset, err := data.LoadCIFAR10Binary("dataset/cifar10/data_batch_1.bin")
	if err != nil {
		fmt.Println("no CIFAR-10 binary batch found, using synthetic data:", err)
		dataset = syntheticCIFAR10(200)
	}

	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{len(dataset.Images), 32, 32, 3},
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

	x := cifarImagesToTensor(dataset.Images)
	y := cifarLabelsToTensor(dataset.Labels)

	trainer := train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.CrossEntropy())
	if _, err := trainer.Fit(x, y, train.Epochs(15), train.BatchSize(16), train.Shuffle(true), train.Seed(2)); err != nil {
		fmt.Println("fit:", err)
		return
	}
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		fmt.Println("evaluate:", err)
		return
	}
	fmt.Printf("train-set accuracy: %.2f%% (classes: %v)\n", metrics.Accuracy, dataset.ClassNames)
}
```

- [ ] **Step 5: `examples/callbacks/main.go`**

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
		fmt.Println("build model:", err)
		return
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	opt := train.Adam(0.05, 0.9, 0.999, 1e-8)
	scheduler := train.StepDecay(opt, 0.5, 200)
	earlyStop := train.EarlyStopping(50)
	checkpoint := train.ModelCheckpoint("callbacks_best_model.json", "loss", "min", true)
	progress := train.ProgressBar(1000, 100)

	trainer := train.New(model, opt, train.BCELoss())
	hist, err := trainer.Fit(x, y,
		train.Epochs(1000), train.BatchSize(4), train.Seed(1),
		train.Validation(x, y), // XOR is small enough to validate on itself for this demo
		train.Callbacks(scheduler, earlyStop, checkpoint, progress),
		train.WithSaveFunc(nn.Save),
	)
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("stopped after %d epochs (early stopping patience 50)\n", len(hist.TrainLoss))
	if checkpoint.LastError != nil {
		fmt.Println("checkpoint save error:", checkpoint.LastError)
	}
}
```

- [ ] **Step 6: `examples/crossval/main.go`**

```go
package main

import (
	"fmt"
	"neugo/nn"
	"neugo/train"
)

func toTensor(rows [][]float32) *nn.Tensor {
	cols := len(rows[0])
	flat := make([]float32, len(rows)*cols)
	for i, row := range rows {
		copy(flat[i*cols:(i+1)*cols], row)
	}
	t, _ := nn.NewTensorFromData(flat, []int{len(rows), cols})
	return t
}

func main() {
	// A small linearly-separable-ish binary classification set, built
	// inline so cross-validation has something nontrivial to fold.
	dataRNG := nn.NewRNG(1)
	n := 60
	x := make([][]float32, n)
	y := make([][]float32, n)
	for i := 0; i < n; i++ {
		class := i % 2
		v := float32(class)*2 - 1 // -1 or 1
		x[i] = []float32{v + (dataRNG.Float32()-0.5)*0.3, v + (dataRNG.Float32()-0.5)*0.3}
		y[i] = []float32{float32(class)}
	}

	folds := train.KFoldSplits(dataRNG, x, y, 5, true)
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
		fmt.Println("cross-validate:", err)
		return
	}
	fmt.Printf("mean accuracy: %.2f%% (± %.2f)  mean F1: %.4f\n", result.MeanAccuracy, result.StdAccuracy, result.MeanF1)
	fmt.Printf("best fold: %d  worst fold: %d\n", result.BestFold, result.WorstFold)
}
```

- [ ] **Step 7: Delete the old flat example files and root `main.go`**

```bash
git rm examples/train.go examples/test_activations.go examples/test_losses.go \
  examples/example_usage.go examples/demo_phase1_phase2.go examples/showcase.go \
  examples/fraud_detection.go examples/wine_quality.go examples/wine_quality_clean.go \
  examples/phase4_demo.go examples/debug_training.go examples/complete_showcase.go \
  examples/check_predictions.go examples/cnn_demo.go examples/cnn_showcase.go \
  examples/cifar10_demo.go examples/fashion_mnist_demo.go examples/cats_vs_dogs_synthetic.go \
  examples/clean_api_demo.go examples/functional_demo.go examples/nnx_demo.go
git rm main.go
```

- [ ] **Step 8: Build and run every example**

Run each of the following; all must succeed (`fashion_mnist`/`cifar10_cnn` printing the synthetic-data fallback message is expected and fine — see Step 3/4's note):

```bash
go build ./...
go run ./examples/xor
go run ./examples/wine_quality
go run ./examples/fashion_mnist
go run ./examples/cifar10_cnn
go run ./examples/callbacks
go run ./examples/crossval
```

Expected: every command exits 0 and prints results (final loss / accuracy / cross-validation summary as appropriate).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(examples): rewrite as six buildable, runnable directories on the new nn/train API"
```

### Task 24: Consolidate documentation

**Files:**
- Modify: `README.md`
- Create: `docs/GUIDE.md`
- Delete: `API_SUMMARY.md`, `BEFORE_AFTER.md`, `CLEAN_API_GUIDE.md`, `CNN_GUIDE.md`, `FEATURE_GUIDE.md`, `FUNCTIONAL_API_GUIDE.md`, `NNX_API_GUIDE.md`, `QUICK_REFERENCE.md`, `examples/CNN_README.md`

Nine overlapping guides (~3.4k lines, design doc §1) collapse into exactly two documents: `README.md` (quickstart + a feature list that matches what actually exists after this plan) and `docs/GUIDE.md` (full API walkthrough). Every claim in the old guides about Adam/Momentum/RMSprop, softmax, or any other feature that was previously dead code (design doc §1) either now describes something real (post-Phase 2/3) or is removed — no doc may describe a feature that doesn't exist.

- [ ] **Step 1: Rewrite `README.md`**

```markdown
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

Six runnable examples live in `examples/`: `xor`, `wine_quality` (a real
dataset, `dataset/wine_quality/winequality-red.csv`), `fashion_mnist` and
`cifar10_cnn` (convolutional; fall back to synthetic data if you haven't
downloaded the real datasets — see each example's source), `callbacks`
(early stopping, checkpointing, LR scheduling, progress reporting), and
`crossval` (k-fold cross-validation). Run any of them with
`go run ./examples/<name>`.

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
```

- [ ] **Step 2: Write `docs/GUIDE.md`**

Required table of contents, each section fully worked with real, compiling code snippets pulled from (or directly matching) the actual `nn`/`train` API established in Tasks 1–21 — not aspirational syntax:

1. **Building models** — `Module`, `Param`, `Context`/`Mode`, every layer constructor's signature, the `inFeatures == 0` shape-inference rule, `Sequential`'s validation behavior and error format.
2. **Initializers** — when to use Xavier vs. He vs. Zeros/Uniform/Normal.
3. **Training** — the full `Trainer.Fit` option list (`Epochs`, `BatchSize`, `Shuffle`, `Seed`, `Validation`, `ClipGrad`, `Callbacks`, `WithSaveFunc`) with one worked example per option; the fused softmax+CrossEntropy behavior and how to tell whether it's active (`CrossEntropyLoss.Fused()`).
4. **Callbacks and schedulers** — one example combining `EarlyStopping` + `ModelCheckpoint` + a scheduler + `ProgressBar`, matching `examples/callbacks/main.go`.
5. **Convolutional models** — the `[batch, h, w, channels]` convention, `Conv2D` vs. `Conv2DSame`, a full CNN example matching `examples/fashion_mnist/main.go`.
6. **Evaluation and cross-validation** — `Metrics` fields, `KFoldSplits`/`StratifiedKFoldSplits`/`CrossValidate`, matching `examples/crossval/main.go`.
7. **Serialization** — `Save`/`Load`, the JSON format's shape (link to Task 21's `moduleDoc` type for anyone reading the source), and the explicit non-goal that RNG seed/optimizer state/training-resume are not supported.
8. **The `data` package** — `LoadCSV`, `CalculateStats`/`NormalizeZScore`/`NormalizeMinMax`, `SplitData`, `AnalyzeClassDistribution`/`BalanceDataset`, `LoadMNISTFromCSV`/`LoadCIFAR10Binary`, and the explicit-RNG convention every one of them follows.
9. **Migrating from the old `Network` API** — a short table mapping old concepts (`Network.NewNetworkWithLoss`, `Network.Fit`, `Network.Train`, `Network.CNN`, the four competing construction APIs) to their `nn`/`train` equivalent, so anyone who used NeuGo before this restructure has a map. Explicitly note: the old JSON model format is not loadable by `nn.Load` — models must be retrained.

- [ ] **Step 3: Delete the superseded guides**

```bash
git rm API_SUMMARY.md BEFORE_AFTER.md CLEAN_API_GUIDE.md CNN_GUIDE.md \
  FEATURE_GUIDE.md FUNCTIONAL_API_GUIDE.md NNX_API_GUIDE.md QUICK_REFERENCE.md \
  examples/CNN_README.md
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/GUIDE.md
git commit -m "docs: consolidate nine overlapping guides into README.md + docs/GUIDE.md"
```

### Task 25: Repo hygiene — `.gitignore` and artifact cleanup

**Files:**
- Create: `.gitignore`
- Delete: `best_wine_model.json`, `fraud_detection_model.json`, `showcase_best_model.json`, `showcase_final_model.json`, `showcase_model.json`, `trained_model.json`, `wine_quality_model.json`, `xor_final_model.json`, `test_predictions.csv`

**Interfaces:** none — this task touches no Go code.

- [ ] **Step 1: Create `.gitignore`**

```
# Trained-model artifacts and prediction output generated by examples —
# regenerate with `go run ./examples/<name>` rather than committing them.
*_model.json
test_predictions.csv

# Standard Go build/test artifacts.
*.test
*.out
```

- [ ] **Step 2: Remove committed artifacts**

```bash
git rm best_wine_model.json fraud_detection_model.json showcase_best_model.json \
  showcase_final_model.json showcase_model.json trained_model.json \
  wine_quality_model.json xor_final_model.json test_predictions.csv
```

`examples/wine_quality/main.go` (Task 23) regenerates an equivalent `wine_quality_model.json` at its repo-root working directory when run — `.gitignore` now keeps that regeneration from being re-committed by accident.

- [ ] **Step 3: Verify `git status` is clean of untracked artifacts**

Run: `git status`
Expected: no untracked `*.json` model files or `test_predictions.csv` listed (they're now gitignored, not merely deleted — running any example again must not resurrect them in `git status`).

- [ ] **Step 4: Commit**

```bash
git add .gitignore -A
git commit -m "chore: add .gitignore, remove committed model artifacts and prediction CSV"
```

### Task 26: Final full verification

**Files:** none — this task is purely verification, closing out design doc §5 phase 4's exit criteria: "Full `go build ./... && go vet ./... && go test ./...` green."

- [ ] **Step 1: Full build**

Run: `go build ./...`
Expected: exits 0, no output. Every package (`nn`, `train`, `data`, and all six `examples/*`) compiles.

- [ ] **Step 2: Full vet**

Run: `go vet ./...`
Expected: exits 0, no output.

- [ ] **Step 3: Full test suite**

Run: `go test ./... -v`
Expected: every test from Tasks 1–22 passes. Then run it a second time back-to-back:

Run: `go test ./... -count=1`
Expected: still all green — confirms no test relies on stale build cache to pass, and (combined with every test's fixed seed, per Global Constraints) confirms determinism.

- [ ] **Step 4: Confirm the Global Constraints hold across the whole tree**

```bash
grep -rn "panic(" nn/ train/ data/ --include="*.go" | grep -v _test.go
grep -rn "println(" nn/ train/ data/ --include="*.go" | grep -v _test.go
grep -rn "rand.Seed\|math/rand\"$" nn/ train/ data/ --include="*.go" | grep -v _test.go
```

Expected: the first two greps return nothing (no `panic` or builtin `println` in library code). The third should show `"math/rand"` imports only in files that also take an explicit `*rand.Rand` parameter (`nn/rng.go`, `nn/init.go`, `nn/linear.go`, `nn/conv.go`, `train/optimizer.go` if it stores per-param maps needing no rand — check each hit by hand) and zero occurrences of `rand.Seed(`.

- [ ] **Step 5: Confirm every example still runs**

Re-run Task 23 Step 8's six `go run` commands. All must still succeed after Tasks 24–25's deletions (they touched no example code, but this confirms nothing in docs/hygiene cleanup accidentally broke a relative path an example depends on, e.g. `dataset/wine_quality/winequality-red.csv`).

- [ ] **Step 6: Final commit (if any of Steps 1–5 required a fix)**

If everything was already green, there is nothing to commit here — Task 25's commit is the natural end of the branch. If a fix was needed, commit it on its own:

```bash
git add -A
git commit -m "fix: address findings from final full-repo verification"
```

---

## Risks and mitigations

Carried forward from design doc §6, with where this plan actually addresses each:

- **Float32 gradient checks are noisy** → every gradient-checked task (4, 5, 6's Sequential composition implicitly, 13, 14, 15, 17) uses the shared `checkInputGradient`/`checkParamGradient` helpers from Task 4 with central differences, `eps = 1e-2`, small tensors (≤ a few dozen elements), and fixed seeds — see Global Constraints for the tolerance rationale.
- **Batched conv backward is the most error-prone math** → Task 13 implements `Conv2DLayer.Backward` directly (not against the old single-sample `Network/conv2d.go`, which is deleted in Task 20 before Task 13 would even have a reference to check against — Task 13 runs in Phase 3, Task 20 at Phase 3's end) and leans entirely on `checkParamGradient`/`checkInputGradient` for correctness, the same mechanism that caught the batch-scaling bug in `Linear.Backward` during this plan's own authoring (see design decision #6) — gradient checks, not eyeballing old code, are this plan's actual safety net here.
- **Scope creep toward autodiff** → no task introduces a computation graph, `grad` transform, or anything beyond per-module manual `Forward`/`Backward`; every module in Tasks 4–17 follows the same two-method shape.
- **Old-format model files** → Task 21's `Load` only understands the `{type, config, params, modules}` format `Save` writes; there is no code path that reads the old `Version: "2.0"` format. Task 25 deletes every committed old-format artifact; Task 23's examples regenerate fresh ones.

## Plan self-review notes

Per the writing-plans skill, this plan was checked against the design doc section-by-section (every §1 bug, §2 goal, the selected §1 "verified-missing feature," and every §3 non-goal was matched to a task or explicitly confirmed absent) and scanned for placeholder language before being handed off. Two real issues surfaced during that process and were fixed in place rather than left as caveats:

1. **A genuine gradient-computation bug**: an early draft of `LinearLayer.Backward` (Task 4) divided `W.Grad`/`B.Grad` by an extra `batch` factor that doesn't belong there, since `Loss.Loss` (Task 7) already bakes the appropriate batch-normalization into `gradOut`. This would have failed Task 4's own `checkParamGradient` test. Found while hand-deriving `BatchNorm.Backward` (Task 17) and required re-deriving the correct chain rule; fixed in both `Linear.Backward` and `Conv2DLayer.Backward` (Task 13), and called out as design decision #6 so no later task repeats it.
2. **A sequencing bug**: `tensor/` was originally scheduled for deletion at the end of Phase 3 (alongside `Network/`), but `data/image.go` and `data/cifar10.go` don't stop depending on it until Task 22 (Phase 4) — deleting it earlier would have broken `go build ./...` on the `data` package between Phase 3 and Phase 4, violating the Global Constraints' "green at the end of every phase" rule. Fixed by moving `tensor/`'s deletion into Task 22.

One accepted gap, noted rather than fixed, because fixing it would ripple through every already-written task: design doc §4.6 says "constructors ... return `(T, error)` on invalid configuration," but `Linear`/`Conv2D`/`Conv2DSame` (Tasks 4, 13) return bare `*LinearLayer`/`*Conv2DLayer` with no error return, because none of their constructor-time arguments have a validatable-at-construction failure mode *except* `Conv2DSame` with an even `kernelSize`, which currently just produces asymmetric-but-computable padding instead of erroring. If this matters for a specific use case, add a parity check to `Conv2DSame` returning `(*Conv2DLayer, error)` — this would require updating every call site across Tasks 13, 20, 21, 23 to handle the new error return, which is why it's flagged here for a deliberate decision rather than silently patched mid-plan.

---

**For agentic workers:** this plan has 26 tasks across 4 phases. Use `superpowers:subagent-driven-development` (one fresh subagent per task, two-stage review between tasks) or `superpowers:executing-plans` (batch execution with checkpoints) to work through them in order — later tasks assume earlier ones' `Produces:` interfaces exist exactly as specified. Do not skip a phase's exit-criteria tests (Task 6 end / Task 12 end / Task 20 / Task 26) before starting the next phase.
