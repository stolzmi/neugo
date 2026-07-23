package train

import (
	"fmt"
	"math/rand"

	"github.com/stolzmi/neugo/data"
	"github.com/stolzmi/neugo/nn"
)

type Trainer struct {
	model *nn.SequentialModel
	opt   Optimizer
	loss  Loss
	// rng backs TrainOnBatch/FitStream's persistent Context across calls
	// (seeded 42 by default, matching FitConfig's default Seed) — unlike
	// Fit, which creates its own fresh RNG from Fit's own Seed(...) option
	// every call, TrainOnBatch has no per-call config to carry a seed, so
	// Trainer itself owns one RNG stream shared across every TrainOnBatch/
	// FitStream call, keeping Dropout/etc. randomness consistent across a
	// whole streamed training run.
	rng *rand.Rand
}

// New builds a Trainer and, if loss is a *CrossEntropyLoss, sets the fused
// softmax+CCE gradient shortcut (design decision #3 in the plan header)
// based on whether the model's last module is *nn.SoftmaxModule. The flag
// is set unconditionally (both true and false) so that reusing a single
// *CrossEntropyLoss instance across multiple Trainers never leaves a stale
// fused=true from a previous model.
func New(model *nn.SequentialModel, opt Optimizer, loss Loss) *Trainer {
	if ce, ok := loss.(*CrossEntropyLoss); ok {
		modules := model.Modules()
		isSoftmax := false
		if n := len(modules); n > 0 {
			_, isSoftmax = modules[n-1].(*nn.SoftmaxModule)
		}
		ce.SetFused(isSoftmax)
	}
	return &Trainer{model: model, opt: opt, loss: loss, rng: nn.NewRNG(42)}
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

func Epochs(n int) FitOption         { return func(c *FitConfig) { c.epochs = n } }
func BatchSize(n int) FitOption      { return func(c *FitConfig) { c.batchSize = n } }
func Shuffle(enabled bool) FitOption { return func(c *FitConfig) { c.shuffle = enabled } }
func Seed(seed int64) FitOption      { return func(c *FitConfig) { c.seed = seed } }
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

// trainStep is the one forward/loss/backward/optimizer-step body shared
// by Fit's batch loop and TrainOnBatch — factored out so both paths apply
// the fused-softmax+CCE shortcut identically instead of duplicating it.
// ctx and opt are passed in rather than read off Trainer because Fit uses
// a ClipNorm-wrapped opt and a Context scoped to one Fit call, while
// TrainOnBatch uses t.opt and t.rng directly.
func (t *Trainer) trainStep(ctx *nn.Context, opt Optimizer, xb, yb *nn.Tensor) (float32, error) {
	modules := t.model.Modules()
	fused := false
	if ce, ok := t.loss.(*CrossEntropyLoss); ok {
		fused = ce.Fused()
	}

	pred, err := t.model.Forward(ctx, xb)
	if err != nil {
		return 0, fmt.Errorf("train: forward: %w", err)
	}
	lossVal, gradOut, err := t.loss.Loss(pred, yb)
	if err != nil {
		return 0, fmt.Errorf("train: loss: %w", err)
	}

	if fused && len(modules) > 0 {
		grad := gradOut
		for i := len(modules) - 2; i >= 0; i-- {
			if grad, err = modules[i].Backward(ctx, grad); err != nil {
				return 0, fmt.Errorf("train: backward: %w", err)
			}
		}
	} else if _, err := t.model.Backward(ctx, gradOut); err != nil {
		return 0, fmt.Errorf("train: backward: %w", err)
	}

	opt.Step(t.model.Params())
	return lossVal, nil
}

// TrainOnBatch runs one forward/loss/backward/optimizer-step cycle on a
// single already-materialized batch (xb, yb) — the primitive FitStream is
// built on, and a lower-level entry point for callers driving their own
// epoch loop (e.g. over a data.DataLoader) instead of Fit's whole-tensor
// one. Uses Trainer's own persistent RNG (see the rng field's doc
// comment), not a fresh one per call.
func (t *Trainer) TrainOnBatch(xb, yb *nn.Tensor) (float32, error) {
	ctx := &nn.Context{Mode: nn.Train, RNG: t.rng}
	return t.trainStep(ctx, t.opt, xb, yb)
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

			lossVal, err := t.trainStep(ctx, opt, xb, yb)
			if err != nil {
				return nil, err
			}

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

// FitStream trains via a data.DataLoader instead of one big materialized
// tensor — an epoch loop built on TrainOnBatch/trainStep, for datasets
// that don't fit in memory as a single *nn.Tensor. convert turns one
// batch of indices (as yielded by loader.Next) into that batch's (x, y)
// tensors; callers own whatever in-memory or on-disk dataset backs those
// indices (mirroring examples/cifar10_cnn's own image-to-tensor
// conversion, kept out of the data package — see data.DataLoader's doc
// comment). Reuses Fit's FitConfig/FitOption set for clip-grad/callbacks/
// validation/epochs; BatchSize/Shuffle/Seed are Fit-only (loader already
// owns batching, shuffling, and its own RNG) and are silently ignored here
// if passed.
func (t *Trainer) FitStream(loader *data.DataLoader, convert func(batchIdx []int) (*nn.Tensor, *nn.Tensor, error), opts ...FitOption) (*History, error) {
	cfg := &FitConfig{seed: 42}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.epochs <= 0 {
		return nil, fmt.Errorf("train: FitStream requires train.Epochs(n) with n > 0")
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
	ctx := &nn.Context{Mode: nn.Train, RNG: t.rng}

	hist := NewHistory()
	hist.OnTrainBegin()
	cfg.callbacks.OnTrainBegin()

	for epoch := 0; epoch < cfg.epochs; epoch++ {
		cfg.callbacks.OnEpochBegin(epoch)
		loader.Reset()

		var epochLoss float32
		numBatches := 0
		for {
			batchIdx, ok := loader.Next()
			if !ok {
				break
			}
			xb, yb, err := convert(batchIdx)
			if err != nil {
				return nil, fmt.Errorf("train: FitStream: convert: %w", err)
			}
			lossVal, err := t.trainStep(ctx, opt, xb, yb)
			if err != nil {
				return nil, err
			}
			epochLoss += lossVal
			numBatches++
			cfg.callbacks.OnBatchEnd(numBatches-1, lossVal)
		}
		avgTrainLoss := epochLoss / float32(numBatches)

		var valMetrics *Metrics
		if cfg.hasVal {
			m, err := t.Evaluate(cfg.valX, cfg.valY)
			if err != nil {
				return nil, fmt.Errorf("train: FitStream: validation evaluate: %w", err)
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
