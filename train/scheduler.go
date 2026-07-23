package train

import (
	"github.com/stolzmi/neugo/nn"
	"math"
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
	if s.decaySteps <= 0 {
		s.opt.SetLR(s.initialLR)
		return
	}
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
	opt              Optimizer
	initialLR, minLR float32
	maxEpochs        int
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
	opt                 Optimizer
	initialLR, targetLR float32
	warmupSteps         int
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

// cosineInterp interpolates from start (progress=0) to end (progress=1)
// along the same cosine curve CosineAnnealingScheduler uses, generalized
// to work whether start < end (annealing up) or start > end (annealing
// down) — CosineAnnealingScheduler is the start>end-only special case.
func cosineInterp(start, end float32, progress float64) float32 {
	cosDecay := 0.5 * (1 + math.Cos(math.Pi*progress))
	return end + (start-end)*float32(cosDecay)
}

// OneCycleLRScheduler implements Smith's "1cycle" policy: LR rises via a
// cosine curve from an initial LR (MaxLR/25) up to MaxLR over the first
// PctStart fraction of TotalSteps, then anneals back down via cosine to a
// much lower final LR (MaxLR/25/1e4) over the remaining steps. Unlike the
// per-epoch schedulers above, it steps on OnBatchEnd — since Fit provides
// no OnBatchBegin/global-step hook, it keeps its own counter incremented
// once per call, ignoring the batch argument's value (which resets every
// epoch); construct it with TotalSteps = epochs * batchesPerEpoch.
type OneCycleLRScheduler struct {
	BaseCallback
	opt                       Optimizer
	initialLR, maxLR, finalLR float32
	pctStart                  float32
	totalSteps                int
	step                      int
}

func OneCycleLR(opt Optimizer, maxLR float32, totalSteps int) *OneCycleLRScheduler {
	s := &OneCycleLRScheduler{
		opt:        opt,
		initialLR:  maxLR / 25,
		maxLR:      maxLR,
		finalLR:    maxLR / 25 / 1e4,
		pctStart:   0.3,
		totalSteps: totalSteps,
	}
	s.opt.SetLR(s.lrAt(0))
	return s
}

func (s *OneCycleLRScheduler) lrAt(step int) float32 {
	warmupSteps := int(float32(s.totalSteps) * s.pctStart)
	if warmupSteps < 1 {
		warmupSteps = 1
	}
	if step >= s.totalSteps {
		step = s.totalSteps - 1
	}
	if step < warmupSteps {
		return cosineInterp(s.initialLR, s.maxLR, float64(step)/float64(warmupSteps))
	}
	decaySteps := s.totalSteps - warmupSteps
	if decaySteps < 1 {
		decaySteps = 1
	}
	return cosineInterp(s.maxLR, s.finalLR, float64(step-warmupSteps)/float64(decaySteps))
}

func (s *OneCycleLRScheduler) OnBatchEnd(batch int, loss float32) {
	s.step++
	if s.step >= s.totalSteps {
		s.opt.SetLR(s.finalLR)
		return
	}
	s.opt.SetLR(s.lrAt(s.step))
}

// CyclicLRScheduler implements Smith's triangular cyclic LR policy: LR
// ramps linearly from BaseLR up to MaxLR over StepSizeUp steps, then back
// down to BaseLR over StepSizeDown steps, repeating indefinitely. Like
// OneCycleLRScheduler, it steps on OnBatchEnd with its own internal
// counter rather than the batch argument.
type CyclicLRScheduler struct {
	BaseCallback
	opt                      Optimizer
	baseLR, maxLR            float32
	stepSizeUp, stepSizeDown int
	step                     int
}

func CyclicLR(opt Optimizer, baseLR, maxLR float32, stepSizeUp, stepSizeDown int) *CyclicLRScheduler {
	s := &CyclicLRScheduler{opt: opt, baseLR: baseLR, maxLR: maxLR, stepSizeUp: stepSizeUp, stepSizeDown: stepSizeDown}
	s.opt.SetLR(s.lrAt(0))
	return s
}

func (s *CyclicLRScheduler) lrAt(step int) float32 {
	cycleLen := s.stepSizeUp + s.stepSizeDown
	if cycleLen <= 0 {
		return s.baseLR
	}
	pos := step % cycleLen
	if pos < s.stepSizeUp {
		if s.stepSizeUp == 0 {
			return s.maxLR
		}
		return s.baseLR + (s.maxLR-s.baseLR)*float32(pos)/float32(s.stepSizeUp)
	}
	downPos := pos - s.stepSizeUp
	if s.stepSizeDown == 0 {
		return s.baseLR
	}
	return s.maxLR - (s.maxLR-s.baseLR)*float32(downPos)/float32(s.stepSizeDown)
}

func (s *CyclicLRScheduler) OnBatchEnd(batch int, loss float32) {
	s.step++
	s.opt.SetLR(s.lrAt(s.step))
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
