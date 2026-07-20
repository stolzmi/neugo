package train

import (
	"math"
	"github.com/stolzmi/neugo/nn"
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
