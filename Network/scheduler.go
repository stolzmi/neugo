package Network

import "math"

// LRScheduler interface for learning rate schedulers
type LRScheduler interface {
	GetLearningRate(epoch int) float32
	Step()
}

// StepDecayScheduler reduces learning rate by a factor every N epochs
type StepDecayScheduler struct {
	InitialLR  float32
	DecayRate  float32
	DecaySteps int
	CurrentLR  float32
	Epoch      int
}

func NewStepDecay(initialLR, decayRate float32, decaySteps int) *StepDecayScheduler {
	return &StepDecayScheduler{
		InitialLR:  initialLR,
		DecayRate:  decayRate,
		DecaySteps: decaySteps,
		CurrentLR:  initialLR,
		Epoch:      0,
	}
}

func (s *StepDecayScheduler) GetLearningRate(epoch int) float32 {
	numDecays := epoch / s.DecaySteps
	return s.InitialLR * float32(math.Pow(float64(s.DecayRate), float64(numDecays)))
}

func (s *StepDecayScheduler) Step() {
	s.Epoch++
	s.CurrentLR = s.GetLearningRate(s.Epoch)
}

// ExponentialDecayScheduler reduces learning rate exponentially
type ExponentialDecayScheduler struct {
	InitialLR float32
	DecayRate float32
	CurrentLR float32
	Epoch     int
}

func NewExponentialDecay(initialLR, decayRate float32) *ExponentialDecayScheduler {
	return &ExponentialDecayScheduler{
		InitialLR: initialLR,
		DecayRate: decayRate,
		CurrentLR: initialLR,
		Epoch:     0,
	}
}

func (s *ExponentialDecayScheduler) GetLearningRate(epoch int) float32 {
	return s.InitialLR * float32(math.Pow(float64(s.DecayRate), float64(epoch)))
}

func (s *ExponentialDecayScheduler) Step() {
	s.Epoch++
	s.CurrentLR = s.GetLearningRate(s.Epoch)
}

// CosineAnnealingScheduler uses cosine annealing
type CosineAnnealingScheduler struct {
	InitialLR float32
	MinLR     float32
	MaxEpochs int
	CurrentLR float32
	Epoch     int
}

func NewCosineAnnealing(initialLR, minLR float32, maxEpochs int) *CosineAnnealingScheduler {
	return &CosineAnnealingScheduler{
		InitialLR: initialLR,
		MinLR:     minLR,
		MaxEpochs: maxEpochs,
		CurrentLR: initialLR,
		Epoch:     0,
	}
}

func (s *CosineAnnealingScheduler) GetLearningRate(epoch int) float32 {
	if epoch >= s.MaxEpochs {
		return s.MinLR
	}
	progress := float64(epoch) / float64(s.MaxEpochs)
	cosineDecay := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	return s.MinLR + (s.InitialLR-s.MinLR)*float32(cosineDecay)
}

func (s *CosineAnnealingScheduler) Step() {
	s.Epoch++
	s.CurrentLR = s.GetLearningRate(s.Epoch)
}

// WarmupScheduler implements linear warmup
type WarmupScheduler struct {
	InitialLR   float32
	TargetLR    float32
	WarmupSteps int
	CurrentLR   float32
	Epoch       int
}

func NewWarmup(initialLR, targetLR float32, warmupSteps int) *WarmupScheduler {
	return &WarmupScheduler{
		InitialLR:   initialLR,
		TargetLR:    targetLR,
		WarmupSteps: warmupSteps,
		CurrentLR:   initialLR,
		Epoch:       0,
	}
}

func (s *WarmupScheduler) GetLearningRate(epoch int) float32 {
	if epoch >= s.WarmupSteps {
		return s.TargetLR
	}
	progress := float32(epoch) / float32(s.WarmupSteps)
	return s.InitialLR + (s.TargetLR-s.InitialLR)*progress
}

func (s *WarmupScheduler) Step() {
	s.Epoch++
	s.CurrentLR = s.GetLearningRate(s.Epoch)
}

// ReduceLROnPlateauScheduler reduces learning rate when metric plateaus
type ReduceLROnPlateauScheduler struct {
	InitialLR    float32
	Factor       float32
	Patience     int
	MinLR        float32
	CurrentLR    float32
	BestMetric   float32
	Counter      int
	Mode         string // "min" or "max"
}

func NewReduceLROnPlateau(initialLR, factor float32, patience int, minLR float32, mode string) *ReduceLROnPlateauScheduler {
	bestMetric := float32(math.Inf(1))
	if mode == "max" {
		bestMetric = float32(math.Inf(-1))
	}

	return &ReduceLROnPlateauScheduler{
		InitialLR:  initialLR,
		Factor:     factor,
		Patience:   patience,
		MinLR:      minLR,
		CurrentLR:  initialLR,
		BestMetric: bestMetric,
		Counter:    0,
		Mode:       mode,
	}
}

func (s *ReduceLROnPlateauScheduler) GetLearningRate(epoch int) float32 {
	return s.CurrentLR
}

func (s *ReduceLROnPlateauScheduler) Step() {
	// No-op, use StepWithMetric instead
}

func (s *ReduceLROnPlateauScheduler) StepWithMetric(metric float32) {
	improved := false

	if s.Mode == "min" {
		if metric < s.BestMetric {
			s.BestMetric = metric
			improved = true
		}
	} else {
		if metric > s.BestMetric {
			s.BestMetric = metric
			improved = true
		}
	}

	if improved {
		s.Counter = 0
	} else {
		s.Counter++
		if s.Counter >= s.Patience {
			newLR := s.CurrentLR * s.Factor
			if newLR > s.MinLR {
				s.CurrentLR = newLR
			}
			s.Counter = 0
		}
	}
}

// FitWithScheduler trains with learning rate scheduling
func (network *Network) FitWithScheduler(
	inputs [][]float32,
	labels [][]float32,
	epochs int,
	batchSize int,
	scheduler LRScheduler,
	verbose bool,
) []float32 {

	losses := make([]float32, 0, epochs)
	numSamples := len(inputs)

	for epoch := 0; epoch < epochs; epoch++ {
		// Get current learning rate from scheduler
		learningRate := scheduler.GetLearningRate(epoch)

		epochLoss := float32(0.0)
		numBatches := 0

		// Process batches
		for i := 0; i < numSamples; i += batchSize {
			end := i + batchSize
			if end > numSamples {
				end = numSamples
			}

			batchInputs := inputs[i:end]
			batchLabels := labels[i:end]

			batchLoss := network.TrainBatch(batchInputs, batchLabels, learningRate)
			epochLoss += batchLoss
			numBatches++
		}

		avgLoss := epochLoss / float32(numBatches)
		losses = append(losses, avgLoss)

		// Step the scheduler
		scheduler.Step()

		if verbose && ((epoch+1)%100 == 0 || epoch == 0) {
			println("Epoch", epoch+1, "- Loss:", avgLoss, "- LR:", learningRate)
		}
	}

	return losses
}
