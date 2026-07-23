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

func TestOneCycleLRSetsInitialLROnConstruction(t *testing.T) {
	opt := SGD(0.1) // whatever the optimizer already had must be overridden
	OneCycleLR(opt, 1.0, 100)
	want := float32(1.0) / 25
	if diff := math.Abs(float64(opt.GetLR() - want)); diff > 1e-5 {
		t.Fatalf("LR after construction = %v, want %v (maxLR/25)", opt.GetLR(), want)
	}
}

func TestOneCycleLRPeaksAtWarmupBoundary(t *testing.T) {
	opt := SGD(0.0)
	totalSteps := 100
	s := OneCycleLR(opt, 1.0, totalSteps)
	warmupSteps := int(float32(totalSteps) * 0.3)
	for i := 0; i < warmupSteps; i++ {
		s.OnBatchEnd(i, 0)
	}
	if diff := math.Abs(float64(opt.GetLR() - 1.0)); diff > 1e-4 {
		t.Fatalf("LR at warmup boundary (step %d) = %v, want 1.0 (maxLR)", warmupSteps, opt.GetLR())
	}
}

func TestOneCycleLREndsAtFinalLR(t *testing.T) {
	opt := SGD(0.0)
	totalSteps := 100
	s := OneCycleLR(opt, 1.0, totalSteps)
	for i := 0; i < totalSteps; i++ {
		s.OnBatchEnd(i, 0)
	}
	want := float32(1.0) / 25 / 1e4
	if diff := math.Abs(float64(opt.GetLR() - want)); diff > 1e-6 {
		t.Fatalf("LR after totalSteps batches = %v, want %v (finalLR)", opt.GetLR(), want)
	}
	// Further calls past totalSteps must stay pinned at finalLR, not error
	// or extrapolate past the schedule.
	s.OnBatchEnd(totalSteps, 0)
	if diff := math.Abs(float64(opt.GetLR() - want)); diff > 1e-6 {
		t.Fatalf("LR past totalSteps = %v, want it to stay at finalLR %v", opt.GetLR(), want)
	}
}

func TestOneCycleLRMonotonicWithinEachPhase(t *testing.T) {
	opt := SGD(0.0)
	totalSteps := 50
	s := OneCycleLR(opt, 1.0, totalSteps)
	warmupSteps := int(float32(totalSteps) * 0.3)
	prev := opt.GetLR()
	for i := 0; i < warmupSteps; i++ {
		s.OnBatchEnd(i, 0)
		if opt.GetLR() < prev {
			t.Fatalf("LR decreased during warmup phase at step %d: %v -> %v", i, prev, opt.GetLR())
		}
		prev = opt.GetLR()
	}
	for i := warmupSteps; i < totalSteps; i++ {
		s.OnBatchEnd(i, 0)
		if opt.GetLR() > prev {
			t.Fatalf("LR increased during decay phase at step %d: %v -> %v", i, prev, opt.GetLR())
		}
		prev = opt.GetLR()
	}
}

func TestCyclicLRTriangularWave(t *testing.T) {
	opt := SGD(0.0)
	s := CyclicLR(opt, 0.1, 1.0, 4, 4)
	if diff := math.Abs(float64(opt.GetLR() - 0.1)); diff > 1e-5 {
		t.Fatalf("LR at construction = %v, want 0.1 (baseLR)", opt.GetLR())
	}
	for i := 0; i < 4; i++ {
		s.OnBatchEnd(i, 0)
	}
	if diff := math.Abs(float64(opt.GetLR() - 1.0)); diff > 1e-5 {
		t.Fatalf("LR after stepSizeUp batches = %v, want 1.0 (maxLR)", opt.GetLR())
	}
	for i := 0; i < 4; i++ {
		s.OnBatchEnd(i, 0)
	}
	if diff := math.Abs(float64(opt.GetLR() - 0.1)); diff > 1e-5 {
		t.Fatalf("LR after one full cycle = %v, want 0.1 (back to baseLR)", opt.GetLR())
	}
	// Second cycle repeats the same shape.
	for i := 0; i < 4; i++ {
		s.OnBatchEnd(i, 0)
	}
	if diff := math.Abs(float64(opt.GetLR() - 1.0)); diff > 1e-5 {
		t.Fatalf("LR at peak of second cycle = %v, want 1.0 (maxLR)", opt.GetLR())
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
