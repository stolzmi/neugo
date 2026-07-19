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
