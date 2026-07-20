package train

import (
	"math"
	"github.com/stolzmi/neugo/nn"
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

func TestAdamZeroWeightDecayMatchesPlainAdam(t *testing.T) {
	p := newTestParam([]float32{10}, []float32{0})
	plain := Adam(0.5, 0.9, 0.999, 1e-8)
	decoupled := AdamW(0.5, 0.9, 0.999, 1e-8, 0)
	for i := 0; i < 10; i++ {
		g := 2 * p.Value.Data[0]
		p.Grad.Data[0] = g
		plain.Step([]*nn.Param{p})
	}
	p2 := newTestParam([]float32{10}, []float32{0})
	for i := 0; i < 10; i++ {
		p2.Grad.Data[0] = 2 * p2.Value.Data[0]
		decoupled.Step([]*nn.Param{p2})
	}
	if diff := math.Abs(float64(p.Value.Data[0] - p2.Value.Data[0])); diff > 1e-6 {
		t.Fatalf("AdamW with WeightDecay=0 diverged from Adam: %v vs %v", p2.Value.Data[0], p.Value.Data[0])
	}
}

func TestAdamWShrinksWeightBeyondGradientUpdate(t *testing.T) {
	// Zero gradient isolates the decay term: only p -= LR*WeightDecay*p
	// should move the value, since Adam's own update needs a nonzero
	// gradient to move at all.
	p := newTestParam([]float32{10}, []float32{0})
	opt := AdamW(0.1, 0.9, 0.999, 1e-8, 0.1)
	opt.Step([]*nn.Param{p})
	want := float32(10 - 0.1*0.1*10) // 9.9
	if diff := math.Abs(float64(p.Value.Data[0] - want)); diff > 1e-5 {
		t.Fatalf("Value after 1 AdamW step with zero grad = %v, want %v", p.Value.Data[0], want)
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
