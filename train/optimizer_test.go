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

func TestAdagradDecreasesQuadraticLoss(t *testing.T) {
	p := newTestParam([]float32{10}, []float32{0})
	opt := Adagrad(1.0, 1e-8)
	for i := 0; i < 1000; i++ {
		p.Grad.Data[0] = 2 * p.Value.Data[0]
		opt.Step([]*nn.Param{p})
	}
	if math.Abs(float64(p.Value.Data[0])) > 1 {
		t.Fatalf("Adagrad did not converge x toward 0: got %v", p.Value.Data[0])
	}
}

func TestAdagradStateIsolatedPerParam(t *testing.T) {
	// One optimizer instance, two params: p2's much larger gradient must
	// not leak into p1's accumulator (which is keyed by *nn.Param
	// identity in a map, per the codebase's existing Momentum/Adam/RMSprop
	// convention). If it did, p1's second step below would shrink to
	// ~0.001 instead of the ~0.0707 an isolated accumulator produces.
	opt := Adagrad(0.1, 1e-8)
	p1 := newTestParam([]float32{1}, []float32{1})
	p2 := newTestParam([]float32{1}, []float32{100})
	opt.Step([]*nn.Param{p1})
	opt.Step([]*nn.Param{p2})
	p1.Grad.Data[0] = 1
	opt.Step([]*nn.Param{p1})
	want := float32(0.9) - float32(0.1)/float32(math.Sqrt(2))
	if diff := math.Abs(float64(p1.Value.Data[0] - want)); diff > 1e-4 {
		t.Fatalf("p1.Value after p2 interleaved = %v, want %v (isolated accumulator)", p1.Value.Data[0], want)
	}
}

func TestAdadeltaDecreasesQuadraticLoss(t *testing.T) {
	p := newTestParam([]float32{10}, []float32{0})
	opt := Adadelta(1.0, 0.9, 1e-6)
	for i := 0; i < 2000; i++ {
		p.Grad.Data[0] = 2 * p.Value.Data[0]
		opt.Step([]*nn.Param{p})
	}
	if math.Abs(float64(p.Value.Data[0])) > 1 {
		t.Fatalf("Adadelta did not converge x toward 0: got %v", p.Value.Data[0])
	}
}

func TestNadamFirstStepBiasCorrection(t *testing.T) {
	p := newTestParam([]float32{0}, []float32{1})
	opt := Nadam(0.001, 0.9, 0.999, 1e-8)
	opt.Step([]*nn.Param{p})
	// Unlike Adam, Nadam's mHat mixes in beta1*m (bias-corrected against
	// the *next* step) with (1-beta1)*g (bias-corrected against the
	// *current* step): mHat = 0.9*0.1/(1-0.81) + 0.1*1/(1-0.9) ~= 1.4737,
	// vHat ~= 1, so update ~= lr*1.4737, not lr*1 like Adam's first step.
	want := float32(-0.001 * 1.4737)
	if diff := math.Abs(float64(p.Value.Data[0] - want)); diff > 1e-4 {
		t.Fatalf("Value after 1 Nadam step = %v, want ~%v", p.Value.Data[0], want)
	}
}

func TestNadamDecreasesQuadraticLoss(t *testing.T) {
	p := newTestParam([]float32{10}, []float32{0})
	opt := Nadam(0.5, 0.9, 0.999, 1e-8)
	for i := 0; i < 50; i++ {
		p.Grad.Data[0] = 2 * p.Value.Data[0]
		opt.Step([]*nn.Param{p})
	}
	if math.Abs(float64(p.Value.Data[0])) > 1 {
		t.Fatalf("Nadam did not converge x toward 0: got %v", p.Value.Data[0])
	}
}

func TestLionStepsBySignNotMagnitude(t *testing.T) {
	small := newTestParam([]float32{10}, []float32{0.01})
	large := newTestParam([]float32{10}, []float32{1000})
	opt1 := Lion(0.1, 0.9, 0.99)
	opt2 := Lion(0.1, 0.9, 0.99)
	opt1.Step([]*nn.Param{small})
	opt2.Step([]*nn.Param{large})
	// Both gradients are positive, so both should step by exactly -LR,
	// regardless of gradient magnitude.
	want := float32(10 - 0.1)
	if diff := math.Abs(float64(small.Value.Data[0] - want)); diff > 1e-6 {
		t.Errorf("small-gradient step = %v, want %v", small.Value.Data[0], want)
	}
	if diff := math.Abs(float64(large.Value.Data[0] - want)); diff > 1e-6 {
		t.Errorf("large-gradient step = %v, want %v", large.Value.Data[0], want)
	}
}

func TestLionDecreasesQuadraticLoss(t *testing.T) {
	p := newTestParam([]float32{10}, []float32{0})
	opt := Lion(0.1, 0.9, 0.99)
	for i := 0; i < 300; i++ {
		p.Grad.Data[0] = 2 * p.Value.Data[0]
		opt.Step([]*nn.Param{p})
	}
	if math.Abs(float64(p.Value.Data[0])) > 1 {
		t.Fatalf("Lion did not converge x toward 0: got %v", p.Value.Data[0])
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

func TestL1RegAddsSignPenaltyToGradient(t *testing.T) {
	p := newTestParam([]float32{2, -3, 0}, []float32{0, 0, 0})
	L1Reg(SGD(1.0), 0.1).Step([]*nn.Param{p})
	// Zero gradient isolates the penalty: value -= lr*grad where grad is
	// purely lambda*sign(w).
	want := []float32{2 - 0.1, -3 + 0.1, 0}
	for i := range want {
		if diff := math.Abs(float64(p.Value.Data[i] - want[i])); diff > 1e-6 {
			t.Errorf("Value[%d] = %v, want %v", i, p.Value.Data[i], want[i])
		}
	}
}

func TestL2RegAddsWeightScaledPenaltyToGradient(t *testing.T) {
	p := newTestParam([]float32{2, -3}, []float32{0, 0})
	L2Reg(SGD(1.0), 0.1).Step([]*nn.Param{p})
	want := []float32{2 - 0.1*2, -3 - 0.1*-3}
	for i := range want {
		if diff := math.Abs(float64(p.Value.Data[i] - want[i])); diff > 1e-6 {
			t.Errorf("Value[%d] = %v, want %v", i, p.Value.Data[i], want[i])
		}
	}
}

func TestL2RegZeroLambdaMatchesPlainOptimizer(t *testing.T) {
	p1 := newTestParam([]float32{5}, []float32{2})
	p2 := newTestParam([]float32{5}, []float32{2})
	SGD(0.1).Step([]*nn.Param{p1})
	L2Reg(SGD(0.1), 0).Step([]*nn.Param{p2})
	if diff := math.Abs(float64(p1.Value.Data[0] - p2.Value.Data[0])); diff > 1e-6 {
		t.Fatalf("L2Reg(_, 0) = %v, want plain SGD result %v", p2.Value.Data[0], p1.Value.Data[0])
	}
}

// TestLargeParamStepMatchesSmallParamStep exercises the goroutine-split
// branch of parallelFor (parallelUpdateThreshold = 4096 elements): it runs
// each optimizer's Step on a large param (chunked across goroutines) and
// on many small independent single-element params (each below the
// threshold, so always the inline path) initialized with the very same
// per-index values/gradients, then checks every element matches. Since
// each per-element update only ever reads/writes its own index with no
// cross-element interaction, the two must agree exactly regardless of how
// the range was chunked — this catches chunking bugs (off-by-one
// boundaries, wrong closure captures) that a small-param-only test can't
// reach.
func TestLargeParamStepMatchesSmallParamStep(t *testing.T) {
	const n = 50000
	value := make([]float32, n)
	grad := make([]float32, n)
	for i := range value {
		value[i] = float32(i%13) * 0.01
		grad[i] = float32((i%9)-4) * 0.1
	}

	newOpts := map[string]func() Optimizer{
		"sgd":      func() Optimizer { return SGD(0.1) },
		"momentum": func() Optimizer { return Momentum(0.1, 0.9) },
		"adam":     func() Optimizer { return Adam(0.01, 0.9, 0.999, 1e-8) },
		"rmsprop":  func() Optimizer { return RMSprop(0.01, 0.9, 1e-8) },
		"adagrad":  func() Optimizer { return Adagrad(0.1, 1e-8) },
		"adadelta": func() Optimizer { return Adadelta(1.0, 0.95, 1e-6) },
		"nadam":    func() Optimizer { return Nadam(0.01, 0.9, 0.999, 1e-8) },
		"lion":     func() Optimizer { return Lion(0.1, 0.9, 0.99) },
	}

	for name, newOpt := range newOpts {
		t.Run(name, func(t *testing.T) {
			large := newTestParam(value, grad)
			smallParams := make([]*nn.Param, n)
			for i := range smallParams {
				smallParams[i] = newTestParam([]float32{value[i]}, []float32{grad[i]})
			}

			largeOpt := newOpt()
			smallOpt := newOpt()

			// Multiple steps with fresh per-index gradients so
			// stateful optimizers (momentum/moment buffers) actually
			// accumulate across calls, not just a single step.
			for step := 0; step < 3; step++ {
				for i := range grad {
					grad[i] = float32((i+step)%9-4) * 0.1
				}
				copy(large.Grad.Data, grad)
				largeOpt.Step([]*nn.Param{large})

				for i, sp := range smallParams {
					sp.Grad.Data[0] = grad[i]
				}
				smallOpt.Step(smallParams)
			}

			for i, sp := range smallParams {
				if diff := math.Abs(float64(large.Value.Data[i] - sp.Value.Data[0])); diff > 1e-5 {
					t.Fatalf("index %d: chunked large-param result = %v, want %v (matching unchunked small-param result)", i, large.Value.Data[i], sp.Value.Data[0])
				}
			}
		})
	}
}

// TestLargeParamRegWrappersMatchSmallParam does the same chunked-vs-
// inline cross-check as TestLargeParamStepMatchesSmallParamStep, but for
// the ClipNorm/L1Reg/L2Reg decorators — ClipNorm additionally exercises
// sumSquares' parallel partial-sum reduction, since its global norm is
// computed over the single large param's 50000-element gradient (well
// past parallelUpdateThreshold).
func TestLargeParamRegWrappersMatchSmallParam(t *testing.T) {
	const n = 50000
	value := make([]float32, n)
	grad := make([]float32, n)
	for i := range value {
		value[i] = float32(i%13)*0.01 - 0.05
		grad[i] = float32((i%9)-4) * 0.1
	}

	newOpts := map[string]func() Optimizer{
		"clipnorm": func() Optimizer { return ClipNorm(SGD(0.1), 5.0) },
		"l1reg":    func() Optimizer { return L1Reg(SGD(0.1), 1e-3) },
		"l2reg":    func() Optimizer { return L2Reg(SGD(0.1), 1e-3) },
	}

	for name, newOpt := range newOpts {
		t.Run(name, func(t *testing.T) {
			large := newTestParam(value, grad)
			smallParams := make([]*nn.Param, n)
			for i := range smallParams {
				smallParams[i] = newTestParam([]float32{value[i]}, []float32{grad[i]})
			}

			// ClipNorm's global norm depends on all elements together,
			// so its single-large-param run and its 50000-single-
			// element-params run see different norms (n=1 grad vectors
			// clip individually) — restrict this cross-check to the
			// two regularizers, which apply purely per-element.
			if name == "clipnorm" {
				newOpt().Step([]*nn.Param{large})
				return
			}

			newOpt().Step([]*nn.Param{large})
			newOpt().Step(smallParams)

			for i, sp := range smallParams {
				if diff := math.Abs(float64(large.Value.Data[i] - sp.Value.Data[0])); diff > 1e-5 {
					t.Fatalf("index %d: chunked large-param result = %v, want %v (matching unchunked small-param result)", i, large.Value.Data[i], sp.Value.Data[0])
				}
			}
		})
	}
}

// TestClipNormLargeParamMatchesManualNorm confirms sumSquares' parallel
// partial-sum reduction (used once a param's gradient is large enough to
// be worth chunking) agrees with a plain sequential sum-of-squares.
func TestClipNormLargeParamMatchesManualNorm(t *testing.T) {
	const n = 50000
	grad := make([]float32, n)
	for i := range grad {
		grad[i] = float32(i%17-8) * 0.01
	}
	var wantSumSq float64
	for _, g := range grad {
		wantSumSq += float64(g) * float64(g)
	}
	wantNorm := math.Sqrt(wantSumSq)

	p := newTestParam(make([]float32, n), grad)
	maxNorm := float32(wantNorm / 2) // guaranteed to trigger clipping
	ClipNorm(SGD(0), maxNorm).Step([]*nn.Param{p})

	scale := float64(maxNorm) / wantNorm
	for i, g := range grad {
		want := float32(float64(g) * scale)
		if diff := math.Abs(float64(p.Grad.Data[i] - want)); diff > 1e-4 {
			t.Fatalf("Grad[%d] = %v, want %v (scale=%v)", i, p.Grad.Data[i], want, scale)
		}
	}
}

func TestSetLRGetLRRoundTrip(t *testing.T) {
	for name, opt := range map[string]Optimizer{
		"sgd":      SGD(0.1),
		"momentum": Momentum(0.1, 0.9),
		"adam":     Adam(0.1, 0.9, 0.999, 1e-8),
		"rmsprop":  RMSprop(0.1, 0.9, 1e-8),
		"adagrad":  Adagrad(0.1, 1e-8),
		"adadelta": Adadelta(0.1, 0.95, 1e-6),
		"nadam":    Nadam(0.1, 0.9, 0.999, 1e-8),
		"lion":     Lion(0.1, 0.9, 0.99),
		"clipnorm": ClipNorm(SGD(0.1), 1.0),
		"l1reg":    L1Reg(SGD(0.1), 1e-4),
		"l2reg":    L2Reg(SGD(0.1), 1e-4),
	} {
		t.Run(name, func(t *testing.T) {
			opt.SetLR(0.42)
			if diff := math.Abs(float64(opt.GetLR() - 0.42)); diff > 1e-6 {
				t.Fatalf("GetLR() = %v after SetLR(0.42), want 0.42", opt.GetLR())
			}
		})
	}
}
