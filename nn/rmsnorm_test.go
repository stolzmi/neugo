package nn

import (
	"math"
	"testing"
)

func TestRMSNormForwardMatchesDefinition(t *testing.T) {
	r := RMSNorm(4)
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4, -1, -2, -3, -4}, []int{2, 4})
	y, err := r.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for row := 0; row < 2; row++ {
		base := row * 4
		var ss float32
		for _, v := range x.Data[base : base+4] {
			ss += v * v
		}
		rms := float32(math.Sqrt(float64(ss/4) + 1e-5))
		for c := 0; c < 4; c++ {
			want := x.Data[base+c] / rms // Gamma starts at 1
			if diff := math.Abs(float64(y.Data[base+c] - want)); diff > 1e-4 {
				t.Errorf("row %d channel %d = %v, want %v", row, c, y.Data[base+c], want)
			}
		}
	}
}

func TestRMSNormRejectsChannelMismatch(t *testing.T) {
	r := RMSNorm(4)
	if _, err := r.OutputShape([]int{2, 3}); err == nil {
		t.Fatal("expected error for channel mismatch, got nil")
	}
}

func TestRMSNormGradients(t *testing.T) {
	r := RMSNorm(5)
	for i := range r.Gamma.Value.Data {
		r.Gamma.Value.Data[i] = 1 + float32(i)*0.1
	}
	x := NewTensor([]int{3, 5})
	for i := range x.Data {
		x.Data[i] = float32(i%9)*0.13 - 0.5
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, r, ctx, x)
	forward := func() (*Tensor, error) { return r.Forward(ctx, x) }
	backward := func(grad *Tensor) (*Tensor, error) { return r.Backward(ctx, grad) }
	for _, p := range r.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestRMSNormSequenceGradients(t *testing.T) {
	// [batch, seqLen, channels] input, matching LayerNorm's supported
	// shapes. Each row's channel values are spread with amplitude ~1 (not
	// clustered near zero): RMSNorm's backward has an invRMS^3 term, so a
	// row with very small magnitude (and thus very large invRMS) makes
	// finite-difference gradcheck numerically unstable at the shared
	// gradCheckEps, independent of whether the analytic gradient is
	// correct — this is purely a test-data concern, not a formula one.
	r := RMSNorm(3)
	x := NewTensor([]int{2, 4, 3})
	for row := 0; row < 8; row++ {
		for c := 0; c < 3; c++ {
			x.Data[row*3+c] = float32(c)*0.7 - 1.0 + float32(row)*0.05
		}
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, r, ctx, x)
	forward := func() (*Tensor, error) { return r.Forward(ctx, x) }
	backward := func(grad *Tensor) (*Tensor, error) { return r.Backward(ctx, grad) }
	for _, p := range r.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

// FuzzRMSNormGradient fuzzes over shape dimensions (batch, channels),
// looking for any combination where the analytic and numeric gradients
// disagree. Test data uses an amplitude-~1, per-channel-spread pattern
// (not tiny/near-zero-variance values), for the same reason
// TestRMSNormSequenceGradients above does: RMSNorm's backward has an
// invRMS^3 term, so a near-zero-magnitude row makes finite-difference
// gradcheck numerically unstable independent of whether the analytic
// gradient is correct — a test-data concern, not a formula one.
func FuzzRMSNormGradient(f *testing.F) {
	f.Add(2, 4)
	f.Add(1, 1)
	f.Fuzz(func(t *testing.T, batch, channels int) {
		batch = clampDim(batch, 1, 5)
		channels = clampDim(channels, 1, 6)

		r := RMSNorm(channels)
		x := NewTensor([]int{batch, channels})
		for row := 0; row < batch; row++ {
			for c := 0; c < channels; c++ {
				x.Data[row*channels+c] = float32(c)*0.7 - 1.0 + float32(row)*0.05
			}
		}
		ctx := &Context{Mode: Train}
		checkInputGradient(t, r, ctx, x)
		forward := func() (*Tensor, error) { return r.Forward(ctx, x) }
		backward := func(grad *Tensor) (*Tensor, error) { return r.Backward(ctx, grad) }
		for _, p := range r.Params() {
			checkParamGradient(t, forward, backward, p)
		}
	})
}
