package nn

import "testing"

func TestGroupNormRejectsNonDivisibleChannels(t *testing.T) {
	g := GroupNorm(3, 8)
	if _, err := g.OutputShape([]int{2, 8}); err == nil {
		t.Fatal("OutputShape with channels not divisible by groups returned nil error, want an error")
	}
}

func TestGroupNormDenseGradients(t *testing.T) {
	g := GroupNorm(2, 8)
	x := NewTensor([]int{4, 8})
	for i := range x.Data {
		x.Data[i] = float32(i%11)*0.07 - 0.35
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, g, ctx, x)
	forward := func() (*Tensor, error) { return g.Forward(ctx, x) }
	backward := func(grad *Tensor) (*Tensor, error) { return g.Backward(ctx, grad) }
	for _, p := range g.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestGroupNormConvGradients(t *testing.T) {
	g := GroupNorm(2, 4)
	x := NewTensor([]int{2, 3, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%13)*0.05 - 0.3
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, g, ctx, x)
	forward := func() (*Tensor, error) { return g.Forward(ctx, x) }
	backward := func(grad *Tensor) (*Tensor, error) { return g.Backward(ctx, grad) }
	for _, p := range g.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestGroupNormIndependentPerSample(t *testing.T) {
	// GroupNorm's stats must not depend on other samples in the batch —
	// unlike BatchNorm, changing one sample must not change another's
	// output.
	g := GroupNorm(1, 4)
	ctx := &Context{Mode: Train}
	x := NewTensor([]int{2, 4})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.3
	}
	out1, err := g.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	row0 := append([]float32(nil), out1.Data[0:4]...)

	x2 := NewTensor([]int{2, 4})
	copy(x2.Data[0:4], x.Data[0:4])
	for i := 4; i < 8; i++ {
		x2.Data[i] = 100 // wildly different second sample
	}
	out2, err := g.Forward(ctx, x2)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := 0; i < 4; i++ {
		if diff := out1Diff(row0[i], out2.Data[i]); diff > 1e-5 {
			t.Errorf("sample 0 output[%d] changed from %v to %v when sample 1 changed", i, row0[i], out2.Data[i])
		}
	}
}

func out1Diff(a, b float32) float32 {
	if a > b {
		return a - b
	}
	return b - a
}
