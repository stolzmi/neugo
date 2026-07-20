package nn

import "testing"

func TestLayerNormOutputShapeRejectsWrongChannels(t *testing.T) {
	l := LayerNorm(6)
	if _, err := l.OutputShape([]int{2, 5}); err == nil {
		t.Fatal("OutputShape with wrong last-dim size returned nil error, want an error")
	}
}

func TestLayerNormDenseGradients(t *testing.T) {
	l := LayerNorm(6)
	x := NewTensor([]int{3, 6})
	for i := range x.Data {
		x.Data[i] = float32(i%9)*0.11 - 0.4
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, l, ctx, x)
	forward := func() (*Tensor, error) { return l.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return l.Backward(ctx, g) }
	for _, p := range l.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestLayerNormSequenceGradients(t *testing.T) {
	l := LayerNorm(4)
	x := NewTensor([]int{2, 3, 4}) // [batch, seqLen, channels]
	for i := range x.Data {
		x.Data[i] = float32(i%11)*0.09 - 0.35
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, l, ctx, x)
	forward := func() (*Tensor, error) { return l.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return l.Backward(ctx, g) }
	for _, p := range l.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

// TestLayerNormPositionsAreIndependent is the regression test for the bug
// this task fixes: changing one sequence position's values must not
// change another position's normalized output, in either the same or a
// different batch element. The old LayerNorm = GroupNorm(1, channels)
// pooled statistics across the whole sequence per sample and would fail
// this test.
func TestLayerNormPositionsAreIndependent(t *testing.T) {
	l := LayerNorm(3)
	ctx := &Context{Mode: Train}
	x := NewTensor([]int{1, 2, 3}) // [batch=1, seqLen=2, channels=3]
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.4
	}
	out1, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	pos0Before := append([]float32(nil), out1.Data[0:3]...)

	x2 := NewTensor([]int{1, 2, 3})
	copy(x2.Data[0:3], x.Data[0:3])
	for i := 3; i < 6; i++ {
		x2.Data[i] = 100 // wildly different second position
	}
	out2, err := l.Forward(ctx, x2)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := 0; i < 3; i++ {
		diff := pos0Before[i] - out2.Data[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-5 {
			t.Errorf("position 0 output[%d] changed from %v to %v when position 1 changed", i, pos0Before[i], out2.Data[i])
		}
	}
}
