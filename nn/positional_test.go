package nn

import "testing"

func TestPositionalEmbeddingOutputShapeUnchanged(t *testing.T) {
	rng := NewRNG(1)
	p := PositionalEmbedding(rng, 10, 4, NormalInit(0, 0.1))
	out, err := p.OutputShape([]int{2, 5, 4})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 5, 4}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestPositionalEmbeddingRejectsSeqLenBeyondMaxLen(t *testing.T) {
	rng := NewRNG(2)
	p := PositionalEmbedding(rng, 3, 4, NormalInit(0, 0.1))
	if _, err := p.OutputShape([]int{1, 5, 4}); err == nil {
		t.Fatal("OutputShape with seqLen > maxLen returned nil error, want an error")
	}
}

func TestPositionalEmbeddingSamePositionAddsSameVectorAcrossBatch(t *testing.T) {
	rng := NewRNG(3)
	p := PositionalEmbedding(rng, 5, 3, NormalInit(0, 0.1))
	ctx := &Context{Mode: Inference}
	x := NewTensor([]int{2, 2, 3}) // batch=2, seqLen=2, dModel=3, all zeros
	out, err := p.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// With x all zero, out == the position embedding itself, so position
	// 0's added vector must be identical for both batch elements.
	for c := 0; c < 3; c++ {
		b0 := out.Data[(0*2+0)*3+c]
		b1 := out.Data[(1*2+0)*3+c]
		if b0 != b1 {
			t.Errorf("position 0 channel %d differs across batch: %v vs %v", c, b0, b1)
		}
	}
}

func TestPositionalEmbeddingGradients(t *testing.T) {
	rng := NewRNG(4)
	p := PositionalEmbedding(rng, 5, 3, NormalInit(0, 0.1))
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.1 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, p, ctx, x)
	forward := func() (*Tensor, error) { return p.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return p.Backward(ctx, g) }
	for _, param := range p.Params() {
		checkParamGradient(t, forward, backward, param)
	}
}
