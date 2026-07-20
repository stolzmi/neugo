package nn

import "testing"

func TestEmbeddingOutputShape(t *testing.T) {
	rng := NewRNG(1)
	e := Embedding(rng, 10, 4, nil)
	out, err := e.OutputShape([]int{2, 3})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 3, 4}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestEmbeddingLooksUpCorrectRows(t *testing.T) {
	rng := NewRNG(2)
	e := Embedding(rng, 5, 3, NormalInit(0, 1))
	x, err := NewTensorFromData([]float32{2, 0, 4, 1}, []int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &Context{Mode: Inference}
	out, err := e.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for row, idx := range []int{2, 0, 4, 1} {
		want := e.W.Value.Data[idx*3 : idx*3+3]
		got := out.Data[row*3 : row*3+3]
		for j := range want {
			if got[j] != want[j] {
				t.Errorf("row %d (index %d) = %v, want %v", row, idx, got, want)
			}
		}
	}
}

func TestEmbeddingOutOfRangeIndexIsAnError(t *testing.T) {
	rng := NewRNG(3)
	e := Embedding(rng, 3, 2, nil)
	x, _ := NewTensorFromData([]float32{0, 5}, []int{1, 2})
	ctx := &Context{Mode: Inference}
	if _, err := e.Forward(ctx, x); err == nil {
		t.Fatal("Forward with out-of-range index returned nil error, want an error")
	}
}

func TestEmbeddingBackwardReturnsZeroInputGradient(t *testing.T) {
	rng := NewRNG(4)
	e := Embedding(rng, 5, 3, nil)
	x, _ := NewTensorFromData([]float32{1, 2}, []int{1, 2})
	ctx := &Context{Mode: Train}
	out, err := e.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gradOut := NewTensor(out.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradIn, err := e.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	for i, v := range gradIn.Data {
		if v != 0 {
			t.Errorf("gradIn[%d] = %v, want 0 (indices are not differentiable)", i, v)
		}
	}
}

// TestEmbeddingWeightGradientAccumulatesRepeats checks the scatter-add:
// index 2 appears twice, so its embedding row's gradient must be the sum
// of gradOut's two rows for those positions, verified by finite
// differences via checkParamGradient.
func TestEmbeddingWeightGradientAccumulatesRepeats(t *testing.T) {
	rng := NewRNG(5)
	e := Embedding(rng, 4, 3, NormalInit(0, 0.5))
	x, _ := NewTensorFromData([]float32{2, 1, 2}, []int{1, 3})
	ctx := &Context{Mode: Train}
	forward := func() (*Tensor, error) { return e.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return e.Backward(ctx, g) }
	checkParamGradient(t, forward, backward, e.W)
}
