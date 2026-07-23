package nn

import "testing"

func TestRNNOutputShape(t *testing.T) {
	r := RNN(NewRNG(1), 3, 5, nil)
	out, err := r.OutputShape([]int{2, 4, 3})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 4, 5}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestRNNRejectsWrongFeatureCount(t *testing.T) {
	r := RNN(NewRNG(1), 3, 5, nil)
	if _, err := r.OutputShape([]int{2, 4, 7}); err == nil {
		t.Fatal("expected error for feature mismatch, got nil")
	}
}

func TestRNNRejectsNon3DInput(t *testing.T) {
	r := RNN(NewRNG(1), 3, 5, nil)
	if _, err := r.OutputShape([]int{2, 3}); err == nil {
		t.Fatal("expected error for non-3D input, got nil")
	}
}

func TestRNNGradients(t *testing.T) {
	rng := NewRNG(7)
	r := RNN(rng, 3, 4, XavierInit())
	// batch=2, seqLen=3, features=3 — multi-timestep BPTT.
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%11)*0.07 - 0.35
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, r, ctx, x)
	forward := func() (*Tensor, error) { return r.Forward(ctx, x) }
	backward := func(grad *Tensor) (*Tensor, error) { return r.Backward(ctx, grad) }
	for _, p := range r.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestRNNDeterministicAcrossRepeatedForward(t *testing.T) {
	rng := NewRNG(3)
	r := RNN(rng, 2, 3, XavierInit())
	x := NewTensor([]int{2, 3, 2})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.1
	}
	ctx := &Context{Mode: Inference}
	out1, err := r.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	out2, err := r.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := range out1.Data {
		if out1.Data[i] != out2.Data[i] {
			t.Errorf("output[%d] differs across identical forward calls: %v vs %v", i, out1.Data[i], out2.Data[i])
		}
	}
}
