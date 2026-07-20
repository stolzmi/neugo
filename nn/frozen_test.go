package nn

import "testing"

func TestFrozenHidesParams(t *testing.T) {
	rng := NewRNG(1)
	l := Linear(rng, 3, 2, XavierInit())
	f := Frozen(l)
	if got := f.Params(); got != nil {
		t.Fatalf("Frozen.Params() = %v, want nil", got)
	}
	if len(l.Params()) == 0 {
		t.Fatal("sanity check failed: wrapped Linear reports no params of its own")
	}
}

func TestFrozenForwardBackwardMatchInner(t *testing.T) {
	rng := NewRNG(2)
	l := Linear(rng, 3, 2, XavierInit())
	f := Frozen(l)
	x := NewTensor([]int{2, 3})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.1
	}
	ctx := &Context{Mode: Train}

	outDirect, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gradOut := NewTensor(outDirect.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradDirect, err := l.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	outWrapped, err := f.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Frozen Forward: %v", err)
	}
	gradWrapped, err := f.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Frozen Backward: %v", err)
	}

	for i := range outDirect.Data {
		if outWrapped.Data[i] != outDirect.Data[i] {
			t.Errorf("Forward output[%d] = %v, want %v", i, outWrapped.Data[i], outDirect.Data[i])
		}
	}
	for i := range gradDirect.Data {
		if gradWrapped.Data[i] != gradDirect.Data[i] {
			t.Errorf("Backward gradIn[%d] = %v, want %v", i, gradWrapped.Data[i], gradDirect.Data[i])
		}
	}
}

func TestFrozenComposesInsideSequential(t *testing.T) {
	rng := NewRNG(3)
	model, err := Sequential([]int{1, 3},
		Frozen(Linear(rng, 3, 4, XavierInit())),
		ReLU(),
		Linear(rng, 4, 2, XavierInit()),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	// model.Params() must only surface the second Linear's 2 params
	// (W, B) — the frozen first Linear's are excluded.
	if got := len(model.Params()); got != 2 {
		t.Fatalf("model.Params() has %d entries, want 2 (frozen layer's params excluded)", got)
	}
}
