package nn

import "testing"

func TestGRUOutputShape(t *testing.T) {
	g := GRU(NewRNG(1), 3, 5, nil)
	out, err := g.OutputShape([]int{2, 4, 3})
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

func TestGRURejectsWrongFeatureCount(t *testing.T) {
	g := GRU(NewRNG(1), 3, 5, nil)
	if _, err := g.OutputShape([]int{2, 4, 7}); err == nil {
		t.Fatal("expected error for feature mismatch, got nil")
	}
}

func TestGRUGateWeightShapes(t *testing.T) {
	g := GRU(NewRNG(1), 3, 5, nil)
	if len(g.Wx.Value.Data) != 3*3*5 {
		t.Errorf("Wx has %d values, want %d (features*3*hidden)", len(g.Wx.Value.Data), 3*3*5)
	}
	if len(g.Wh.Value.Data) != 5*3*5 {
		t.Errorf("Wh has %d values, want %d (hidden*3*hidden)", len(g.Wh.Value.Data), 5*3*5)
	}
	if len(g.Bx.Value.Data) != 3*5 || len(g.Bh.Value.Data) != 3*5 {
		t.Errorf("Bx/Bh have %d/%d values, want %d each (3*hidden)", len(g.Bx.Value.Data), len(g.Bh.Value.Data), 3*5)
	}
}

func TestGRUGradients(t *testing.T) {
	rng := NewRNG(13)
	g := GRU(rng, 3, 4, XavierInit())
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%13)*0.06 - 0.35
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, g, ctx, x)
	forward := func() (*Tensor, error) { return g.Forward(ctx, x) }
	backward := func(grad *Tensor) (*Tensor, error) { return g.Backward(ctx, grad) }
	for _, p := range g.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestGRUSingleTimestepMatchesGateEquations(t *testing.T) {
	// batch=1, seqLen=1, features=1, hidden=1, all-zero weights: r=z=0.5,
	// n=tanh(0)=0, h_prev=0 -> h_0 = (1-0.5)*0 + 0.5*0 = 0.
	g := GRU(NewRNG(1), 1, 1, ZerosInit())
	x, _ := NewTensorFromData([]float32{0}, []int{1, 1, 1})
	out, err := g.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out.Data[0] != 0 {
		t.Fatalf("h_0 = %v, want 0", out.Data[0])
	}
}
