package nn

import "testing"

func TestConv1DOutputShapeValid(t *testing.T) {
	rng := NewRNG(1)
	c := Conv1D(rng, 1, 4, 3, HeInit())
	out, err := c.OutputShape([]int{2, 8, 1})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 6, 4} // (8-3)/1+1 = 6
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestConv1DSamePreservesLength(t *testing.T) {
	rng := NewRNG(1)
	c := Conv1DSame(rng, 1, 4, 3, HeInit())
	out, err := c.OutputShape([]int{2, 8, 1})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	if out[1] != 8 {
		t.Fatalf("Conv1DSame output length = %v, want 8", out[1])
	}
}

func TestConv1DGradients(t *testing.T) {
	rng := NewRNG(2)
	c := Conv1D(rng, 2, 3, 3, HeInit())
	if _, err := c.OutputShape([]int{1, 7, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 7, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%6)*0.11 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
	forward := func() (*Tensor, error) { return c.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return c.Backward(ctx, g) }
	for _, p := range c.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestConv1DStridedGradients(t *testing.T) {
	rng := NewRNG(3)
	c := Conv1DStrided(rng, 2, 3, 3, 2, 1, HeInit())
	if _, err := c.OutputShape([]int{1, 9, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 9, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.09 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
	forward := func() (*Tensor, error) { return c.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return c.Backward(ctx, g) }
	for _, p := range c.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}
