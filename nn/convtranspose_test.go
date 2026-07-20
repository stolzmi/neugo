package nn

import "testing"

func TestConvTranspose2DOutputShapeUpsamples(t *testing.T) {
	rng := NewRNG(1)
	c := ConvTranspose2D(rng, 4, 2, 3, 2, 1, HeInit())
	out, err := c.OutputShape([]int{2, 4, 4, 4})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 7, 7, 2} // (4-1)*2-2*1+3 = 7
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestConvTranspose2DGradientsStrideOne(t *testing.T) {
	rng := NewRNG(2)
	c := ConvTranspose2D(rng, 2, 3, 3, 1, 1, HeInit())
	if _, err := c.OutputShape([]int{1, 4, 4, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 4, 4, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.09 - 0.28
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
	forward := func() (*Tensor, error) { return c.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return c.Backward(ctx, g) }
	for _, p := range c.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestConvTranspose2DGradientsStrideTwoOverlap(t *testing.T) {
	// stride 2 < kernel 3 forces overlapping scatter windows in Forward
	// and overlapping gather windows in Backward — the case most likely
	// to expose an accumulation bug.
	rng := NewRNG(3)
	c := ConvTranspose2D(rng, 2, 2, 3, 2, 1, HeInit())
	if _, err := c.OutputShape([]int{1, 3, 3, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 3, 3, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.13 - 0.26
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
	forward := func() (*Tensor, error) { return c.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return c.Backward(ctx, g) }
	for _, p := range c.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

// TestConvTranspose2DInvertsConv2DShape confirms the everyday usage
// pattern — a downsampling Conv2DStrided followed by a ConvTranspose2D
// with matching kernel/stride/padding restores the original spatial
// size, the property that makes it useful in decoders/segmentation heads.
func TestConvTranspose2DInvertsConv2DShape(t *testing.T) {
	rng := NewRNG(4)
	down := Conv2DStrided(rng, 3, 8, 3, 2, 1, HeInit())
	up := ConvTranspose2D(rng, 8, 3, 3, 2, 1, HeInit())
	mid, err := down.OutputShape([]int{1, 9, 9, 3})
	if err != nil {
		t.Fatalf("down.OutputShape: %v", err)
	}
	out, err := up.OutputShape(mid)
	if err != nil {
		t.Fatalf("up.OutputShape: %v", err)
	}
	if out[1] != 9 || out[2] != 9 {
		t.Fatalf("round-tripped spatial dims = %v, want 9x9", out)
	}
}
