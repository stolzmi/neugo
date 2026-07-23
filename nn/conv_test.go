// nn/conv_test.go
package nn

import "testing"

func TestConv2DOutputShapeValid(t *testing.T) {
	rng := NewRNG(1)
	c := Conv2D(rng, 1, 4, 3, HeInit())
	out, err := c.OutputShape([]int{2, 8, 8, 1})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 6, 6, 4} // (8-3)/1+1 = 6
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestConv2DSamePreservesSpatialDims(t *testing.T) {
	rng := NewRNG(1)
	c := Conv2DSame(rng, 1, 4, 3, HeInit())
	out, err := c.OutputShape([]int{2, 8, 8, 1})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	if out[1] != 8 || out[2] != 8 {
		t.Fatalf("Conv2DSame output spatial dims = %v, want [.. 8 8 ..]", out)
	}
}

func TestConv2DInputGradient(t *testing.T) {
	rng := NewRNG(2)
	c := Conv2D(rng, 2, 3, 3, HeInit())
	if _, err := c.OutputShape([]int{1, 5, 5, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 5, 5, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.1 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
}

func TestConv2DParamGradients(t *testing.T) {
	rng := NewRNG(3)
	c := Conv2D(rng, 2, 3, 3, HeInit())
	if _, err := c.OutputShape([]int{1, 5, 5, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 5, 5, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.15 - 0.3
	}
	ctx := &Context{Mode: Inference}
	forward := func() (*Tensor, error) { return c.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return c.Backward(ctx, g) }
	for _, p := range c.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestConv2DNegativePaddingReturnsErrorNotPanic(t *testing.T) {
	rng := NewRNG(5)
	c := newConv2D(rng, 1, 4, 3, -1, 1, HeInit())
	_, err := c.OutputShape([]int{2, 8, 8, 1})
	if err == nil {
		t.Fatal("OutputShape with negative padding returned nil error, want a clean error instead of proceeding")
	}
}

func TestConv2DStridedOutputShape(t *testing.T) {
	rng := NewRNG(6)
	c := Conv2DStrided(rng, 1, 4, 3, 2, 1, HeInit())
	out, err := c.OutputShape([]int{2, 8, 8, 1})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 4, 4, 4} // (8+2*1-3)/2+1 = 4
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestConv2DStridedGradients(t *testing.T) {
	rng := NewRNG(7)
	c := Conv2DStrided(rng, 2, 3, 3, 2, 1, HeInit())
	if _, err := c.OutputShape([]int{1, 7, 7, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 7, 7, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%6)*0.12 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
	forward := func() (*Tensor, error) { return c.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return c.Backward(ctx, g) }
	for _, p := range c.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestConv2DSameGradients(t *testing.T) {
	rng := NewRNG(4)
	c := Conv2DSame(rng, 1, 2, 3, HeInit())
	if _, err := c.OutputShape([]int{1, 4, 4, 1}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 4, 4, 1})
	for i := range x.Data {
		x.Data[i] = float32(i%3)*0.2 - 0.2
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, c, ctx, x)
}

// FuzzConv2DGradient fuzzes over shape dimensions ("same" padding, so any
// odd kernel size is valid regardless of spatial size — see Conv2DSame's
// doc comment), looking for any combination where the analytic and
// numeric gradients disagree.
func FuzzConv2DGradient(f *testing.F) {
	f.Add(1, 4, 4, 1, 2, 1)
	f.Add(2, 3, 3, 2, 1, 2)
	f.Fuzz(func(t *testing.T, batch, h, w, inChannels, outChannels, kernelPick int) {
		batch = clampDim(batch, 1, 3)
		h = clampDim(h, 2, 6)
		w = clampDim(w, 2, 6)
		inChannels = clampDim(inChannels, 1, 3)
		outChannels = clampDim(outChannels, 1, 3)
		kernelSize := clampDim(kernelPick, 0, 2)*2 + 1 // odd: 1, 3, or 5

		rng := NewRNG(1)
		c := Conv2DSame(rng, inChannels, outChannels, kernelSize, HeInit())
		if _, err := c.OutputShape([]int{batch, h, w, inChannels}); err != nil {
			t.Skip("shape combination rejected by OutputShape:", err)
		}
		x := NewTensor([]int{batch, h, w, inChannels})
		for i := range x.Data {
			x.Data[i] = float32((i*7+h+w)%11)*0.05 - 0.25
		}
		ctx := &Context{Mode: Train}
		checkInputGradient(t, c, ctx, x)
		forward := func() (*Tensor, error) { return c.Forward(ctx, x) }
		backward := func(g *Tensor) (*Tensor, error) { return c.Backward(ctx, g) }
		for _, p := range c.Params() {
			checkParamGradient(t, forward, backward, p)
		}
	})
}
