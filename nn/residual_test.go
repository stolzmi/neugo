// nn/residual_test.go
package nn

import "testing"

func TestResidualIdentityShortcutOutputShape(t *testing.T) {
	rng := NewRNG(1)
	r := Residual(nil,
		Conv2DSame(rng, 4, 4, 3, HeInit()),
		ReLU(),
		Conv2DSame(rng, 4, 4, 3, HeInit()),
	)
	out, err := r.OutputShape([]int{2, 8, 8, 4})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 8, 8, 4}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestResidualShapeMismatchWithoutShortcutIsAnError(t *testing.T) {
	rng := NewRNG(2)
	// Conv2D (valid padding) shrinks spatial dims, so identity (nil
	// shortcut) can't match — this must be a clean error, not a panic.
	r := Residual(nil, Conv2D(rng, 4, 4, 3, HeInit()))
	if _, err := r.OutputShape([]int{2, 8, 8, 4}); err == nil {
		t.Fatal("OutputShape with mismatched shortcut shape returned nil error, want an error")
	}
}

func TestResidualIdentityGradients(t *testing.T) {
	rng := NewRNG(3)
	// Tanh, not ReLU: a multi-layer chain has a real chance some
	// pre-activation lands within gradCheckEps of a ReLU kink, which is a
	// finite-difference artifact rather than a bug in this block's
	// addition/backward-distribution logic (which is what's under test
	// here — ReLU's own gradient is already covered by activation_test.go).
	r := Residual(nil,
		Conv2DSame(rng, 2, 2, 3, HeInit()),
		Tanh(),
	)
	if _, err := r.OutputShape([]int{1, 5, 5, 2}); err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	x := NewTensor([]int{1, 5, 5, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.1 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, r, ctx, x)
	forward := func() (*Tensor, error) { return r.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return r.Backward(ctx, g) }
	for _, p := range r.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

// TestResidualProjectionShortcutGradients exercises the ResNet-style
// downsampling case: inner halves spatial dims and doubles channels via a
// stride-2 conv, and a stride-2 1x1 projection shortcut brings x to the
// same shape so the two branches can be added.
func TestResidualProjectionShortcutGradients(t *testing.T) {
	rng := NewRNG(4)
	r := Residual(
		Conv2DStrided(rng, 2, 4, 1, 2, 0, HeInit()),
		Conv2DStrided(rng, 2, 4, 3, 2, 1, HeInit()),
		Tanh(), // see TestResidualIdentityGradients on why not ReLU here
		Conv2DSame(rng, 4, 4, 3, HeInit()),
	)
	shape, err := r.OutputShape([]int{1, 6, 6, 2})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{1, 3, 3, 4}
	for i := range want {
		if shape[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", shape, want)
		}
	}
	x := NewTensor([]int{1, 6, 6, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.13 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, r, ctx, x)
	forward := func() (*Tensor, error) { return r.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return r.Backward(ctx, g) }
	for _, p := range r.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestResidualComposesInsideSequential(t *testing.T) {
	rng := NewRNG(5)
	model, err := Sequential([]int{1, 6, 6, 2},
		Residual(nil, Conv2DSame(rng, 2, 2, 3, HeInit()), ReLU()),
		Flatten(),
		Linear(rng, 0, 3, XavierInit()),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 6, 6, 2})
	ctx := &Context{Mode: Inference}
	out, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out.Shape[0] != 1 || out.Shape[1] != 3 {
		t.Fatalf("output shape = %v, want [1 3]", out.Shape)
	}
}
