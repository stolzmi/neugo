// nn/linear_test.go
package nn

import "testing"

func TestLinearForwardShape(t *testing.T) {
	rng := NewRNG(1)
	l := Linear(rng, 3, 4, XavierInit())
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	ctx := &Context{Mode: Inference}
	y, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if y.Shape[0] != 2 || y.Shape[1] != 4 {
		t.Fatalf("output shape = %v, want [2 4]", y.Shape)
	}
}

func TestLinearOutputShapeInfersInFeatures(t *testing.T) {
	rng := NewRNG(1)
	l := Linear(rng, 0, 5, XavierInit())
	out, err := l.OutputShape([]int{8, 12})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	if out[0] != 8 || out[1] != 5 {
		t.Fatalf("OutputShape = %v, want [8 5]", out)
	}
	if len(l.Params()) != 2 || l.Params()[0].Value.Shape[0] != 12 {
		t.Fatalf("Linear did not build W with inferred inFeatures=12: %+v", l.Params()[0].Value.Shape)
	}
}

func TestLinearInputGradient(t *testing.T) {
	rng := NewRNG(2)
	l := Linear(rng, 3, 2, XavierInit())
	x, _ := NewTensorFromData([]float32{0.5, -1.2, 0.3, 1.1, 0.2, -0.7}, []int{2, 3})
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, l, ctx, x)
}

func TestLinearParamGradients(t *testing.T) {
	rng := NewRNG(3)
	l := Linear(rng, 3, 2, XavierInit())
	x, _ := NewTensorFromData([]float32{0.5, -1.2, 0.3, 1.1, 0.2, -0.7}, []int{2, 3})
	ctx := &Context{Mode: Inference}
	forward := func() (*Tensor, error) { return l.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return l.Backward(ctx, g) }
	checkParamGradient(t, forward, backward, l.Params()[0]) // W
	checkParamGradient(t, forward, backward, l.Params()[1]) // B
}

func TestLinearForward3DSequenceInput(t *testing.T) {
	rng := NewRNG(4)
	l := Linear(rng, 3, 2, XavierInit())
	x := NewTensor([]int{2, 4, 3}) // [batch=2, seqLen=4, features=3]
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.1 - 0.2
	}
	ctx := &Context{Mode: Inference}
	y, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []int{2, 4, 2}
	for i := range want {
		if y.Shape[i] != want[i] {
			t.Fatalf("output shape = %v, want %v", y.Shape, want)
		}
	}
}

func TestLinear3DGradients(t *testing.T) {
	rng := NewRNG(5)
	l := Linear(rng, 3, 2, XavierInit())
	x := NewTensor([]int{2, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.08 - 0.25
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, l, ctx, x)
	forward := func() (*Tensor, error) { return l.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return l.Backward(ctx, g) }
	for _, p := range l.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

// FuzzLinearGradient fuzzes over shape dimensions rather than hand-picked
// sizes, looking for any (batch, inFeatures, outFeatures) combination
// where the analytic and numeric gradients disagree.
func FuzzLinearGradient(f *testing.F) {
	f.Add(2, 3, 4)
	f.Add(1, 1, 1)
	f.Fuzz(func(t *testing.T, batch, inFeatures, outFeatures int) {
		batch = clampDim(batch, 1, 6)
		inFeatures = clampDim(inFeatures, 1, 6)
		outFeatures = clampDim(outFeatures, 1, 6)

		rng := NewRNG(1)
		l := Linear(rng, inFeatures, outFeatures, XavierInit())
		x := NewTensor([]int{batch, inFeatures})
		for i := range x.Data {
			x.Data[i] = float32((i*7+batch+inFeatures)%13)*0.05 - 0.3
		}
		ctx := &Context{Mode: Train}
		checkInputGradient(t, l, ctx, x)
		forward := func() (*Tensor, error) { return l.Forward(ctx, x) }
		backward := func(g *Tensor) (*Tensor, error) { return l.Backward(ctx, g) }
		for _, p := range l.Params() {
			checkParamGradient(t, forward, backward, p)
		}
	})
}
