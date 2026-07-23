package nn

import (
	"math"
	"testing"
)

func TestRotaryMultiHeadAttentionOutputShapePreserved(t *testing.T) {
	rng := NewRNG(1)
	m := RotaryMultiHeadAttention(rng, 8, 2, false, XavierInit())
	out, err := m.OutputShape([]int{2, 5, 8})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	want := []int{2, 5, 8}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("OutputShape = %v, want %v", out, want)
		}
	}
}

func TestRotaryMultiHeadAttentionRejectsIndivisibleHeads(t *testing.T) {
	rng := NewRNG(2)
	m := RotaryMultiHeadAttention(rng, 8, 3, false, XavierInit())
	if _, err := m.OutputShape([]int{1, 4, 8}); err == nil {
		t.Fatal("OutputShape with dModel not divisible by numHeads returned nil error, want an error")
	}
}

func TestRotaryMultiHeadAttentionRejectsOddHeadDim(t *testing.T) {
	// dModel/numHeads = 3, odd — RoPE needs pairs of dimensions.
	rng := NewRNG(3)
	m := RotaryMultiHeadAttention(rng, 6, 2, false, XavierInit())
	if _, err := m.OutputShape([]int{1, 4, 6}); err == nil {
		t.Fatal("OutputShape with odd head dimension returned nil error, want an error")
	}
}

func TestRotaryMultiHeadAttentionNonCausalGradients(t *testing.T) {
	rng := NewRNG(4)
	m := RotaryMultiHeadAttention(rng, 4, 2, false, XavierInit())
	x := NewTensor([]int{2, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%9)*0.1 - 0.4
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, m, ctx, x)
	forward := func() (*Tensor, error) { return m.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return m.Backward(ctx, g) }
	for _, p := range m.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestRotaryMultiHeadAttentionCausalGradients(t *testing.T) {
	rng := NewRNG(5)
	m := RotaryMultiHeadAttention(rng, 4, 2, true, XavierInit())
	x := NewTensor([]int{1, 4, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.11 - 0.3
	}
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, m, ctx, x)
	forward := func() (*Tensor, error) { return m.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return m.Backward(ctx, g) }
	for _, p := range m.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}

func TestRotaryMultiHeadAttentionCausalMaskBlocksFuture(t *testing.T) {
	rng := NewRNG(6)
	m := RotaryMultiHeadAttention(rng, 4, 2, true, XavierInit())
	ctx := &Context{Mode: Inference}
	x := NewTensor([]int{1, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.2
	}
	out1, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	pos0Before := append([]float32(nil), out1.Data[0:4]...)

	x2 := NewTensor([]int{1, 3, 4})
	copy(x2.Data, x.Data)
	for i := 8; i < 12; i++ {
		x2.Data[i] = 100 // change position 2 only
	}
	out2, err := m.Forward(ctx, x2)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := 0; i < 4; i++ {
		if diff := abs32(pos0Before[i] - out2.Data[i]); diff > 1e-5 {
			t.Errorf("causal position 0 output[%d] changed from %v to %v when a later position changed", i, pos0Before[i], out2.Data[i])
		}
	}
}

// TestRoPERelativePositionInvariance checks RoPE's defining property
// directly against the rotation primitives: the dot product of a rotated
// query and a rotated key depends only on the *difference* between their
// positions, not on the absolute positions themselves.
func TestRoPERelativePositionInvariance(t *testing.T) {
	const dHead = 4
	m := &RotaryMultiHeadAttentionLayer{dModel: dHead, numHeads: 1, dHead: dHead, ropeBase: 10000}
	m.buildRopeTables(200)

	q0 := []float32{0.6, -0.3, 0.9, 0.2}
	k0 := []float32{0.1, 0.4, -0.5, 0.7}

	dot := func(posQ, posK int) float32 {
		rq := make([]float32, dHead)
		rk := make([]float32, dHead)
		m.rotateRow(q0, rq, posQ)
		m.rotateRow(k0, rk, posK)
		var d float32
		for i := range rq {
			d += rq[i] * rk[i]
		}
		return d
	}

	// Offset 2, at two very different absolute position pairs.
	d1 := dot(5, 3)
	d2 := dot(105, 103)
	if diff := math.Abs(float64(d1 - d2)); diff > 1e-4 {
		t.Fatalf("dot products at the same relative offset (2) differ: %v (pos 5,3) vs %v (pos 105,103)", d1, d2)
	}

	// Offset 0 (same position), at two different absolute positions.
	d3 := dot(5, 5)
	d4 := dot(50, 50)
	if diff := math.Abs(float64(d3 - d4)); diff > 1e-4 {
		t.Fatalf("dot products at offset 0 differ: %v (pos 5,5) vs %v (pos 50,50)", d3, d4)
	}

	// A genuinely different offset should (generically) give a different
	// dot product — otherwise this test would pass vacuously.
	d5 := dot(10, 3) // offset 7
	if diff := math.Abs(float64(d1 - d5)); diff < 1e-3 {
		t.Fatalf("dot products at different offsets (2 vs 7) unexpectedly matched: %v vs %v", d1, d5)
	}
}

func TestRoPEUnrotateIsInverseOfRotate(t *testing.T) {
	const dHead = 4
	m := &RotaryMultiHeadAttentionLayer{dModel: dHead, numHeads: 1, dHead: dHead, ropeBase: 10000}
	m.buildRopeTables(10)

	src := []float32{0.3, -0.7, 1.1, 0.05}
	rotated := make([]float32, dHead)
	m.rotateRow(src, rotated, 7)
	recovered := make([]float32, dHead)
	m.unrotateRow(rotated, recovered, 7)
	for i := range src {
		if diff := math.Abs(float64(src[i] - recovered[i])); diff > 1e-5 {
			t.Errorf("recovered[%d] = %v, want %v (original)", i, recovered[i], src[i])
		}
	}
}
