package nn

import "testing"

func TestMultiHeadAttentionOutputShapePreserved(t *testing.T) {
	rng := NewRNG(1)
	m := MultiHeadAttention(rng, 8, 2, false, XavierInit())
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

func TestMultiHeadAttentionRejectsIndivisibleHeads(t *testing.T) {
	rng := NewRNG(2)
	m := MultiHeadAttention(rng, 8, 3, false, XavierInit())
	if _, err := m.OutputShape([]int{1, 4, 8}); err == nil {
		t.Fatal("OutputShape with dModel not divisible by numHeads returned nil error, want an error")
	}
}

func TestMultiHeadAttentionNonCausalGradients(t *testing.T) {
	rng := NewRNG(3)
	m := MultiHeadAttention(rng, 4, 2, false, XavierInit())
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

func TestMultiHeadAttentionCausalGradients(t *testing.T) {
	rng := NewRNG(4)
	m := MultiHeadAttention(rng, 4, 2, true, XavierInit())
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

// TestMultiHeadAttentionCausalMaskBlocksFuture confirms the causal mask
// is actually applied: position 0's output must not change when a later
// position's input changes (it can only attend to itself).
func TestMultiHeadAttentionCausalMaskBlocksFuture(t *testing.T) {
	rng := NewRNG(5)
	m := MultiHeadAttention(rng, 4, 2, true, XavierInit())
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
		diff := pos0Before[i] - out2.Data[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-5 {
			t.Errorf("causal position 0 output[%d] changed from %v to %v when a later position changed", i, pos0Before[i], out2.Data[i])
		}
	}
}

func TestCrossAttentionOutputShapeMatchesQuery(t *testing.T) {
	rng := NewRNG(6)
	c := CrossAttention(rng, 4, 2, XavierInit())
	ctx := &Context{Mode: Inference}
	query := NewTensor([]int{2, 3, 4})  // qLen=3
	context := NewTensor([]int{2, 5, 4}) // ctxLen=5, different from qLen
	out, err := c.Forward(ctx, query, context)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []int{2, 3, 4}
	for i := range want {
		if out.Shape[i] != want[i] {
			t.Fatalf("output shape = %v, want %v", out.Shape, want)
		}
	}
}

func TestCrossAttentionGradientsDifferingLengths(t *testing.T) {
	rng := NewRNG(7)
	c := CrossAttention(rng, 4, 2, XavierInit())
	ctx := &Context{Mode: Inference}
	query := NewTensor([]int{1, 2, 4})
	for i := range query.Data {
		query.Data[i] = float32(i%5)*0.1 - 0.2
	}
	context := NewTensor([]int{1, 3, 4})
	for i := range context.Data {
		context.Data[i] = float32(i%6)*0.12 - 0.3
	}

	forward := func() (*Tensor, error) { return c.Forward(ctx, query, context) }
	out, err := forward()
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gradOut := NewTensor(out.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradQuery, gradContext, err := c.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Gradcheck gradQuery.
	for i := range query.Data {
		orig := query.Data[i]
		query.Data[i] = orig + gradCheckEps
		yPlus, _ := forward()
		query.Data[i] = orig - gradCheckEps
		yMinus, _ := forward()
		query.Data[i] = orig
		numGrad := (sumTensor(yPlus) - sumTensor(yMinus)) / (2 * gradCheckEps)
		if diff := abs32(numGrad - gradQuery.Data[i]); diff > gradCheckTol {
			t.Errorf("gradQuery mismatch at %d: analytic=%v numeric=%v", i, gradQuery.Data[i], numGrad)
		}
	}
	forward() // restore cached state

	// Gradcheck gradContext.
	for i := range context.Data {
		orig := context.Data[i]
		context.Data[i] = orig + gradCheckEps
		yPlus, _ := forward()
		context.Data[i] = orig - gradCheckEps
		yMinus, _ := forward()
		context.Data[i] = orig
		numGrad := (sumTensor(yPlus) - sumTensor(yMinus)) / (2 * gradCheckEps)
		if diff := abs32(numGrad - gradContext.Data[i]); diff > gradCheckTol {
			t.Errorf("gradContext mismatch at %d: analytic=%v numeric=%v", i, gradContext.Data[i], numGrad)
		}
	}
	forward()

	for _, p := range c.Params() {
		checkParamGradient(t, forward, func(g *Tensor) (*Tensor, error) {
			gq, _, err := c.Backward(ctx, g)
			return gq, err
		}, p)
	}
}

// FuzzMultiHeadAttentionGradient fuzzes over shape dimensions (batch,
// seqLen, numHeads, headDim, and causal-or-not), looking for any
// combination where the analytic and numeric gradients disagree.
func FuzzMultiHeadAttentionGradient(f *testing.F) {
	f.Add(2, 3, 2, 2, false)
	f.Add(1, 4, 1, 4, true)
	f.Fuzz(func(t *testing.T, batch, seqLen, numHeads, headDim int, causal bool) {
		batch = clampDim(batch, 1, 3)
		seqLen = clampDim(seqLen, 1, 4)
		numHeads = clampDim(numHeads, 1, 3)
		headDim = clampDim(headDim, 1, 3)
		dModel := numHeads * headDim

		rng := NewRNG(1)
		m := MultiHeadAttention(rng, dModel, numHeads, causal, XavierInit())
		x := NewTensor([]int{batch, seqLen, dModel})
		for i := range x.Data {
			x.Data[i] = float32((i*7+seqLen+dModel)%11)*0.05 - 0.25
		}
		ctx := &Context{Mode: Train}
		checkInputGradient(t, m, ctx, x)
		forward := func() (*Tensor, error) { return m.Forward(ctx, x) }
		backward := func(g *Tensor) (*Tensor, error) { return m.Backward(ctx, g) }
		for _, p := range m.Params() {
			checkParamGradient(t, forward, backward, p)
		}
	})
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
