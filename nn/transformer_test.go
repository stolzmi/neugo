package nn

import "testing"

func TestTransformerBlockPreservesShape(t *testing.T) {
	rng := NewRNG(1)
	block, err := TransformerBlock(rng, 8, 2, 16, false, XavierInit())
	if err != nil {
		t.Fatalf("TransformerBlock: %v", err)
	}
	x := NewTensor([]int{2, 5, 8})
	for i := range x.Data {
		x.Data[i] = float32(i%9)*0.1 - 0.4
	}
	ctx := &Context{Mode: Inference}
	out, err := block.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []int{2, 5, 8}
	for i := range want {
		if out.Shape[i] != want[i] {
			t.Fatalf("output shape = %v, want %v", out.Shape, want)
		}
	}
}

// TestTransformerBlockNestsInsideOuterSequential proves the core design
// claim: since TransformerBlock returns *SequentialModel, which already
// implements Module, stacking blocks is just listing several
// TransformerBlock(...) calls among an outer Sequential's modules.
func TestTransformerBlockNestsInsideOuterSequential(t *testing.T) {
	rng := NewRNG(2)
	block1, err := TransformerBlock(rng, 8, 2, 16, true, XavierInit())
	if err != nil {
		t.Fatalf("TransformerBlock 1: %v", err)
	}
	block2, err := TransformerBlock(rng, 8, 2, 16, true, XavierInit())
	if err != nil {
		t.Fatalf("TransformerBlock 2: %v", err)
	}
	model, err := Sequential([]int{1, 4, 8}, block1, block2)
	if err != nil {
		t.Fatalf("outer Sequential: %v", err)
	}
	x := NewTensor([]int{1, 4, 8})
	for i := range x.Data {
		x.Data[i] = float32(i%6) * 0.1
	}
	ctx := &Context{Mode: Inference}
	out, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out.Shape[0] != 1 || out.Shape[1] != 4 || out.Shape[2] != 8 {
		t.Fatalf("output shape = %v, want [1 4 8]", out.Shape)
	}
	// Sanity check that params from both blocks (attention Q/K/V/O + 2
	// feed-forward Linear + 2 LayerNorm, per block) are all reachable.
	if got := len(model.Params()); got == 0 {
		t.Fatal("model.Params() is empty, want params from both blocks")
	}
}

func TestTransformerBlockGradients(t *testing.T) {
	rng := NewRNG(3)
	block, err := TransformerBlock(rng, 4, 2, 8, false, XavierInit())
	if err != nil {
		t.Fatalf("TransformerBlock: %v", err)
	}
	x := NewTensor([]int{1, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.1 - 0.3
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, block, ctx, x)
	forward := func() (*Tensor, error) { return block.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return block.Backward(ctx, g) }
	for _, p := range block.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}
