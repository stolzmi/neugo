package nn

import "testing"

func TestSequentialValidChain(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{4, 3},
		Linear(rng, 0, 5, XavierInit()),
		ReLU(),
		Linear(rng, 0, 2, XavierInit()),
		Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{4, 3})
	y, err := model.Forward(&Context{Mode: Inference}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if y.Shape[0] != 4 || y.Shape[1] != 2 {
		t.Fatalf("output shape = %v, want [4 2]", y.Shape)
	}
}

func TestSequentialRejectsMismatchedChain(t *testing.T) {
	rng := NewRNG(1)
	_, err := Sequential([]int{4, 3},
		Linear(rng, 5, 5, XavierInit()), // configured for 5 input features, gets 3
		ReLU(),
	)
	if err == nil {
		t.Fatal("expected error for mismatched Linear input size, got nil")
	}
}

func TestSequentialBackwardMatchesParamCount(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 3}, Linear(rng, 0, 4, XavierInit()), ReLU())
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	ctx := &Context{Mode: Inference}
	x := NewTensor([]int{2, 3})
	y, _ := model.Forward(ctx, x)
	gradOut := NewTensor(y.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradIn, err := model.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if gradIn.Shape[0] != 2 || gradIn.Shape[1] != 3 {
		t.Fatalf("input gradient shape = %v, want [2 3]", gradIn.Shape)
	}
	if len(model.Params()) != 2 { // W, B of the one Linear
		t.Fatalf("Params() len = %d, want 2", len(model.Params()))
	}
}
