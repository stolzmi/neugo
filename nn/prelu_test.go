package nn

import "testing"

func TestPReLUForward(t *testing.T) {
	p := PReLU(2)
	p.Alpha.Value.Data[0] = 0.2
	p.Alpha.Value.Data[1] = 0.5
	x, _ := NewTensorFromData([]float32{-2, 3, 1, -4}, []int{2, 2})
	y, err := p.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []float32{-2 * 0.2, 3, 1, -4 * 0.5}
	for i := range want {
		if y.Data[i] != want[i] {
			t.Errorf("PReLU output[%d] = %v, want %v", i, y.Data[i], want[i])
		}
	}
}

func TestPReLUGradients(t *testing.T) {
	x, _ := NewTensorFromData([]float32{-1.5, 0.3, -0.4, 2.1, 0.8, -2.2}, []int{3, 2})
	ctx := &Context{Mode: Inference}

	p := PReLU(2)
	p.Alpha.Value.Data[0] = 0.3
	p.Alpha.Value.Data[1] = 0.6
	checkInputGradient(t, p, ctx, x.Clone())

	checkParamGradient(t,
		func() (*Tensor, error) { return p.Forward(ctx, x) },
		func(gradOut *Tensor) (*Tensor, error) { return p.Backward(ctx, gradOut) },
		p.Alpha,
	)
}

func TestPReLUOutputShapeRejectsChannelMismatch(t *testing.T) {
	p := PReLU(3)
	if _, err := p.OutputShape([]int{2, 4}); err == nil {
		t.Fatal("expected error for channel mismatch, got nil")
	}
}
