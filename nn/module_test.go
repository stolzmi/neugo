package nn

import "testing"

func TestNewParamGradShapeMatchesValue(t *testing.T) {
	v, _ := NewTensorFromData([]float32{1, 2, 3}, []int{3})
	p := NewParam(v)
	if p.Grad.Size() != p.Value.Size() {
		t.Fatalf("Grad size = %d, want %d", p.Grad.Size(), p.Value.Size())
	}
}

func TestParamZeroGrad(t *testing.T) {
	v := NewTensor([]int{2})
	p := NewParam(v)
	p.Grad.Data[0], p.Grad.Data[1] = 5, -3
	p.ZeroGrad()
	for i, g := range p.Grad.Data {
		if g != 0 {
			t.Errorf("Grad[%d] = %v after ZeroGrad, want 0", i, g)
		}
	}
}
