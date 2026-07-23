package nn

import (
	"math"
	"testing"
)

func TestReLUForward(t *testing.T) {
	x, _ := NewTensorFromData([]float32{-2, -0.5, 0, 1, 3}, []int{5})
	y, err := ReLU().Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []float32{0, 0, 0, 1, 3}
	for i := range want {
		if y.Data[i] != want[i] {
			t.Errorf("ReLU(%v) = %v, want %v", x.Data[i], y.Data[i], want[i])
		}
	}
}

func TestGELUExactFormula(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1.0}, []int{1})
	y, _ := GELU().Forward(&Context{}, x)
	want := float32(0.5 * 1.0 * (1 + math.Erf(1.0/math.Sqrt2)))
	if diff := math.Abs(float64(y.Data[0] - want)); diff > 1e-5 {
		t.Fatalf("GELU(1.0) = %v, want %v", y.Data[0], want)
	}
}

func TestActivationGradients(t *testing.T) {
	x, _ := NewTensorFromData([]float32{-1.5, -0.3, 0.4, 2.1}, []int{4})
	ctx := &Context{Mode: Inference}
	for _, tc := range []struct {
		name string
		m    Module
	}{
		{"relu", ReLU()},
		{"sigmoid", Sigmoid()},
		{"tanh", Tanh()},
		{"leaky_relu", LeakyReLU(0.01)},
		{"gelu", GELU()},
		{"elu", ELU(1.0)},
		{"selu", SELU()},
		{"silu", SiLU()},
		{"softplus", Softplus()},
		{"mish", Mish()},
		{"hardswish", Hardswish()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			xc := x.Clone()
			checkInputGradient(t, tc.m, ctx, xc)
		})
	}
}

func TestSoftmaxRowsSumToOne(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 0, 0, 0}, []int{2, 3})
	y, err := Softmax().Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for b := 0; b < 2; b++ {
		var sum float32
		for c := 0; c < 3; c++ {
			sum += y.Data[b*3+c]
		}
		if diff := sum - 1; diff > 1e-5 || diff < -1e-5 {
			t.Errorf("row %d sums to %v, want 1", b, sum)
		}
	}
}

func TestSoftmaxGradient(t *testing.T) {
	x, _ := NewTensorFromData([]float32{0.2, 1.5, -0.3, 2.0, 0.1, -1.0}, []int{2, 3})
	ctx := &Context{Mode: Inference}
	checkInputGradient(t, Softmax(), ctx, x)
}
