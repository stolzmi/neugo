// nn/gradcheck_test.go
package nn

import (
	"math"
	"testing"
)

const gradCheckEps = 1e-2
const gradCheckTol = 1e-2

func sumTensor(t *Tensor) float32 {
	var s float32
	for _, v := range t.Data {
		s += v
	}
	return s
}

// checkInputGradient verifies m.Backward's returned input-gradient against
// central finite differences of sum(m.Forward(x)), perturbing one element
// of x at a time. Use for modules with no learnable Params, or to check the
// input-gradient path of a module that also has Params (call
// checkParamGradient separately for those).
func checkInputGradient(t *testing.T, m Module, ctx *Context, x *Tensor) {
	t.Helper()
	y, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	gradOut := NewTensor(y.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradIn, err := m.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}
	for i := range x.Data {
		orig := x.Data[i]

		x.Data[i] = orig + gradCheckEps
		yPlus, _ := m.Forward(ctx, x)

		x.Data[i] = orig - gradCheckEps
		yMinus, _ := m.Forward(ctx, x)

		x.Data[i] = orig

		numGrad := (sumTensor(yPlus) - sumTensor(yMinus)) / (2 * gradCheckEps)
		if diff := math.Abs(float64(numGrad - gradIn.Data[i])); diff > gradCheckTol {
			t.Errorf("input gradient mismatch at index %d: analytic=%v numeric=%v", i, gradIn.Data[i], numGrad)
		}
	}
	// Restore module state (input cache etc.) to the unperturbed forward pass.
	m.Forward(ctx, x)
}

// checkParamGradient verifies backward's accumulated gradient on p against
// central finite differences of sum(forward()), perturbing one element of
// p.Value at a time. forward/backward must be closures over the module
// under test (e.g. `func() (*Tensor, error) { return layer.Forward(ctx, x) }`).
func checkParamGradient(t *testing.T, forward func() (*Tensor, error), backward func(*Tensor) (*Tensor, error), p *Param) {
	t.Helper()
	y, err := forward()
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	gradOut := NewTensor(y.Shape)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	if _, err := backward(gradOut); err != nil {
		t.Fatalf("backward: %v", err)
	}
	analytic := append([]float32(nil), p.Grad.Data...)

	for i := range p.Value.Data {
		orig := p.Value.Data[i]

		p.Value.Data[i] = orig + gradCheckEps
		yPlus, _ := forward()

		p.Value.Data[i] = orig - gradCheckEps
		yMinus, _ := forward()

		p.Value.Data[i] = orig

		numGrad := (sumTensor(yPlus) - sumTensor(yMinus)) / (2 * gradCheckEps)
		if diff := math.Abs(float64(numGrad - analytic[i])); diff > gradCheckTol {
			t.Errorf("param gradient mismatch at index %d: analytic=%v numeric=%v", i, analytic[i], numGrad)
		}
	}
	// Restore the module's cached forward state.
	forward()
}
