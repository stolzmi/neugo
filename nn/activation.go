package nn

import (
	"fmt"
	"math"
)

type activationFn struct {
	apply func(float32) float32
	deriv func(float32) float32 // derivative w.r.t. the pre-activation input x
}

// ActivationModule applies an elementwise activation and its exact
// derivative. deriv is always evaluated at the cached pre-activation input,
// never the output — required for GELU, applied uniformly for consistency.
type ActivationModule struct {
	name  string
	alpha float32
	fn    activationFn
	input *Tensor
}

func newActivation(name string, alpha float32, fn activationFn) *ActivationModule {
	return &ActivationModule{name: name, alpha: alpha, fn: fn}
}

func ReLU() *ActivationModule {
	return newActivation("relu", 0, activationFn{
		apply: func(x float32) float32 {
			if x > 0 {
				return x
			}
			return 0
		},
		deriv: func(x float32) float32 {
			if x > 0 {
				return 1
			}
			return 0
		},
	})
}

func Sigmoid() *ActivationModule {
	sig := func(x float32) float32 { return float32(1 / (1 + math.Exp(float64(-x)))) }
	return newActivation("sigmoid", 0, activationFn{
		apply: sig,
		deriv: func(x float32) float32 { s := sig(x); return s * (1 - s) },
	})
}

func Tanh() *ActivationModule {
	return newActivation("tanh", 0, activationFn{
		apply: func(x float32) float32 { return float32(math.Tanh(float64(x))) },
		deriv: func(x float32) float32 {
			t := float32(math.Tanh(float64(x)))
			return 1 - t*t
		},
	})
}

func LeakyReLU(alpha float32) *ActivationModule {
	return newActivation("leaky_relu", alpha, activationFn{
		apply: func(x float32) float32 {
			if x > 0 {
				return x
			}
			return alpha * x
		},
		deriv: func(x float32) float32 {
			if x > 0 {
				return 1
			}
			return alpha
		},
	})
}

// GELU uses the exact formula 0.5*x*(1+erf(x/sqrt(2))), not the SiLU
// approximation the old Network/nnx.go mislabeled as GELU.
func GELU() *ActivationModule {
	return newActivation("gelu", 0, activationFn{
		apply: func(x float32) float32 {
			return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
		},
		deriv: func(x float32) float32 {
			cdf := 0.5 * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
			pdf := float32(math.Exp(-float64(x)*float64(x)/2)) / float32(math.Sqrt(2*math.Pi))
			return cdf + x*pdf
		},
	})
}

func (a *ActivationModule) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	a.input = x
	out := NewTensor(x.Shape)
	for i, v := range x.Data {
		out.Data[i] = a.fn.apply(v)
	}
	return out, nil
}

func (a *ActivationModule) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(a.input.Shape)
	for i, v := range a.input.Data {
		gradIn.Data[i] = gradOut.Data[i] * a.fn.deriv(v)
	}
	return gradIn, nil
}

func (a *ActivationModule) Params() []*Param { return nil }

func (a *ActivationModule) OutputShape(inShape []int) ([]int, error) { return inShape, nil }

func (a *ActivationModule) Name() string { return a.name }

func (a *ActivationModule) Alpha() float32 { return a.alpha }

// SoftmaxModule normalizes each row of a [batch, classes] tensor.
type SoftmaxModule struct {
	output *Tensor
}

func Softmax() *SoftmaxModule { return &SoftmaxModule{} }

func (s *SoftmaxModule) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 2 {
		return nil, fmt.Errorf("nn: Softmax expects input shape [batch, classes], got %v", x.Shape)
	}
	batch, classes := x.Shape[0], x.Shape[1]
	out := NewTensor(x.Shape)
	for b := 0; b < batch; b++ {
		maxV := x.Data[b*classes]
		for c := 1; c < classes; c++ {
			if v := x.Data[b*classes+c]; v > maxV {
				maxV = v
			}
		}
		var sum float32
		for c := 0; c < classes; c++ {
			e := float32(math.Exp(float64(x.Data[b*classes+c] - maxV)))
			out.Data[b*classes+c] = e
			sum += e
		}
		for c := 0; c < classes; c++ {
			out.Data[b*classes+c] /= sum
		}
	}
	s.output = out
	return out, nil
}

func (s *SoftmaxModule) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, classes := s.output.Shape[0], s.output.Shape[1]
	gradIn := NewTensor(s.output.Shape)
	for b := 0; b < batch; b++ {
		var dot float32
		for c := 0; c < classes; c++ {
			dot += gradOut.Data[b*classes+c] * s.output.Data[b*classes+c]
		}
		for c := 0; c < classes; c++ {
			y := s.output.Data[b*classes+c]
			gradIn.Data[b*classes+c] = y * (gradOut.Data[b*classes+c] - dot)
		}
	}
	return gradIn, nil
}

func (s *SoftmaxModule) Params() []*Param { return nil }

func (s *SoftmaxModule) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 2 {
		return nil, fmt.Errorf("nn: Softmax expects input shape [batch, classes], got %v", inShape)
	}
	return inShape, nil
}
