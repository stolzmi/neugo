// nn/linear.go
package nn

import (
	"fmt"
	"math/rand"
)

type LinearLayer struct {
	inFeatures, outFeatures int
	W, B                    *Param
	init                    Initializer
	rng                     *rand.Rand
	input                   *Tensor
}

// Linear creates a dense layer. inFeatures == 0 defers weight allocation
// until OutputShape is called with the real preceding shape (see design
// decision #2 in the plan header).
func Linear(rng *rand.Rand, inFeatures, outFeatures int, init Initializer) *LinearLayer {
	if init == nil {
		init = XavierInit()
	}
	l := &LinearLayer{inFeatures: inFeatures, outFeatures: outFeatures, init: init, rng: rng}
	if inFeatures > 0 {
		l.build(inFeatures)
	}
	return l
}

func (l *LinearLayer) build(inFeatures int) {
	l.inFeatures = inFeatures
	l.W = NewParam(l.init(l.rng, []int{inFeatures, l.outFeatures}))
	l.B = NewParam(NewTensor([]int{l.outFeatures}))
}

func (l *LinearLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 2 {
		return nil, fmt.Errorf("nn: Linear expects input shape [batch, features], got %v", inShape)
	}
	in := inShape[1]
	if l.inFeatures == 0 {
		l.build(in)
	} else if l.inFeatures != in {
		return nil, fmt.Errorf("nn: Linear configured for %d input features, got %d", l.inFeatures, in)
	}
	return []int{inShape[0], l.outFeatures}, nil
}

func (l *LinearLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if l.W == nil {
		return nil, fmt.Errorf("nn: Linear not built — call OutputShape or construct via Sequential first")
	}
	if len(x.Shape) != 2 || x.Shape[1] != l.inFeatures {
		return nil, fmt.Errorf("nn: Linear expected input shape [batch, %d], got %v", l.inFeatures, x.Shape)
	}
	l.input = x
	batch := x.Shape[0]
	out := NewTensor([]int{batch, l.outFeatures})
	for b := 0; b < batch; b++ {
		for o := 0; o < l.outFeatures; o++ {
			sum := l.B.Value.Data[o]
			for i := 0; i < l.inFeatures; i++ {
				sum += x.Data[b*l.inFeatures+i] * l.W.Value.Data[i*l.outFeatures+o]
			}
			out.Data[b*l.outFeatures+o] = sum
		}
	}
	return out, nil
}

// Backward implements plain chain rule with no extra batch normalization:
// gradOut already carries whatever batch-scaling the Loss applied (see
// Task 7), so W.Grad/B.Grad are raw sums over the batch, exactly like
// gradIn — introducing an additional /batch here would silently shrink
// every parameter gradient by a factor of batch and fail the Task 4
// gradient-check tests.
func (l *LinearLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch := l.input.Shape[0]
	gradIn := NewTensor([]int{batch, l.inFeatures})
	l.W.ZeroGrad()
	l.B.ZeroGrad()
	for b := 0; b < batch; b++ {
		for o := 0; o < l.outFeatures; o++ {
			g := gradOut.Data[b*l.outFeatures+o]
			l.B.Grad.Data[o] += g
			for i := 0; i < l.inFeatures; i++ {
				l.W.Grad.Data[i*l.outFeatures+o] += g * l.input.Data[b*l.inFeatures+i]
				gradIn.Data[b*l.inFeatures+i] += g * l.W.Value.Data[i*l.outFeatures+o]
			}
		}
	}
	return gradIn, nil
}

func (l *LinearLayer) Params() []*Param {
	return []*Param{l.W, l.B}
}
