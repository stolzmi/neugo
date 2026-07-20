// nn/linear.go
package nn

import (
	"fmt"
	"math/rand"
	"runtime"
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
	if len(inShape) < 2 {
		return nil, fmt.Errorf("nn: Linear expects input shape [..., features] with at least 2 dims, got %v", inShape)
	}
	in := inShape[len(inShape)-1]
	if l.inFeatures == 0 {
		l.build(in)
	} else if l.inFeatures != in {
		return nil, fmt.Errorf("nn: Linear configured for %d input features, got %d", l.inFeatures, in)
	}
	outShape := append([]int(nil), inShape[:len(inShape)-1]...)
	return append(outShape, l.outFeatures), nil
}

// Forward treats every leading dimension as an effective batch ("rows" =
// their product) and the last dimension as features — the physical
// layout already has features as the contiguous, fastest-varying axis
// regardless of rank, so a [batch, seqLen, features] tensor's rows are
// exactly as row-major-contiguous as a plain [batch, features] tensor's,
// and the per-row loop below is unchanged from the old 2D-only version.
func (l *LinearLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if l.W == nil {
		return nil, fmt.Errorf("nn: Linear not built — call OutputShape or construct via Sequential first")
	}
	if len(x.Shape) < 2 || x.Shape[len(x.Shape)-1] != l.inFeatures {
		return nil, fmt.Errorf("nn: Linear expected input shape [..., %d], got %v", l.inFeatures, x.Shape)
	}
	l.input = x
	rows := len(x.Data) / l.inFeatures
	outShape := append([]int(nil), x.Shape[:len(x.Shape)-1]...)
	outShape = append(outShape, l.outFeatures)
	out := NewTensor(outShape)
	// Row-parallel: out rows are per-row disjoint; W/B are read-only here.
	parallelChunks(rows, func(_, rStart, rEnd int) {
		for r := rStart; r < rEnd; r++ {
			for o := 0; o < l.outFeatures; o++ {
				sum := l.B.Value.Data[o]
				for i := 0; i < l.inFeatures; i++ {
					sum += x.Data[r*l.inFeatures+i] * l.W.Value.Data[i*l.outFeatures+o]
				}
				out.Data[r*l.outFeatures+o] = sum
			}
		}
	})
	return out, nil
}

// Backward implements plain chain rule with no extra batch normalization
// (see design decision #6 in the original plan header) — W.Grad/B.Grad
// are raw sums over rows, exactly like gradIn.
func (l *LinearLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	rows := len(l.input.Data) / l.inFeatures
	gradIn := NewTensor(l.input.Shape)
	l.W.ZeroGrad()
	l.B.ZeroGrad()
	// Row-parallel: gradIn rows are per-row disjoint; shared W/B gradients
	// accumulate into per-chunk partials reduced in chunk order below.
	numChunks := len(chunkRanges(rows, runtime.GOMAXPROCS(0)))
	wPartials := make([][]float32, numChunks)
	bPartials := make([][]float32, numChunks)
	parallelChunks(rows, func(chunk, rStart, rEnd int) {
		wGrad := make([]float32, len(l.W.Grad.Data))
		bGrad := make([]float32, len(l.B.Grad.Data))
		wPartials[chunk], bPartials[chunk] = wGrad, bGrad
		for r := rStart; r < rEnd; r++ {
			for o := 0; o < l.outFeatures; o++ {
				g := gradOut.Data[r*l.outFeatures+o]
				bGrad[o] += g
				for i := 0; i < l.inFeatures; i++ {
					wGrad[i*l.outFeatures+o] += g * l.input.Data[r*l.inFeatures+i]
					gradIn.Data[r*l.inFeatures+i] += g * l.W.Value.Data[i*l.outFeatures+o]
				}
			}
		}
	})
	for chunk := 0; chunk < numChunks; chunk++ {
		for i, v := range wPartials[chunk] {
			l.W.Grad.Data[i] += v
		}
		for i, v := range bPartials[chunk] {
			l.B.Grad.Data[i] += v
		}
	}
	return gradIn, nil
}

func (l *LinearLayer) Params() []*Param {
	return []*Param{l.W, l.B}
}
