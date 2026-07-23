// nn/conv1d.go
package nn

import (
	"fmt"
	"math/rand"
)

// Conv1DLayer is Conv2D's 1D sibling — input/output are [batch, length,
// channels] instead of [batch, h, w, channels]. Mirrors Conv2D's
// structure (including the fused axpy inner loop over contiguous
// transposed-weight rows and per-chunk weight-gradient partials) with one
// fewer spatial dimension.
type Conv1DLayer struct {
	inChannels, outChannels, kernelSize, padding, stride int
	W, B                                                 *Param
	init                                                 Initializer
	rng                                                  *rand.Rand
	input                                                *Tensor
}

// Conv1D is stride-1, zero-padding ("valid").
func Conv1D(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv1DLayer {
	return newConv1D(rng, inChannels, outChannels, kernelSize, 0, 1, init)
}

// Conv1DSame is stride-1 with padding (kernelSize-1)/2 ("same"); kernelSize
// must be odd.
func Conv1DSame(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv1DLayer {
	return newConv1D(rng, inChannels, outChannels, kernelSize, (kernelSize-1)/2, 1, init)
}

// Conv1DStrided exposes explicit stride and padding control.
func Conv1DStrided(rng *rand.Rand, inChannels, outChannels, kernelSize, stride, padding int, init Initializer) *Conv1DLayer {
	return newConv1D(rng, inChannels, outChannels, kernelSize, padding, stride, init)
}

func newConv1D(rng *rand.Rand, inChannels, outChannels, kernelSize, padding, stride int, init Initializer) *Conv1DLayer {
	if init == nil {
		init = HeInit()
	}
	c := &Conv1DLayer{inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize, padding: padding, stride: stride, init: init, rng: rng}
	c.W = NewParam(init(rng, []int{outChannels, inChannels, kernelSize}))
	c.B = NewParam(NewTensor([]int{outChannels}))
	return c
}

func (c *Conv1DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 3 || inShape[2] != c.inChannels {
		return nil, fmt.Errorf("nn: Conv1D expects input shape [batch, length, %d], got %v", c.inChannels, inShape)
	}
	if c.padding < 0 {
		return nil, fmt.Errorf("nn: Conv1D padding must be non-negative, got %d", c.padding)
	}
	if c.stride <= 0 {
		return nil, fmt.Errorf("nn: Conv1D stride must be positive, got %d", c.stride)
	}
	length := inShape[1]
	outLen := (length+2*c.padding-c.kernelSize)/c.stride + 1
	if outLen <= 0 {
		return nil, fmt.Errorf("nn: Conv1D input length %d too small for kernel %d with padding %d", length, c.kernelSize, c.padding)
	}
	return []int{inShape[0], outLen, c.outChannels}, nil
}

func (c *Conv1DLayer) transposedWeights() []float32 {
	k, inC, outC := c.kernelSize, c.inChannels, c.outChannels
	wT := make([]float32, k*inC*outC)
	wData := c.W.Value.Data
	for oc := 0; oc < outC; oc++ {
		for ic := 0; ic < inC; ic++ {
			for kk := 0; kk < k; kk++ {
				wT[(kk*inC+ic)*outC+oc] = wData[(oc*inC+ic)*k+kk]
			}
		}
	}
	return wT
}

func (c *Conv1DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 3 || x.Shape[2] != c.inChannels {
		return nil, fmt.Errorf("nn: Conv1D expects input shape [batch, length, %d], got %v", c.inChannels, x.Shape)
	}
	c.input = x
	batch, length := x.Shape[0], x.Shape[1]
	k, pad, stride := c.kernelSize, c.padding, c.stride
	outLen := (length+2*pad-k)/stride + 1
	out := NewTensor([]int{batch, outLen, c.outChannels})
	inC, outC := c.inChannels, c.outChannels
	wT := c.transposedWeights()
	bias := c.B.Value.Data
	xData := x.Data

	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for ol := 0; ol < outLen; ol++ {
				outBase := (b*outLen + ol) * outC
				outRow := out.Data[outBase : outBase+outC]
				copy(outRow, bias)
				for kk := 0; kk < k; kk++ {
					il := ol*stride - pad + kk
					if il < 0 || il >= length {
						continue
					}
					xBase := (b*length + il) * inC
					xSeg := xData[xBase : xBase+inC]
					wBase := kk * inC * outC
					for ic, xv := range xSeg {
						wRow := wT[wBase+ic*outC : wBase+(ic+1)*outC]
						for n, wv := range wRow {
							outRow[n] += xv * wv
						}
					}
				}
			}
		}
	})
	return out, nil
}

func (c *Conv1DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, length := c.input.Shape[0], c.input.Shape[1]
	k, pad, stride := c.kernelSize, c.padding, c.stride
	outLen := gradOut.Shape[1]

	gradIn := NewTensor(c.input.Shape)
	c.W.ZeroGrad()
	c.B.ZeroGrad()

	inC, outC := c.inChannels, c.outChannels
	wT := c.transposedWeights()
	xData := c.input.Data

	numChunks := numParallelChunks(batch)
	wPartials := make([][]float32, numChunks)
	bPartials := make([][]float32, numChunks)
	parallelChunks(batch, func(chunk, bStart, bEnd int) {
		wGradT := make([]float32, len(c.W.Grad.Data))
		bGrad := make([]float32, outC)
		wPartials[chunk], bPartials[chunk] = wGradT, bGrad
		for b := bStart; b < bEnd; b++ {
			for ol := 0; ol < outLen; ol++ {
				gBase := (b*outLen + ol) * outC
				gRow := gradOut.Data[gBase : gBase+outC]
				for n, gv := range gRow {
					bGrad[n] += gv
				}
				for kk := 0; kk < k; kk++ {
					il := ol*stride - pad + kk
					if il < 0 || il >= length {
						continue
					}
					xBase := (b*length + il) * inC
					xSeg := xData[xBase : xBase+inC]
					giSeg := gradIn.Data[xBase : xBase+inC]
					wBase := kk * inC * outC
					for ic, xv := range xSeg {
						wRow := wT[wBase+ic*outC : wBase+(ic+1)*outC]
						wgRow := wGradT[wBase+ic*outC : wBase+(ic+1)*outC]
						var dot float32
						for n, gv := range gRow {
							dot += gv * wRow[n]
							wgRow[n] += xv * gv
						}
						giSeg[ic] += dot
					}
				}
			}
		}
	})
	wGrad := c.W.Grad.Data
	bGradTotal := c.B.Grad.Data
	for chunk := 0; chunk < numChunks; chunk++ {
		wGradT := wPartials[chunk]
		for kk := 0; kk < k; kk++ {
			for ic := 0; ic < inC; ic++ {
				tBase := (kk*inC + ic) * outC
				for oc := 0; oc < outC; oc++ {
					wGrad[(oc*inC+ic)*k+kk] += wGradT[tBase+oc]
				}
			}
		}
		for i, v := range bPartials[chunk] {
			bGradTotal[i] += v
		}
	}
	return gradIn, nil
}

func (c *Conv1DLayer) Params() []*Param { return []*Param{c.W, c.B} }
