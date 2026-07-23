// nn/conv.go
package nn

import (
	"fmt"
	"math/rand"
)

type Conv2DLayer struct {
	inChannels, outChannels, kernelSize, padding, stride int
	W, B                                                 *Param
	init                                                 Initializer
	rng                                                  *rand.Rand
	input                                                *Tensor
}

// Conv2D is stride-1, zero-padding ("valid").
func Conv2D(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer {
	return newConv2D(rng, inChannels, outChannels, kernelSize, 0, 1, init)
}

// Conv2DSame is stride-1 with padding (kernelSize-1)/2 ("same"); kernelSize
// must be odd.
func Conv2DSame(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer {
	return newConv2D(rng, inChannels, outChannels, kernelSize, (kernelSize-1)/2, 1, init)
}

// Conv2DStrided exposes explicit stride and padding control — e.g. a
// stride-2 1x1 or 3x3 conv for ResNet-style downsampling shortcuts, where
// neither Conv2D's fixed stride-1/no-padding nor Conv2DSame's fixed
// stride-1/same-padding apply.
func Conv2DStrided(rng *rand.Rand, inChannels, outChannels, kernelSize, stride, padding int, init Initializer) *Conv2DLayer {
	return newConv2D(rng, inChannels, outChannels, kernelSize, padding, stride, init)
}

func newConv2D(rng *rand.Rand, inChannels, outChannels, kernelSize, padding, stride int, init Initializer) *Conv2DLayer {
	if init == nil {
		init = HeInit()
	}
	c := &Conv2DLayer{inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize, padding: padding, stride: stride, init: init, rng: rng}
	c.W = NewParam(init(rng, []int{outChannels, inChannels, kernelSize, kernelSize}))
	c.B = NewParam(NewTensor([]int{outChannels}))
	return c
}

func (c *Conv2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 || inShape[3] != c.inChannels {
		return nil, fmt.Errorf("nn: Conv2D expects input shape [batch, h, w, %d], got %v", c.inChannels, inShape)
	}
	if c.padding < 0 {
		return nil, fmt.Errorf("nn: Conv2D padding must be non-negative, got %d", c.padding)
	}
	if c.stride <= 0 {
		return nil, fmt.Errorf("nn: Conv2D stride must be positive, got %d", c.stride)
	}
	h, w := inShape[1], inShape[2]
	outH := (h+2*c.padding-c.kernelSize)/c.stride + 1
	outW := (w+2*c.padding-c.kernelSize)/c.stride + 1
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("nn: Conv2D input %dx%d too small for kernel %d with padding %d", h, w, c.kernelSize, c.padding)
	}
	return []int{inShape[0], outH, outW, c.outChannels}, nil
}

func (c *Conv2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 4 || x.Shape[3] != c.inChannels {
		return nil, fmt.Errorf("nn: Conv2D expects input shape [batch, h, w, %d], got %v", c.inChannels, x.Shape)
	}
	c.input = x
	batch, h, w := x.Shape[0], x.Shape[1], x.Shape[2]
	k, pad, stride := c.kernelSize, c.padding, c.stride
	outH := (h+2*pad-k)/stride + 1
	outW := (w+2*pad-k)/stride + 1
	out := NewTensor([]int{batch, outH, outW, c.outChannels})
	inC, outC := c.inChannels, c.outChannels
	wT := c.transposedWeights()
	bias := c.B.Value.Data
	xData := x.Data

	// Batch-parallel: each worker owns a contiguous batch chunk, so all
	// out writes are disjoint; W/B are read-only here. The inner loops are
	// the fused im2col/axpy form: for every in-bounds kernel tap, both the
	// input pixel's channels and the transposed weight row are contiguous,
	// so the hot loop is a pure multiply-add over slices with no index
	// arithmetic or bounds branches.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					outBase := ((b*outH+oh)*outW + ow) * outC
					outRow := out.Data[outBase : outBase+outC]
					copy(outRow, bias)
					for kh := 0; kh < k; kh++ {
						ih := oh*stride - pad + kh
						if ih < 0 || ih >= h {
							continue
						}
						for kw := 0; kw < k; kw++ {
							iw := ow*stride - pad + kw
							if iw < 0 || iw >= w {
								continue
							}
							xBase := ((b*h+ih)*w + iw) * inC
							xSeg := xData[xBase : xBase+inC]
							wBase := (kh*k + kw) * inC * outC
							for ic, xv := range xSeg {
								wRow := wT[wBase+ic*outC : wBase+(ic+1)*outC]
								for n, wv := range wRow {
									outRow[n] += xv * wv
								}
							}
						}
					}
				}
			}
		}
	})
	return out, nil
}

// transposedWeights relays W from [oc][ic][kh][kw] into rows of
// [(kh*k+kw)*inC+ic][outC] so the conv hot loops read weights
// contiguously. W is at most a few tens of KB, so rebuilding per call is
// noise next to the conv itself.
func (c *Conv2DLayer) transposedWeights() []float32 {
	k, inC, outC := c.kernelSize, c.inChannels, c.outChannels
	wT := make([]float32, k*k*inC*outC)
	wData := c.W.Value.Data
	for oc := 0; oc < outC; oc++ {
		for ic := 0; ic < inC; ic++ {
			for kh := 0; kh < k; kh++ {
				for kw := 0; kw < k; kw++ {
					wT[((kh*k+kw)*inC+ic)*outC+oc] = wData[((oc*inC+ic)*k+kh)*k+kw]
				}
			}
		}
	}
	return wT
}

// Backward implements plain chain rule with no extra batch normalization:
// gradOut already carries whatever batch-scaling the Loss applied, so
// W.Grad/B.Grad are raw sums over the batch, exactly like gradIn (see
// design decision #6).
func (c *Conv2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, h, w := c.input.Shape[0], c.input.Shape[1], c.input.Shape[2]
	k, pad, stride := c.kernelSize, c.padding, c.stride
	outH, outW := gradOut.Shape[1], gradOut.Shape[2]

	// No batch-scaling on W.Grad/B.Grad here — see design decision #6.
	gradIn := NewTensor(c.input.Shape)
	c.W.ZeroGrad()
	c.B.ZeroGrad()

	inC, outC := c.inChannels, c.outChannels
	wT := c.transposedWeights()
	xData := c.input.Data

	// Batch-parallel: gradIn writes are per-b disjoint; the shared W/B
	// gradients go into per-chunk partial buffers reduced in chunk order
	// below, so results don't depend on goroutine scheduling. Same fused
	// im2col/axpy structure as Forward: per kernel tap, the gradOut row,
	// weight row, weight-gradient row, and input/gradIn channels are all
	// contiguous. Weight-gradient partials use the transposed layout and
	// are folded back to [oc][ic][kh][kw] in the reduce.
	numChunks := numParallelChunks(batch)
	wPartials := make([][]float32, numChunks)
	bPartials := make([][]float32, numChunks)
	parallelChunks(batch, func(chunk, bStart, bEnd int) {
		wGradT := make([]float32, len(c.W.Grad.Data))
		bGrad := make([]float32, outC)
		wPartials[chunk], bPartials[chunk] = wGradT, bGrad
		for b := bStart; b < bEnd; b++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					gBase := ((b*outH+oh)*outW + ow) * outC
					gRow := gradOut.Data[gBase : gBase+outC]
					for n, gv := range gRow {
						bGrad[n] += gv
					}
					for kh := 0; kh < k; kh++ {
						ih := oh*stride - pad + kh
						if ih < 0 || ih >= h {
							continue
						}
						for kw := 0; kw < k; kw++ {
							iw := ow*stride - pad + kw
							if iw < 0 || iw >= w {
								continue
							}
							xBase := ((b*h+ih)*w + iw) * inC
							xSeg := xData[xBase : xBase+inC]
							giSeg := gradIn.Data[xBase : xBase+inC]
							wBase := (kh*k + kw) * inC * outC
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
			}
		}
	})
	wGrad := c.W.Grad.Data
	bGradTotal := c.B.Grad.Data
	for chunk := 0; chunk < numChunks; chunk++ {
		wGradT := wPartials[chunk]
		for kh := 0; kh < k; kh++ {
			for kw := 0; kw < k; kw++ {
				for ic := 0; ic < inC; ic++ {
					tBase := ((kh*k+kw)*inC + ic) * outC
					for oc := 0; oc < outC; oc++ {
						wGrad[((oc*inC+ic)*k+kh)*k+kw] += wGradT[tBase+oc]
					}
				}
			}
		}
		for i, v := range bPartials[chunk] {
			bGradTotal[i] += v
		}
	}
	return gradIn, nil
}

func (c *Conv2DLayer) Params() []*Param { return []*Param{c.W, c.B} }
