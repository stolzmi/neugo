// nn/convtranspose.go
package nn

import (
	"fmt"
	"math/rand"
)

// ConvTranspose2DLayer is a transposed ("deconvolution"/upsampling) 2D
// conv: its output is larger than its input (for stride > 1), the
// standard building block for decoders, segmentation heads, and
// generative upsampling paths. Weight shape is [inChannels, outChannels,
// kh, kw] — transposed relative to Conv2D's [outChannels, inChannels, kh,
// kw] — matching the common ConvTranspose weight-layout convention.
//
// Forward scatters each input position into a kernel-sized window of the
// output (accumulating on overlap); Backward is the mirror-image dual of
// Conv2D's forward/backward (a well-known identity: transposed-conv's
// input-gradient computation has exactly the gather structure of a
// regular conv forward pass, and vice versa for its weight gradient).
// Unlike Conv2D, this implementation does not use the transposed-weight
// axpy optimization — it is deliberately kept as straightforward nested
// loops, correctness-first, since upsampling layers are typically far
// fewer and smaller than the downsampling convs earlier in a network.
type ConvTranspose2DLayer struct {
	inChannels, outChannels, kernelSize, padding, stride int
	W, B                                                 *Param
	init                                                 Initializer
	rng                                                  *rand.Rand
	input                                                *Tensor
}

// ConvTranspose2D does not support output_padding (the PyTorch parameter
// that resolves ambiguous output sizes for stride > 1); its output size
// is exactly (in-1)*stride - 2*padding + kernelSize.
func ConvTranspose2D(rng *rand.Rand, inChannels, outChannels, kernelSize, stride, padding int, init Initializer) *ConvTranspose2DLayer {
	if init == nil {
		init = HeInit()
	}
	c := &ConvTranspose2DLayer{inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize, padding: padding, stride: stride, init: init, rng: rng}
	c.W = NewParam(init(rng, []int{inChannels, outChannels, kernelSize, kernelSize}))
	c.B = NewParam(NewTensor([]int{outChannels}))
	return c
}

func (c *ConvTranspose2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 || inShape[3] != c.inChannels {
		return nil, fmt.Errorf("nn: ConvTranspose2D expects input shape [batch, h, w, %d], got %v", c.inChannels, inShape)
	}
	if c.padding < 0 {
		return nil, fmt.Errorf("nn: ConvTranspose2D padding must be non-negative, got %d", c.padding)
	}
	if c.stride <= 0 {
		return nil, fmt.Errorf("nn: ConvTranspose2D stride must be positive, got %d", c.stride)
	}
	h, w := inShape[1], inShape[2]
	outH := (h-1)*c.stride - 2*c.padding + c.kernelSize
	outW := (w-1)*c.stride - 2*c.padding + c.kernelSize
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("nn: ConvTranspose2D input %dx%d produces non-positive output for kernel %d, stride %d, padding %d", h, w, c.kernelSize, c.stride, c.padding)
	}
	return []int{inShape[0], outH, outW, c.outChannels}, nil
}

func (c *ConvTranspose2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	outShape, err := c.OutputShape(x.Shape)
	if err != nil {
		return nil, err
	}
	c.input = x
	batch, h, w := x.Shape[0], x.Shape[1], x.Shape[2]
	outH, outW := outShape[1], outShape[2]
	k, pad, stride := c.kernelSize, c.padding, c.stride
	inC, outC := c.inChannels, c.outChannels
	out := NewTensor(outShape)
	wData := c.W.Value.Data
	xData := x.Data
	bias := c.B.Value.Data

	// Batch-parallel: every write below touches only out's b-th slice, and
	// workers own disjoint batch ranges, so scatter-accumulation within a
	// slice (from overlapping kernel windows) never races across workers.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for ih := 0; ih < h; ih++ {
				for iw := 0; iw < w; iw++ {
					xBase := ((b*h+ih)*w + iw) * inC
					xSeg := xData[xBase : xBase+inC]
					for kh := 0; kh < k; kh++ {
						oh := ih*stride - pad + kh
						if oh < 0 || oh >= outH {
							continue
						}
						for kw := 0; kw < k; kw++ {
							ow := iw*stride - pad + kw
							if ow < 0 || ow >= outW {
								continue
							}
							outBase := ((b*outH+oh)*outW + ow) * outC
							outRow := out.Data[outBase : outBase+outC]
							for ic, xv := range xSeg {
								wBase := (ic*outC)*k*k + kh*k + kw
								for oc := 0; oc < outC; oc++ {
									outRow[oc] += xv * wData[wBase+oc*k*k]
								}
							}
						}
					}
				}
			}
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					outBase := ((b*outH+oh)*outW + ow) * outC
					outRow := out.Data[outBase : outBase+outC]
					for oc, bv := range bias {
						outRow[oc] += bv
					}
				}
			}
		}
	})
	return out, nil
}

func (c *ConvTranspose2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, h, w := c.input.Shape[0], c.input.Shape[1], c.input.Shape[2]
	outH, outW := gradOut.Shape[1], gradOut.Shape[2]
	k, pad, stride := c.kernelSize, c.padding, c.stride
	inC, outC := c.inChannels, c.outChannels

	gradIn := NewTensor(c.input.Shape)
	c.W.ZeroGrad()
	c.B.ZeroGrad()

	xData := c.input.Data
	wData := c.W.Value.Data

	numChunks := numParallelChunks(batch)
	wPartials := make([][]float32, numChunks)
	bPartials := make([][]float32, numChunks)
	parallelChunks(batch, func(chunk, bStart, bEnd int) {
		wGrad := make([]float32, len(c.W.Grad.Data))
		bGrad := make([]float32, outC)
		wPartials[chunk], bPartials[chunk] = wGrad, bGrad
		for b := bStart; b < bEnd; b++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					gBase := ((b*outH+oh)*outW + ow) * outC
					for oc, gv := range gradOut.Data[gBase : gBase+outC] {
						bGrad[oc] += gv
					}
				}
			}
			for ih := 0; ih < h; ih++ {
				for iw := 0; iw < w; iw++ {
					xBase := ((b*h+ih)*w + iw) * inC
					xSeg := xData[xBase : xBase+inC]
					for kh := 0; kh < k; kh++ {
						oh := ih*stride - pad + kh
						if oh < 0 || oh >= outH {
							continue
						}
						for kw := 0; kw < k; kw++ {
							ow := iw*stride - pad + kw
							if ow < 0 || ow >= outW {
								continue
							}
							gBase := ((b*outH+oh)*outW + ow) * outC
							gRow := gradOut.Data[gBase : gBase+outC]
							for ic, xv := range xSeg {
								wBase := (ic*outC)*k*k + kh*k + kw
								var dot float32
								for oc, gv := range gRow {
									wIdx := wBase + oc*k*k
									dot += gv * wData[wIdx]
									wGrad[wIdx] += xv * gv
								}
								gradIn.Data[xBase+ic] += dot
							}
						}
					}
				}
			}
		}
	})
	wGradTotal := c.W.Grad.Data
	bGradTotal := c.B.Grad.Data
	for chunk := 0; chunk < numChunks; chunk++ {
		for i, v := range wPartials[chunk] {
			wGradTotal[i] += v
		}
		for i, v := range bPartials[chunk] {
			bGradTotal[i] += v
		}
	}
	return gradIn, nil
}

func (c *ConvTranspose2DLayer) Params() []*Param { return []*Param{c.W, c.B} }
