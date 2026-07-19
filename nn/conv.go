// nn/conv.go
package nn

import (
	"fmt"
	"math/rand"
)

type Conv2DLayer struct {
	inChannels, outChannels, kernelSize, padding int
	W, B                                         *Param
	init                                         Initializer
	rng                                          *rand.Rand
	input                                        *Tensor
}

// Conv2D is stride-1, zero-padding ("valid").
func Conv2D(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer {
	return newConv2D(rng, inChannels, outChannels, kernelSize, 0, init)
}

// Conv2DSame is stride-1 with padding (kernelSize-1)/2 ("same"); kernelSize
// must be odd.
func Conv2DSame(rng *rand.Rand, inChannels, outChannels, kernelSize int, init Initializer) *Conv2DLayer {
	return newConv2D(rng, inChannels, outChannels, kernelSize, (kernelSize-1)/2, init)
}

func newConv2D(rng *rand.Rand, inChannels, outChannels, kernelSize, padding int, init Initializer) *Conv2DLayer {
	if init == nil {
		init = HeInit()
	}
	c := &Conv2DLayer{inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize, padding: padding, init: init, rng: rng}
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
	h, w := inShape[1], inShape[2]
	outH := (h+2*c.padding-c.kernelSize)/1 + 1
	outW := (w+2*c.padding-c.kernelSize)/1 + 1
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
	k, pad := c.kernelSize, c.padding
	outH := (h+2*pad-k)/1 + 1
	outW := (w+2*pad-k)/1 + 1
	out := NewTensor([]int{batch, outH, outW, c.outChannels})

	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for oc := 0; oc < c.outChannels; oc++ {
					sum := c.B.Value.Data[oc]
					for ic := 0; ic < c.inChannels; ic++ {
						for kh := 0; kh < k; kh++ {
							ih := oh - pad + kh
							if ih < 0 || ih >= h {
								continue
							}
							for kw := 0; kw < k; kw++ {
								iw := ow - pad + kw
								if iw < 0 || iw >= w {
									continue
								}
								xVal := x.Data[((b*h+ih)*w+iw)*c.inChannels+ic]
								wVal := c.W.Value.Data[((oc*c.inChannels+ic)*k+kh)*k+kw]
								sum += xVal * wVal
							}
						}
					}
					out.Data[((b*outH+oh)*outW+ow)*c.outChannels+oc] = sum
				}
			}
		}
	}
	return out, nil
}

// Backward implements plain chain rule with no extra batch normalization:
// gradOut already carries whatever batch-scaling the Loss applied, so
// W.Grad/B.Grad are raw sums over the batch, exactly like gradIn (see
// design decision #6).
func (c *Conv2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, h, w := c.input.Shape[0], c.input.Shape[1], c.input.Shape[2]
	k, pad := c.kernelSize, c.padding
	outH, outW := gradOut.Shape[1], gradOut.Shape[2]

	// No batch-scaling on W.Grad/B.Grad here — see design decision #6.
	gradIn := NewTensor(c.input.Shape)
	c.W.ZeroGrad()
	c.B.ZeroGrad()

	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for oc := 0; oc < c.outChannels; oc++ {
					g := gradOut.Data[((b*outH+oh)*outW+ow)*c.outChannels+oc]
					c.B.Grad.Data[oc] += g
					for ic := 0; ic < c.inChannels; ic++ {
						for kh := 0; kh < k; kh++ {
							ih := oh - pad + kh
							if ih < 0 || ih >= h {
								continue
							}
							for kw := 0; kw < k; kw++ {
								iw := ow - pad + kw
								if iw < 0 || iw >= w {
									continue
								}
								xVal := c.input.Data[((b*h+ih)*w+iw)*c.inChannels+ic]
								wVal := c.W.Value.Data[((oc*c.inChannels+ic)*k+kh)*k+kw]
								c.W.Grad.Data[((oc*c.inChannels+ic)*k+kh)*k+kw] += g * xVal
								gradIn.Data[((b*h+ih)*w+iw)*c.inChannels+ic] += g * wVal
							}
						}
					}
				}
			}
		}
	}
	return gradIn, nil
}

func (c *Conv2DLayer) Params() []*Param { return []*Param{c.W, c.B} }
