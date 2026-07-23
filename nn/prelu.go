// nn/prelu.go
package nn

import (
	"fmt"
)

// PReLULayer applies a per-channel Parametric ReLU: f(x) = x for x > 0,
// Alpha[c]*x otherwise, where c indexes the tensor's last (fastest-varying)
// dimension — the same channel convention used by BatchNorm/GroupNorm/
// LayerNorm. Unlike LeakyReLU's fixed alpha, PReLU's alpha is learned.
type PReLULayer struct {
	channels int
	Alpha    *Param
	input    *Tensor
}

// PReLU creates a PReLU layer over channels, with Alpha initialized to
// 0.25 per channel (the default from the original PReLU paper).
func PReLU(channels int) *PReLULayer {
	alpha := NewTensor([]int{channels})
	for i := range alpha.Data {
		alpha.Data[i] = 0.25
	}
	return &PReLULayer{channels: channels, Alpha: NewParam(alpha)}
}

func (p *PReLULayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) == 0 || inShape[len(inShape)-1] != p.channels {
		return nil, fmt.Errorf("nn: PReLU configured for %d channels, got shape %v", p.channels, inShape)
	}
	return inShape, nil
}

func (p *PReLULayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := p.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	p.input = x
	channels := p.channels
	rows := len(x.Data) / channels
	out := NewTensor(x.Shape)
	parallelChunks(rows, func(_, start, end int) {
		for r := start; r < end; r++ {
			base := r * channels
			for c := 0; c < channels; c++ {
				v := x.Data[base+c]
				if v > 0 {
					out.Data[base+c] = v
				} else {
					out.Data[base+c] = p.Alpha.Value.Data[c] * v
				}
			}
		}
	})
	return out, nil
}

// Backward accumulates Alpha's gradient across every row via per-chunk
// partials reduced serially afterward, matching LinearLayer's pattern for
// a Param shared across the parallelized batch dimension.
func (p *PReLULayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	channels := p.channels
	rows := len(p.input.Data) / channels
	gradIn := NewTensor(p.input.Shape)
	p.Alpha.ZeroGrad()

	numChunks := numParallelChunks(rows)
	aPartials := make([][]float32, numChunks)
	parallelChunks(rows, func(chunk, start, end int) {
		aGrad := make([]float32, channels)
		aPartials[chunk] = aGrad
		for r := start; r < end; r++ {
			base := r * channels
			for c := 0; c < channels; c++ {
				x := p.input.Data[base+c]
				g := gradOut.Data[base+c]
				if x > 0 {
					gradIn.Data[base+c] = g
				} else {
					gradIn.Data[base+c] = g * p.Alpha.Value.Data[c]
					aGrad[c] += g * x
				}
			}
		}
	})
	for chunk := 0; chunk < numChunks; chunk++ {
		for c, v := range aPartials[chunk] {
			p.Alpha.Grad.Data[c] += v
		}
	}
	return gradIn, nil
}

func (p *PReLULayer) Params() []*Param { return []*Param{p.Alpha} }
