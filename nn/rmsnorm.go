// nn/rmsnorm.go
package nn

import (
	"fmt"
	"math"
)

// RMSNormLayer normalizes each row of `channels` contiguous elements by its
// root-mean-square only — no mean subtraction, no bias — the normalization
// used in place of LayerNorm by LLaMA/Gemma-style Transformers (Zhang &
// Sennrich, 2019). Like LayerNorm/GroupNorm, it behaves identically in
// Train and Inference mode (no running stats).
type RMSNormLayer struct {
	channels int
	Gamma    *Param
	eps      float32

	input      *Tensor
	normalized []float32 // x * invRMS, cached for Gamma's gradient
	rowInvRMS  []float32
}

func RMSNorm(channels int) *RMSNormLayer {
	gamma := NewTensor([]int{channels})
	for i := range gamma.Data {
		gamma.Data[i] = 1
	}
	return &RMSNormLayer{channels: channels, Gamma: NewParam(gamma), eps: 1e-5}
}

func (r *RMSNormLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) == 0 || inShape[len(inShape)-1] != r.channels {
		return nil, fmt.Errorf("nn: RMSNorm configured for %d channels, got shape %v", r.channels, inShape)
	}
	return inShape, nil
}

func (r *RMSNormLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := r.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	channels := r.channels
	rows := len(x.Data) / channels
	out := NewTensor(x.Shape)
	r.input = x
	r.normalized = make([]float32, len(x.Data))
	r.rowInvRMS = make([]float32, rows)

	parallelChunks(rows, func(_, start, end int) {
		for row := start; row < end; row++ {
			base := row * channels
			data := x.Data[base : base+channels]
			var ss float32
			for _, v := range data {
				ss += v * v
			}
			meanSq := ss / float32(channels)
			invRMS := 1.0 / float32(math.Sqrt(float64(meanSq+r.eps)))
			r.rowInvRMS[row] = invRMS
			for c, v := range data {
				xhat := v * invRMS
				r.normalized[base+c] = xhat
				out.Data[base+c] = r.Gamma.Value.Data[c] * xhat
			}
		}
	})
	return out, nil
}

// Backward: for y_c = gamma_c*x_c*invRMS with invRMS = (mean(x^2)+eps)^-0.5,
//
//	dL/dx_j = gamma_j*invRMS*dy_j - (invRMS^3/channels)*x_j*S
//
// where S = sum_c(gamma_c*x_c*dy_c) over the row — the RMSNorm analogue of
// LayerNorm's dvar/dmean terms, but collapsed to one term since there's no
// mean to also propagate through.
func (r *RMSNormLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	channels := r.channels
	rows := len(r.input.Data) / channels
	nf := float32(channels)

	gradIn := NewTensor(r.input.Shape)
	r.Gamma.ZeroGrad()

	for base := 0; base < len(gradOut.Data); base += channels {
		row := gradOut.Data[base : base+channels]
		for c, dy := range row {
			r.Gamma.Grad.Data[c] += dy * r.normalized[base+c]
		}
	}

	parallelChunks(rows, func(_, start, end int) {
		for row := start; row < end; row++ {
			base := row * channels
			invRMS := r.rowInvRMS[row]
			var s float32
			for c := 0; c < channels; c++ {
				s += r.Gamma.Value.Data[c] * r.input.Data[base+c] * gradOut.Data[base+c]
			}
			coeff := invRMS * invRMS * invRMS / nf
			for c := 0; c < channels; c++ {
				gradIn.Data[base+c] = r.Gamma.Value.Data[c]*invRMS*gradOut.Data[base+c] - coeff*r.input.Data[base+c]*s
			}
		}
	})
	return gradIn, nil
}

func (r *RMSNormLayer) Params() []*Param { return []*Param{r.Gamma} }
