// nn/layernorm.go
package nn

import (
	"fmt"
	"math"
)

// LayerNormLayer normalizes each row of `channels` contiguous elements
// independently — every leading index (batch, and any spatial/sequence
// positions) stays unreduced. This is the standard Transformer/RNN
// LayerNorm definition (Ba, Kiros & Hinton, 2016): for dense [batch,
// features] input each sample is one row, so this coincides with
// "normalize each sample independently"; for [batch, seqLen, channels]
// input (e.g. attention output) each (sample, position) pair is
// normalized independently, over just its channels — unlike
// GroupNorm(1, channels), which would pool statistics across seqLen too.
// LayerNorm behaves identically in Train and Inference mode (no running
// stats), same as GroupNorm.
type LayerNormLayer struct {
	channels    int
	Gamma, Beta *Param
	eps         float32

	input      *Tensor
	normalized []float32
	rowMean    []float32
	rowVar     []float32
}

func LayerNorm(channels int) *LayerNormLayer {
	gamma := NewTensor([]int{channels})
	for i := range gamma.Data {
		gamma.Data[i] = 1
	}
	return &LayerNormLayer{
		channels: channels,
		Gamma:    NewParam(gamma), Beta: NewParam(NewTensor([]int{channels})),
		eps: 1e-5,
	}
}

func (l *LayerNormLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) == 0 || inShape[len(inShape)-1] != l.channels {
		return nil, fmt.Errorf("nn: LayerNorm configured for %d channels, got shape %v", l.channels, inShape)
	}
	return inShape, nil
}

func (l *LayerNormLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := l.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	channels := l.channels
	rows := len(x.Data) / channels
	out := NewTensor(x.Shape)
	l.input = x
	l.normalized = make([]float32, len(x.Data))
	l.rowMean = make([]float32, rows)
	l.rowVar = make([]float32, rows)

	parallelChunks(rows, func(_, start, end int) {
		for r := start; r < end; r++ {
			base := r * channels
			row := x.Data[base : base+channels]
			var mean float32
			for _, v := range row {
				mean += v
			}
			mean /= float32(channels)
			var variance float32
			for _, v := range row {
				d := v - mean
				variance += d * d
			}
			variance /= float32(channels)
			l.rowMean[r], l.rowVar[r] = mean, variance
			invStd := 1.0 / float32(math.Sqrt(float64(variance+l.eps)))
			for c, v := range row {
				xhat := (v - mean) * invStd
				l.normalized[base+c] = xhat
				out.Data[base+c] = l.Gamma.Value.Data[c]*xhat + l.Beta.Value.Data[c]
			}
		}
	})
	return out, nil
}

// Backward is the standard normalization-layer backward derivation (same
// shape as GroupNorm's), reduced over each row's `channels` elements —
// nf below is always the channel count, never the whole batch or
// sequence, which is exactly the fix this task makes.
func (l *LayerNormLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	channels := l.channels
	rows := len(l.input.Data) / channels
	nf := float32(channels)

	gradIn := NewTensor(l.input.Shape)
	l.Gamma.ZeroGrad()
	l.Beta.ZeroGrad()

	for base := 0; base < len(gradOut.Data); base += channels {
		row := gradOut.Data[base : base+channels]
		for c, dy := range row {
			l.Gamma.Grad.Data[c] += dy * l.normalized[base+c]
			l.Beta.Grad.Data[c] += dy
		}
	}

	parallelChunks(rows, func(_, start, end int) {
		for r := start; r < end; r++ {
			base := r * channels
			mean, variance := l.rowMean[r], l.rowVar[r]
			invStd := 1.0 / float32(math.Sqrt(float64(variance+l.eps)))

			var dvar, dmean float32
			for c := 0; c < channels; c++ {
				idx := base + c
				dxh := gradOut.Data[idx] * l.Gamma.Value.Data[c]
				centered := l.input.Data[idx] - mean
				dvar += dxh * centered * -0.5 * invStd * invStd * invStd
				dmean += dxh * -invStd
			}
			for c := 0; c < channels; c++ {
				idx := base + c
				dxh := gradOut.Data[idx] * l.Gamma.Value.Data[c]
				centered := l.input.Data[idx] - mean
				gradIn.Data[idx] = dxh*invStd + dvar*2*centered/nf + dmean/nf
			}
		}
	})
	return gradIn, nil
}

func (l *LayerNormLayer) Params() []*Param { return []*Param{l.Gamma, l.Beta} }
