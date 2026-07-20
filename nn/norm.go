// nn/norm.go
package nn

import (
	"fmt"
	"math"
)

// BatchNormLayer normalizes over the last (channel) dimension. Because
// channels are always the fastest-varying axis in this codebase's tensor
// layout (design decision #5), one implementation serves both dense
// [batch, features] and conv [batch, h, w, channels] inputs: a flat index
// idx belongs to channel idx % channels, and its statistics are computed
// over the N = size/channels elements sharing that channel.
type BatchNormLayer struct {
	channels                int
	Gamma, Beta             *Param
	eps, momentum           float32
	runningMean, runningVar []float32

	input               *Tensor
	normalized          []float32
	batchMean, batchVar []float32
}

func BatchNorm(channels int) *BatchNormLayer {
	gamma := NewTensor([]int{channels})
	for i := range gamma.Data {
		gamma.Data[i] = 1
	}
	runningVar := make([]float32, channels)
	for i := range runningVar {
		runningVar[i] = 1
	}
	return &BatchNormLayer{
		channels:    channels,
		Gamma:       NewParam(gamma),
		Beta:        NewParam(NewTensor([]int{channels})),
		eps:         1e-5,
		momentum:    0.9,
		runningMean: make([]float32, channels),
		runningVar:  runningVar,
	}
}

func (bn *BatchNormLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) == 0 || inShape[len(inShape)-1] != bn.channels {
		return nil, fmt.Errorf("nn: BatchNorm configured for %d channels, got shape %v", bn.channels, inShape)
	}
	return inShape, nil
}

func (bn *BatchNormLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := bn.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	out := NewTensor(x.Shape)

	if ctx.Mode != Train {
		// Precompute per-channel scale/shift so the elementwise pass is a
		// single FMA; channels are the fastest axis, so a wrapping counter
		// replaces the per-element modulo.
		scale := make([]float32, bn.channels)
		shift := make([]float32, bn.channels)
		for c := range scale {
			s := bn.Gamma.Value.Data[c] / float32(math.Sqrt(float64(bn.runningVar[c]+bn.eps)))
			scale[c] = s
			shift[c] = bn.Beta.Value.Data[c] - s*bn.runningMean[c]
		}
		parallelChunks(len(x.Data), func(_, start, end int) {
			c := start % bn.channels
			for i := start; i < end; i++ {
				out.Data[i] = scale[c]*x.Data[i] + shift[c]
				c++
				if c == bn.channels {
					c = 0
				}
			}
		})
		return out, nil
	}

	n := len(x.Data) / bn.channels
	mean := make([]float32, bn.channels)
	for base := 0; base < len(x.Data); base += bn.channels {
		row := x.Data[base : base+bn.channels]
		for c, v := range row {
			mean[c] += v
		}
	}
	for c := range mean {
		mean[c] /= float32(n)
	}
	variance := make([]float32, bn.channels)
	for base := 0; base < len(x.Data); base += bn.channels {
		row := x.Data[base : base+bn.channels]
		for c, v := range row {
			d := v - mean[c]
			variance[c] += d * d
		}
	}
	for c := range variance {
		variance[c] /= float32(n)
	}

	bn.input = x
	bn.batchMean = mean
	bn.batchVar = variance
	bn.normalized = make([]float32, len(x.Data))
	invStd := make([]float32, bn.channels)
	for c := range invStd {
		invStd[c] = 1.0 / float32(math.Sqrt(float64(variance[c]+bn.eps)))
	}
	parallelChunks(len(x.Data), func(_, start, end int) {
		c := start % bn.channels
		for i := start; i < end; i++ {
			xhat := (x.Data[i] - mean[c]) * invStd[c]
			bn.normalized[i] = xhat
			out.Data[i] = bn.Gamma.Value.Data[c]*xhat + bn.Beta.Value.Data[c]
			c++
			if c == bn.channels {
				c = 0
			}
		}
	})
	for c := 0; c < bn.channels; c++ {
		bn.runningMean[c] = bn.momentum*bn.runningMean[c] + (1-bn.momentum)*mean[c]
		bn.runningVar[c] = bn.momentum*bn.runningVar[c] + (1-bn.momentum)*variance[c]
	}
	return out, nil
}

// Backward is the standard batchnorm backward derivation; see design
// decision #6 for why no additional batch-scaling is applied to
// Gamma.Grad/Beta.Grad.
func (bn *BatchNormLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	n := len(bn.input.Data) / bn.channels
	nf := float32(n)
	gradIn := NewTensor(bn.input.Shape)
	bn.Gamma.ZeroGrad()
	bn.Beta.ZeroGrad()

	dxhat := make([]float32, len(bn.input.Data))
	for base := 0; base < len(gradOut.Data); base += bn.channels {
		gRow := gradOut.Data[base : base+bn.channels]
		for c, dy := range gRow {
			bn.Gamma.Grad.Data[c] += dy * bn.normalized[base+c]
			bn.Beta.Grad.Data[c] += dy
			dxhat[base+c] = dy * bn.Gamma.Value.Data[c]
		}
	}

	invStd := make([]float32, bn.channels)
	for c := 0; c < bn.channels; c++ {
		invStd[c] = 1.0 / float32(math.Sqrt(float64(bn.batchVar[c]+bn.eps)))
	}

	dvar := make([]float32, bn.channels)
	dmean := make([]float32, bn.channels)
	for base := 0; base < len(dxhat); base += bn.channels {
		row := dxhat[base : base+bn.channels]
		for c, dxh := range row {
			centered := bn.input.Data[base+c] - bn.batchMean[c]
			dvar[c] += dxh * centered * -0.5 * invStd[c] * invStd[c] * invStd[c]
			dmean[c] += dxh * -invStd[c]
		}
	}

	parallelChunks(len(bn.input.Data), func(_, start, end int) {
		c := start % bn.channels
		for i := start; i < end; i++ {
			centered := bn.input.Data[i] - bn.batchMean[c]
			gradIn.Data[i] = dxhat[i]*invStd[c] + dvar[c]*2*centered/nf + dmean[c]/nf
			c++
			if c == bn.channels {
				c = 0
			}
		}
	})
	return gradIn, nil
}

func (bn *BatchNormLayer) Params() []*Param { return []*Param{bn.Gamma, bn.Beta} }
