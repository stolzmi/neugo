// nn/groupnorm.go
package nn

import (
	"fmt"
	"math"
)

// GroupNormLayer splits the channel axis (always fastest-varying, per this
// codebase's tensor layout) into Groups groups and normalizes each
// (sample, group) independently over that group's channels together with
// every spatial position — unlike BatchNorm, which reduces across the
// batch, GroupNorm's statistics never depend on other samples, so it
// behaves identically in Train and Inference mode and needs no running
// stats.
type GroupNormLayer struct {
	groups, channels int
	Gamma, Beta      *Param
	eps              float32

	input      *Tensor
	normalized []float32
	groupMean  []float32 // per (batch, group)
	groupVar   []float32
}

func GroupNorm(groups, channels int) *GroupNormLayer {
	gamma := NewTensor([]int{channels})
	for i := range gamma.Data {
		gamma.Data[i] = 1
	}
	return &GroupNormLayer{
		groups: groups, channels: channels,
		Gamma: NewParam(gamma), Beta: NewParam(NewTensor([]int{channels})),
		eps: 1e-5,
	}
}

func (g *GroupNormLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) == 0 || inShape[len(inShape)-1] != g.channels {
		return nil, fmt.Errorf("nn: GroupNorm configured for %d channels, got shape %v", g.channels, inShape)
	}
	if g.groups <= 0 || g.channels%g.groups != 0 {
		return nil, fmt.Errorf("nn: GroupNorm channels %d must be evenly divisible by groups %d", g.channels, g.groups)
	}
	return inShape, nil
}

// dims returns (spatialSize, channelsPerGroup, groupSize) for x.Shape,
// where spatialSize is the product of every non-batch, non-channel
// dimension (1 for dense [batch, features] input).
func (g *GroupNormLayer) dims(shape []int) (spatialSize, channelsPerGroup, groupSize int) {
	batch := shape[0]
	total := shapeSize(shape)
	spatialSize = total / (batch * g.channels)
	channelsPerGroup = g.channels / g.groups
	groupSize = spatialSize * channelsPerGroup
	return
}

func (g *GroupNormLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := g.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	batch := x.Shape[0]
	spatialSize, channelsPerGroup, groupSize := g.dims(x.Shape)
	channels := g.channels

	out := NewTensor(x.Shape)
	g.input = x
	g.normalized = make([]float32, len(x.Data))
	g.groupMean = make([]float32, batch*g.groups)
	g.groupVar = make([]float32, batch*g.groups)

	parallelChunks(batch*g.groups, func(_, start, end int) {
		for bg := start; bg < end; bg++ {
			b, grp := bg/g.groups, bg%g.groups
			cStart := grp * channelsPerGroup

			var mean float32
			for s := 0; s < spatialSize; s++ {
				rowBase := (b*spatialSize+s)*channels + cStart
				for _, v := range x.Data[rowBase : rowBase+channelsPerGroup] {
					mean += v
				}
			}
			mean /= float32(groupSize)

			var variance float32
			for s := 0; s < spatialSize; s++ {
				rowBase := (b*spatialSize+s)*channels + cStart
				for _, v := range x.Data[rowBase : rowBase+channelsPerGroup] {
					d := v - mean
					variance += d * d
				}
			}
			variance /= float32(groupSize)
			g.groupMean[bg] = mean
			g.groupVar[bg] = variance
			invStd := 1.0 / float32(math.Sqrt(float64(variance+g.eps)))

			for s := 0; s < spatialSize; s++ {
				rowBase := (b*spatialSize+s)*channels + cStart
				for c := 0; c < channelsPerGroup; c++ {
					idx := rowBase + c
					xhat := (x.Data[idx] - mean) * invStd
					g.normalized[idx] = xhat
					out.Data[idx] = g.Gamma.Value.Data[cStart+c]*xhat + g.Beta.Value.Data[cStart+c]
				}
			}
		}
	})
	return out, nil
}

// Backward is the standard normalization-layer backward derivation (same
// shape as BatchNorm's, see nn/norm.go), but reduced over each group's
// groupSize elements instead of the whole batch per channel.
func (g *GroupNormLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch := g.input.Shape[0]
	spatialSize, channelsPerGroup, groupSize := g.dims(g.input.Shape)
	channels := g.channels
	nf := float32(groupSize)

	gradIn := NewTensor(g.input.Shape)
	g.Gamma.ZeroGrad()
	g.Beta.ZeroGrad()

	dxhat := make([]float32, len(g.input.Data))
	for base := 0; base < len(gradOut.Data); base += channels {
		gRow := gradOut.Data[base : base+channels]
		for c, dy := range gRow {
			g.Gamma.Grad.Data[c] += dy * g.normalized[base+c]
			g.Beta.Grad.Data[c] += dy
			dxhat[base+c] = dy * g.Gamma.Value.Data[c]
		}
	}

	parallelChunks(batch*g.groups, func(_, start, end int) {
		for bg := start; bg < end; bg++ {
			b, grp := bg/g.groups, bg%g.groups
			cStart := grp * channelsPerGroup
			mean, variance := g.groupMean[bg], g.groupVar[bg]
			invStd := 1.0 / float32(math.Sqrt(float64(variance+g.eps)))

			var dvar, dmean float32
			for s := 0; s < spatialSize; s++ {
				rowBase := (b*spatialSize+s)*channels + cStart
				for c := 0; c < channelsPerGroup; c++ {
					idx := rowBase + c
					centered := g.input.Data[idx] - mean
					dxh := dxhat[idx]
					dvar += dxh * centered * -0.5 * invStd * invStd * invStd
					dmean += dxh * -invStd
				}
			}
			for s := 0; s < spatialSize; s++ {
				rowBase := (b*spatialSize+s)*channels + cStart
				for c := 0; c < channelsPerGroup; c++ {
					idx := rowBase + c
					centered := g.input.Data[idx] - mean
					gradIn.Data[idx] = dxhat[idx]*invStd + dvar*2*centered/nf + dmean/nf
				}
			}
		}
	})
	return gradIn, nil
}

func (g *GroupNormLayer) Params() []*Param { return []*Param{g.Gamma, g.Beta} }
