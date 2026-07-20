// nn/positional.go
package nn

import (
	"fmt"
	"math/rand"
)

// PositionalEmbeddingLayer adds a learned, per-position vector to each
// position of a [batch, seqLen, dModel] input — without it, attention is
// permutation-invariant and cannot distinguish sequence order.
type PositionalEmbeddingLayer struct {
	maxLen, dModel int
	embed          *EmbeddingLayer
	seqLen         int
}

// PositionalEmbedding builds position indices 0..seqLen-1 internally on
// every Forward call — callers never construct them. Forward errors if
// seqLen exceeds maxLen rather than truncating or indexing out of bounds.
func PositionalEmbedding(rng *rand.Rand, maxLen, dModel int, init Initializer) *PositionalEmbeddingLayer {
	return &PositionalEmbeddingLayer{
		maxLen: maxLen, dModel: dModel,
		embed: Embedding(rng, maxLen, dModel, init),
	}
}

func (p *PositionalEmbeddingLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 3 || inShape[2] != p.dModel {
		return nil, fmt.Errorf("nn: PositionalEmbedding expects input shape [batch, seqLen, %d], got %v", p.dModel, inShape)
	}
	if inShape[1] > p.maxLen {
		return nil, fmt.Errorf("nn: PositionalEmbedding: sequence length %d exceeds maxLen %d", inShape[1], p.maxLen)
	}
	return inShape, nil
}

func (p *PositionalEmbeddingLayer) positions(seqLen int) *Tensor {
	idx := NewTensor([]int{1, seqLen})
	for i := range idx.Data {
		idx.Data[i] = float32(i)
	}
	return idx
}

func (p *PositionalEmbeddingLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := p.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	batch, seqLen, dModel := x.Shape[0], x.Shape[1], p.dModel
	p.seqLen = seqLen

	posEmbed, err := p.embed.Forward(ctx, p.positions(seqLen)) // [1, seqLen, dModel]
	if err != nil {
		return nil, fmt.Errorf("nn: PositionalEmbedding: %w", err)
	}

	out := NewTensor(x.Shape)
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			base := b * seqLen * dModel
			for i, v := range x.Data[base : base+seqLen*dModel] {
				out.Data[base+i] = v + posEmbed.Data[i]
			}
		}
	})
	return out, nil
}

// Backward: out = x + posEmbed(positions), a broadcast-add over the
// batch axis (posEmbed has an implicit batch of 1). x's own branch of
// the addition passes gradOut straight through as gradIn; posEmbed's
// branch needs gradOut summed over the batch axis before it can be fed
// into EmbeddingLayer.Backward's existing scatter-add.
func (p *PositionalEmbeddingLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, seqLen, dModel := gradOut.Shape[0], p.seqLen, p.dModel
	posGrad := NewTensor([]int{1, seqLen, dModel})
	for b := 0; b < batch; b++ {
		base := b * seqLen * dModel
		for i, g := range gradOut.Data[base : base+seqLen*dModel] {
			posGrad.Data[i] += g
		}
	}
	if _, err := p.embed.Backward(ctx, posGrad); err != nil {
		return nil, fmt.Errorf("nn: PositionalEmbedding: %w", err)
	}
	gradIn := NewTensor(gradOut.Shape)
	copy(gradIn.Data, gradOut.Data)
	return gradIn, nil
}

func (p *PositionalEmbeddingLayer) Params() []*Param { return p.embed.Params() }
