// nn/embedding.go
package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// EmbeddingLayer maps integer indices to dense vectors via a learned
// lookup table.
type EmbeddingLayer struct {
	vocabSize, embedDim int
	W                    *Param
	init                 Initializer
	rng                  *rand.Rand
	input                *Tensor
}

// Embedding maps indices — [batch, seqLen] of float32 values holding
// non-negative integers — to dense vectors: output [batch, seqLen,
// embedDim]. Indices are not differentiable, so Backward always returns
// an all-zero input gradient; instead it scatter-accumulates into W.Grad,
// summing gradOut's row into every index it saw so a repeated index
// correctly picks up gradient contributions from every occurrence. init
// defaults to N(0, 1) when nil, matching common embedding-table init.
func Embedding(rng *rand.Rand, vocabSize, embedDim int, init Initializer) *EmbeddingLayer {
	if init == nil {
		init = NormalInit(0, 1)
	}
	return &EmbeddingLayer{
		vocabSize: vocabSize, embedDim: embedDim,
		W:    NewParam(init(rng, []int{vocabSize, embedDim})),
		init: init, rng: rng,
	}
}

func (e *EmbeddingLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 2 {
		return nil, fmt.Errorf("nn: Embedding expects input shape [batch, seq_len], got %v", inShape)
	}
	return []int{inShape[0], inShape[1], e.embedDim}, nil
}

func (e *EmbeddingLayer) index(v float32) int {
	return int(math.Round(float64(v)))
}

func (e *EmbeddingLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 2 {
		return nil, fmt.Errorf("nn: Embedding expects input shape [batch, seq_len], got %v", x.Shape)
	}
	e.input = x
	out := NewTensor([]int{x.Shape[0], x.Shape[1], e.embedDim})
	for i, v := range x.Data {
		idx := e.index(v)
		if idx < 0 || idx >= e.vocabSize {
			return nil, fmt.Errorf("nn: Embedding index %d out of range [0, %d)", idx, e.vocabSize)
		}
		copy(out.Data[i*e.embedDim:(i+1)*e.embedDim], e.W.Value.Data[idx*e.embedDim:(idx+1)*e.embedDim])
	}
	return out, nil
}

// Backward scatter-adds into W.Grad; the lookup pattern makes this a
// classic case of unpredictable overlapping writes (any index can repeat
// anywhere in the batch), so unlike the other layers in this package it
// is deliberately left un-parallelized rather than risking races or a
// partial-accumulator scheme for what is normally a small op.
func (e *EmbeddingLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	e.W.ZeroGrad()
	for i, v := range e.input.Data {
		idx := e.index(v)
		gRow := gradOut.Data[i*e.embedDim : (i+1)*e.embedDim]
		wgRow := e.W.Grad.Data[idx*e.embedDim : (idx+1)*e.embedDim]
		for j, g := range gRow {
			wgRow[j] += g
		}
	}
	return NewTensor(e.input.Shape), nil
}

func (e *EmbeddingLayer) Params() []*Param { return []*Param{e.W} }
