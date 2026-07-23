// nn/rope.go
package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// RotaryMultiHeadAttentionLayer is scaled-dot-product multi-head
// self-attention using Rotary Position Embedding (Su et al., 2021,
// "RoFormer") instead of an added positional embedding: Q and K are each
// rotated, per head and per pair of dimensions, by an angle proportional
// to sequence position before the dot product, so attention scores
// naturally depend on relative (not absolute) position — V is left
// untouched. It duplicates MultiHeadAttentionLayer's Q/K/V projection,
// masking, and weighted-sum machinery exactly, inserting only the
// rotate/un-rotate steps around the dot product; the two types are kept
// separate (rather than a shared base with a flag) since RoPE requires an
// even head dimension and carries its own cached state (the rotated Q/K
// and the position-dependent sin/cos tables), which would otherwise
// complicate MultiHeadAttentionLayer for every caller that doesn't want
// RoPE. Grouped/multi-query attention and KV-caching for autoregressive
// decoding are out of scope here.
type RotaryMultiHeadAttentionLayer struct {
	dModel, numHeads, dHead int
	causal                  bool
	scale                   float32
	ropeBase                float64
	wq, wk, wv, wo          *LinearLayer

	batch, seqLen  int
	q, k, v        *Tensor   // wq/wk/wv outputs, pre-rotation
	rq, rk         *Tensor   // rotated q, k — what the dot product actually uses
	cosTab, sinTab []float32 // [seqLen * dHead/2], position-major
	attnWeights    []float32 // [batch*numHeads*seqLen*seqLen]
}

func RotaryMultiHeadAttention(rng *rand.Rand, dModel, numHeads int, causal bool, init Initializer) *RotaryMultiHeadAttentionLayer {
	dHead := dModel / numHeads
	return &RotaryMultiHeadAttentionLayer{
		dModel: dModel, numHeads: numHeads, dHead: dHead, causal: causal,
		scale:    float32(1.0 / math.Sqrt(float64(dHead))),
		ropeBase: 10000,
		wq:       Linear(rng, dModel, dModel, init),
		wk:       Linear(rng, dModel, dModel, init),
		wv:       Linear(rng, dModel, dModel, init),
		wo:       Linear(rng, dModel, dModel, init),
	}
}

func (m *RotaryMultiHeadAttentionLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 3 || inShape[2] != m.dModel {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention expects input shape [batch, seqLen, %d], got %v", m.dModel, inShape)
	}
	if m.dModel%m.numHeads != 0 {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention dModel %d must be evenly divisible by numHeads %d", m.dModel, m.numHeads)
	}
	if m.dHead%2 != 0 {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention requires an even head dimension (dModel/numHeads), got %d", m.dHead)
	}
	return inShape, nil
}

// buildRopeTables precomputes cos/sin(pos*freq_i) for every position in
// [0, seqLen) and every pair index i in [0, dHead/2), freq_i =
// ropeBase^(-2i/dHead) — the standard RoPE frequency schedule (lower pair
// indices rotate faster, mirroring sinusoidal positional embeddings).
func (m *RotaryMultiHeadAttentionLayer) buildRopeTables(seqLen int) {
	half := m.dHead / 2
	m.cosTab = make([]float32, seqLen*half)
	m.sinTab = make([]float32, seqLen*half)
	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < half; i++ {
			freq := math.Pow(m.ropeBase, -2*float64(i)/float64(m.dHead))
			theta := float64(pos) * freq
			m.cosTab[pos*half+i] = float32(math.Cos(theta))
			m.sinTab[pos*half+i] = float32(math.Sin(theta))
		}
	}
}

// rotateRow applies RoPE's forward rotation to one head's dHead-length
// row: pairs (src[2i], src[2i+1]) rotate by angle theta_i = pos*freq_i via
// the standard 2D rotation matrix [[cos,-sin],[sin,cos]].
func (m *RotaryMultiHeadAttentionLayer) rotateRow(src, dst []float32, pos int) {
	half := m.dHead / 2
	cosRow := m.cosTab[pos*half : pos*half+half]
	sinRow := m.sinTab[pos*half : pos*half+half]
	for i := 0; i < half; i++ {
		x0, x1 := src[2*i], src[2*i+1]
		c, s := cosRow[i], sinRow[i]
		dst[2*i] = x0*c - x1*s
		dst[2*i+1] = x0*s + x1*c
	}
}

// unrotateRow applies the rotation's transpose (= inverse, since 2D
// rotation matrices are orthogonal) — used in Backward to turn a gradient
// w.r.t. the rotated Q/K back into a gradient w.r.t. the original,
// pre-rotation Q/K that wq/wk actually produced.
func (m *RotaryMultiHeadAttentionLayer) unrotateRow(src, dst []float32, pos int) {
	half := m.dHead / 2
	cosRow := m.cosTab[pos*half : pos*half+half]
	sinRow := m.sinTab[pos*half : pos*half+half]
	for i := 0; i < half; i++ {
		y0, y1 := src[2*i], src[2*i+1]
		c, s := cosRow[i], sinRow[i]
		dst[2*i] = y0*c + y1*s
		dst[2*i+1] = -y0*s + y1*c
	}
}

func (m *RotaryMultiHeadAttentionLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := m.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	batch, seqLen, dModel := x.Shape[0], x.Shape[1], m.dModel
	numHeads, dHead, scale := m.numHeads, m.dHead, m.scale

	q, err := m.wq.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention wq: %w", err)
	}
	k, err := m.wk.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention wk: %w", err)
	}
	v, err := m.wv.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention wv: %w", err)
	}
	m.batch, m.seqLen, m.q, m.k, m.v = batch, seqLen, q, k, v

	m.buildRopeTables(seqLen)
	rq := NewTensor(q.Shape)
	rk := NewTensor(k.Shape)
	// Batch*position-parallel: every (b,pos) writes a disjoint set of
	// dModel-wide slices (one per head), so this is safe without partials.
	parallelChunks(batch*seqLen, func(_, start, end int) {
		for bs := start; bs < end; bs++ {
			b, pos := bs/seqLen, bs%seqLen
			for h := 0; h < numHeads; h++ {
				base := (b*seqLen+pos)*dModel + h*dHead
				m.rotateRow(q.Data[base:base+dHead], rq.Data[base:base+dHead], pos)
				m.rotateRow(k.Data[base:base+dHead], rk.Data[base:base+dHead], pos)
			}
		}
	})
	m.rq, m.rk = rq, rk

	m.attnWeights = make([]float32, batch*numHeads*seqLen*seqLen)
	attnOut := NewTensor([]int{batch, seqLen, dModel})
	// Batch*head-parallel, exactly as MultiHeadAttentionLayer.Forward: the
	// dot product below reads rq/rk (rotated) instead of q/k, everything
	// else — masking, softmax, weighted sum over v (unrotated) — is
	// identical.
	parallelChunks(batch*numHeads, func(_, start, end int) {
		scores := make([]float32, seqLen*seqLen)
		for bh := start; bh < end; bh++ {
			b, h := bh/numHeads, bh%numHeads
			for i := 0; i < seqLen; i++ {
				qBase := (b*seqLen+i)*dModel + h*dHead
				qRow := rq.Data[qBase : qBase+dHead]
				limit := seqLen
				if m.causal {
					limit = i + 1
				}
				for j := 0; j < limit; j++ {
					kBase := (b*seqLen+j)*dModel + h*dHead
					kRow := rk.Data[kBase : kBase+dHead]
					var dot float32
					for d, qv := range qRow {
						dot += qv * kRow[d]
					}
					scores[i*seqLen+j] = dot * scale
				}
				for j := limit; j < seqLen; j++ {
					scores[i*seqLen+j] = float32(math.Inf(-1))
				}
			}
			for i := 0; i < seqLen; i++ {
				row := scores[i*seqLen : i*seqLen+seqLen]
				rowMax := row[0]
				for _, s := range row[1:] {
					if s > rowMax {
						rowMax = s
					}
				}
				var sumExp float32
				for j, s := range row {
					e := float32(math.Exp(float64(s - rowMax)))
					row[j] = e
					sumExp += e
				}
				for j := range row {
					row[j] /= sumExp
				}
			}
			copy(m.attnWeights[bh*seqLen*seqLen:(bh+1)*seqLen*seqLen], scores)
			for i := 0; i < seqLen; i++ {
				outBase := (b*seqLen+i)*dModel + h*dHead
				outRow := attnOut.Data[outBase : outBase+dHead]
				wRow := scores[i*seqLen : i*seqLen+seqLen]
				for j, wij := range wRow {
					if wij == 0 {
						continue
					}
					vBase := (b*seqLen+j)*dModel + h*dHead
					vRow := v.Data[vBase : vBase+dHead]
					for d, vv := range vRow {
						outRow[d] += wij * vv
					}
				}
			}
		}
	})

	output, err := m.wo.Forward(ctx, attnOut)
	if err != nil {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention wo: %w", err)
	}
	return output, nil
}

func (m *RotaryMultiHeadAttentionLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradAttnOut, err := m.wo.Backward(ctx, gradOut)
	if err != nil {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention wo backward: %w", err)
	}

	batch, seqLen, dModel := m.batch, m.seqLen, m.dModel
	numHeads, dHead, scale := m.numHeads, m.dHead, m.scale

	dRQ := NewTensor([]int{batch, seqLen, dModel})
	dRK := NewTensor([]int{batch, seqLen, dModel})
	dV := NewTensor([]int{batch, seqLen, dModel})

	parallelChunks(batch*numHeads, func(_, start, end int) {
		for bh := start; bh < end; bh++ {
			b, h := bh/numHeads, bh%numHeads
			weights := m.attnWeights[bh*seqLen*seqLen : (bh+1)*seqLen*seqLen]
			dWeights := make([]float32, seqLen*seqLen)

			for i := 0; i < seqLen; i++ {
				dOutBase := (b*seqLen+i)*dModel + h*dHead
				dOutRow := gradAttnOut.Data[dOutBase : dOutBase+dHead]
				for j := 0; j < seqLen; j++ {
					wij := weights[i*seqLen+j]
					if wij == 0 {
						continue
					}
					vBase := (b*seqLen+j)*dModel + h*dHead
					vRow := m.v.Data[vBase : vBase+dHead]
					var dot float32
					for d, dv := range dOutRow {
						dot += dv * vRow[d]
					}
					dWeights[i*seqLen+j] = dot

					dVRow := dV.Data[vBase : vBase+dHead]
					for d, dv := range dOutRow {
						dVRow[d] += wij * dv
					}
				}
			}

			dScores := make([]float32, seqLen*seqLen)
			for i := 0; i < seqLen; i++ {
				row := weights[i*seqLen : i*seqLen+seqLen]
				dRow := dWeights[i*seqLen : i*seqLen+seqLen]
				var rowDot float32
				for j, wij := range row {
					rowDot += dRow[j] * wij
				}
				for j, wij := range row {
					dScores[i*seqLen+j] = wij * (dRow[j] - rowDot)
				}
			}

			for i := 0; i < seqLen; i++ {
				qBase := (b*seqLen+i)*dModel + h*dHead
				qRow := m.rq.Data[qBase : qBase+dHead]
				dQRow := dRQ.Data[qBase : qBase+dHead]
				for j := 0; j < seqLen; j++ {
					ds := dScores[i*seqLen+j]
					if ds == 0 {
						continue
					}
					kBase := (b*seqLen+j)*dModel + h*dHead
					kRow := m.rk.Data[kBase : kBase+dHead]
					dKRow := dRK.Data[kBase : kBase+dHead]
					for d := 0; d < dHead; d++ {
						dQRow[d] += ds * scale * kRow[d]
						dKRow[d] += ds * scale * qRow[d]
					}
				}
			}
		}
	})

	// Un-rotate dRQ/dRK into gradients w.r.t. the original (pre-RoPE) q/k
	// that wq/wk produced — see unrotateRow's doc comment.
	dQ := NewTensor([]int{batch, seqLen, dModel})
	dK := NewTensor([]int{batch, seqLen, dModel})
	parallelChunks(batch*seqLen, func(_, start, end int) {
		for bs := start; bs < end; bs++ {
			b, pos := bs/seqLen, bs%seqLen
			for h := 0; h < numHeads; h++ {
				base := (b*seqLen+pos)*dModel + h*dHead
				m.unrotateRow(dRQ.Data[base:base+dHead], dQ.Data[base:base+dHead], pos)
				m.unrotateRow(dRK.Data[base:base+dHead], dK.Data[base:base+dHead], pos)
			}
		}
	})

	gradFromQ, err := m.wq.Backward(ctx, dQ)
	if err != nil {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention wq backward: %w", err)
	}
	gradFromK, err := m.wk.Backward(ctx, dK)
	if err != nil {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention wk backward: %w", err)
	}
	gradFromV, err := m.wv.Backward(ctx, dV)
	if err != nil {
		return nil, fmt.Errorf("nn: RotaryMultiHeadAttention wv backward: %w", err)
	}

	gradIn := NewTensor(gradFromQ.Shape)
	for i := range gradIn.Data {
		gradIn.Data[i] = gradFromQ.Data[i] + gradFromK.Data[i] + gradFromV.Data[i]
	}
	return gradIn, nil
}

func (m *RotaryMultiHeadAttentionLayer) Params() []*Param {
	var params []*Param
	params = append(params, m.wq.Params()...)
	params = append(params, m.wk.Params()...)
	params = append(params, m.wv.Params()...)
	params = append(params, m.wo.Params()...)
	return params
}
