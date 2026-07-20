// nn/attention.go
package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// MultiHeadAttentionLayer is standard scaled-dot-product multi-head
// self-attention. It implements Module by internally composing four
// Linear projections (Wq/Wk/Wv/Wo) around the attention core, so it
// composes into Sequential/Residual like every other layer in this
// package. Padding masks are not supported — only the fixed
// causal/non-causal mask — that scope boundary is intentional (see the
// design doc), not a gap.
type MultiHeadAttentionLayer struct {
	dModel, numHeads, dHead int
	causal                  bool
	scale                   float32
	wq, wk, wv, wo          *LinearLayer

	batch, seqLen int
	q, k, v       *Tensor
	attnWeights   []float32 // [batch*numHeads*seqLen*seqLen], row-major per (b,h)
}

func MultiHeadAttention(rng *rand.Rand, dModel, numHeads int, causal bool, init Initializer) *MultiHeadAttentionLayer {
	dHead := dModel / numHeads
	return &MultiHeadAttentionLayer{
		dModel: dModel, numHeads: numHeads, dHead: dHead, causal: causal,
		scale: float32(1.0 / math.Sqrt(float64(dHead))),
		wq:    Linear(rng, dModel, dModel, init),
		wk:    Linear(rng, dModel, dModel, init),
		wv:    Linear(rng, dModel, dModel, init),
		wo:    Linear(rng, dModel, dModel, init),
	}
}

func (m *MultiHeadAttentionLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 3 || inShape[2] != m.dModel {
		return nil, fmt.Errorf("nn: MultiHeadAttention expects input shape [batch, seqLen, %d], got %v", m.dModel, inShape)
	}
	if m.dModel%m.numHeads != 0 {
		return nil, fmt.Errorf("nn: MultiHeadAttention dModel %d must be evenly divisible by numHeads %d", m.dModel, m.numHeads)
	}
	return inShape, nil
}

func (m *MultiHeadAttentionLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := m.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	batch, seqLen, dModel := x.Shape[0], x.Shape[1], m.dModel
	numHeads, dHead, scale := m.numHeads, m.dHead, m.scale

	q, err := m.wq.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("nn: MultiHeadAttention wq: %w", err)
	}
	k, err := m.wk.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("nn: MultiHeadAttention wk: %w", err)
	}
	v, err := m.wv.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("nn: MultiHeadAttention wv: %w", err)
	}
	m.batch, m.seqLen, m.q, m.k, m.v = batch, seqLen, q, k, v
	m.attnWeights = make([]float32, batch*numHeads*seqLen*seqLen)

	attnOut := NewTensor([]int{batch, seqLen, dModel})
	// Batch*head-parallel: head h always writes the dModel-column range
	// [h*dHead:(h+1)*dHead] and batch b always writes rows for that b, so
	// distinct (b,h) pairs never write overlapping regions of attnOut,
	// q, k, or v — safe without partial accumulators.
	parallelChunks(batch*numHeads, func(_, start, end int) {
		scores := make([]float32, seqLen*seqLen)
		for bh := start; bh < end; bh++ {
			b, h := bh/numHeads, bh%numHeads
			for i := 0; i < seqLen; i++ {
				qBase := (b*seqLen+i)*dModel + h*dHead
				qRow := q.Data[qBase : qBase+dHead]
				limit := seqLen
				if m.causal {
					limit = i + 1
				}
				for j := 0; j < limit; j++ {
					kBase := (b*seqLen+j)*dModel + h*dHead
					kRow := k.Data[kBase : kBase+dHead]
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
		return nil, fmt.Errorf("nn: MultiHeadAttention wo: %w", err)
	}
	return output, nil
}

// Backward mirrors Forward in reverse: wo backward, then per-(b,h) the
// weighted-sum backward (dV, dWeights), softmax backward
// (dScores = weights * (dWeights - rowSum(dWeights*weights))), the
// scaled-dot-product backward (dQ, dK), then wq/wk/wv backward. x was
// used three times (once each for q, k, v), so gradIn sums all three
// branches' contributions — standard multi-use chain rule.
func (m *MultiHeadAttentionLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradAttnOut, err := m.wo.Backward(ctx, gradOut)
	if err != nil {
		return nil, fmt.Errorf("nn: MultiHeadAttention wo backward: %w", err)
	}

	batch, seqLen, dModel := m.batch, m.seqLen, m.dModel
	numHeads, dHead, scale := m.numHeads, m.dHead, m.scale

	dQ := NewTensor([]int{batch, seqLen, dModel})
	dK := NewTensor([]int{batch, seqLen, dModel})
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
				qRow := m.q.Data[qBase : qBase+dHead]
				dQRow := dQ.Data[qBase : qBase+dHead]
				for j := 0; j < seqLen; j++ {
					ds := dScores[i*seqLen+j]
					if ds == 0 {
						continue
					}
					kBase := (b*seqLen+j)*dModel + h*dHead
					kRow := m.k.Data[kBase : kBase+dHead]
					dKRow := dK.Data[kBase : kBase+dHead]
					for d := 0; d < dHead; d++ {
						dQRow[d] += ds * scale * kRow[d]
						dKRow[d] += ds * scale * qRow[d]
					}
				}
			}
		}
	})

	gradFromQ, err := m.wq.Backward(ctx, dQ)
	if err != nil {
		return nil, fmt.Errorf("nn: MultiHeadAttention wq backward: %w", err)
	}
	gradFromK, err := m.wk.Backward(ctx, dK)
	if err != nil {
		return nil, fmt.Errorf("nn: MultiHeadAttention wk backward: %w", err)
	}
	gradFromV, err := m.wv.Backward(ctx, dV)
	if err != nil {
		return nil, fmt.Errorf("nn: MultiHeadAttention wv backward: %w", err)
	}

	gradIn := NewTensor(gradFromQ.Shape)
	for i := range gradIn.Data {
		gradIn.Data[i] = gradFromQ.Data[i] + gradFromK.Data[i] + gradFromV.Data[i]
	}
	return gradIn, nil
}

func (m *MultiHeadAttentionLayer) Params() []*Param {
	var params []*Param
	params = append(params, m.wq.Params()...)
	params = append(params, m.wk.Params()...)
	params = append(params, m.wv.Params()...)
	params = append(params, m.wo.Params()...)
	return params
}

// CrossAttentionLayer attends from a query sequence to a separate
// context sequence (e.g. a decoder attending to an encoder's output).
// It shares MultiHeadAttentionLayer's core math but deliberately does
// NOT implement Module — two inputs, two output gradients, neither of
// which fits Module's one-in-one-out signature. Call it directly from
// hand-written decoder Forward/Backward code instead of composing it via
// Sequential. No causal masking (not meaningful for cross-attention).
type CrossAttentionLayer struct {
	dModel, numHeads, dHead int
	scale                   float32
	wq, wk, wv, wo          *LinearLayer

	batch, qLen, ctxLen int
	q, k, v             *Tensor
	attnWeights         []float32 // [batch*numHeads*qLen*ctxLen]
}

func CrossAttention(rng *rand.Rand, dModel, numHeads int, init Initializer) *CrossAttentionLayer {
	dHead := dModel / numHeads
	return &CrossAttentionLayer{
		dModel: dModel, numHeads: numHeads, dHead: dHead,
		scale: float32(1.0 / math.Sqrt(float64(dHead))),
		wq:    Linear(rng, dModel, dModel, init),
		wk:    Linear(rng, dModel, dModel, init),
		wv:    Linear(rng, dModel, dModel, init),
		wo:    Linear(rng, dModel, dModel, init),
	}
}

func (c *CrossAttentionLayer) checkShapes(query, context *Tensor) error {
	if len(query.Shape) != 3 || query.Shape[2] != c.dModel {
		return fmt.Errorf("nn: CrossAttention expects query shape [batch, qLen, %d], got %v", c.dModel, query.Shape)
	}
	if len(context.Shape) != 3 || context.Shape[2] != c.dModel {
		return fmt.Errorf("nn: CrossAttention expects context shape [batch, ctxLen, %d], got %v", c.dModel, context.Shape)
	}
	if query.Shape[0] != context.Shape[0] {
		return fmt.Errorf("nn: CrossAttention query batch %d does not match context batch %d", query.Shape[0], context.Shape[0])
	}
	if c.dModel%c.numHeads != 0 {
		return fmt.Errorf("nn: CrossAttention dModel %d must be evenly divisible by numHeads %d", c.dModel, c.numHeads)
	}
	return nil
}

func (c *CrossAttentionLayer) Forward(ctx *Context, query, context *Tensor) (*Tensor, error) {
	if err := c.checkShapes(query, context); err != nil {
		return nil, err
	}
	batch, qLen, ctxLen, dModel := query.Shape[0], query.Shape[1], context.Shape[1], c.dModel
	numHeads, dHead, scale := c.numHeads, c.dHead, c.scale

	q, err := c.wq.Forward(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("nn: CrossAttention wq: %w", err)
	}
	k, err := c.wk.Forward(ctx, context)
	if err != nil {
		return nil, fmt.Errorf("nn: CrossAttention wk: %w", err)
	}
	v, err := c.wv.Forward(ctx, context)
	if err != nil {
		return nil, fmt.Errorf("nn: CrossAttention wv: %w", err)
	}
	c.batch, c.qLen, c.ctxLen, c.q, c.k, c.v = batch, qLen, ctxLen, q, k, v
	c.attnWeights = make([]float32, batch*numHeads*qLen*ctxLen)

	attnOut := NewTensor([]int{batch, qLen, dModel})
	parallelChunks(batch*numHeads, func(_, start, end int) {
		scores := make([]float32, qLen*ctxLen)
		for bh := start; bh < end; bh++ {
			b, h := bh/numHeads, bh%numHeads
			for i := 0; i < qLen; i++ {
				qBase := (b*qLen+i)*dModel + h*dHead
				qRow := q.Data[qBase : qBase+dHead]
				for j := 0; j < ctxLen; j++ {
					kBase := (b*ctxLen+j)*dModel + h*dHead
					kRow := k.Data[kBase : kBase+dHead]
					var dot float32
					for d, qv := range qRow {
						dot += qv * kRow[d]
					}
					scores[i*ctxLen+j] = dot * scale
				}
			}
			for i := 0; i < qLen; i++ {
				row := scores[i*ctxLen : i*ctxLen+ctxLen]
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
			copy(c.attnWeights[bh*qLen*ctxLen:(bh+1)*qLen*ctxLen], scores)
			for i := 0; i < qLen; i++ {
				outBase := (b*qLen+i)*dModel + h*dHead
				outRow := attnOut.Data[outBase : outBase+dHead]
				wRow := scores[i*ctxLen : i*ctxLen+ctxLen]
				for j, wij := range wRow {
					if wij == 0 {
						continue
					}
					vBase := (b*ctxLen+j)*dModel + h*dHead
					vRow := v.Data[vBase : vBase+dHead]
					for d, vv := range vRow {
						outRow[d] += wij * vv
					}
				}
			}
		}
	})

	output, err := c.wo.Forward(ctx, attnOut)
	if err != nil {
		return nil, fmt.Errorf("nn: CrossAttention wo: %w", err)
	}
	return output, nil
}

func (c *CrossAttentionLayer) Backward(ctx *Context, gradOut *Tensor) (gradQuery, gradContext *Tensor, err error) {
	gradAttnOut, err := c.wo.Backward(ctx, gradOut)
	if err != nil {
		return nil, nil, fmt.Errorf("nn: CrossAttention wo backward: %w", err)
	}

	batch, qLen, ctxLen, dModel := c.batch, c.qLen, c.ctxLen, c.dModel
	numHeads, dHead, scale := c.numHeads, c.dHead, c.scale

	dQ := NewTensor([]int{batch, qLen, dModel})
	dK := NewTensor([]int{batch, ctxLen, dModel})
	dV := NewTensor([]int{batch, ctxLen, dModel})

	parallelChunks(batch*numHeads, func(_, start, end int) {
		for bh := start; bh < end; bh++ {
			b, h := bh/numHeads, bh%numHeads
			weights := c.attnWeights[bh*qLen*ctxLen : (bh+1)*qLen*ctxLen]
			dWeights := make([]float32, qLen*ctxLen)

			for i := 0; i < qLen; i++ {
				dOutBase := (b*qLen+i)*dModel + h*dHead
				dOutRow := gradAttnOut.Data[dOutBase : dOutBase+dHead]
				for j := 0; j < ctxLen; j++ {
					wij := weights[i*ctxLen+j]
					if wij == 0 {
						continue
					}
					vBase := (b*ctxLen+j)*dModel + h*dHead
					vRow := c.v.Data[vBase : vBase+dHead]
					var dot float32
					for d, dv := range dOutRow {
						dot += dv * vRow[d]
					}
					dWeights[i*ctxLen+j] = dot

					dVRow := dV.Data[vBase : vBase+dHead]
					for d, dv := range dOutRow {
						dVRow[d] += wij * dv
					}
				}
			}

			dScores := make([]float32, qLen*ctxLen)
			for i := 0; i < qLen; i++ {
				row := weights[i*ctxLen : i*ctxLen+ctxLen]
				dRow := dWeights[i*ctxLen : i*ctxLen+ctxLen]
				var rowDot float32
				for j, wij := range row {
					rowDot += dRow[j] * wij
				}
				for j, wij := range row {
					dScores[i*ctxLen+j] = wij * (dRow[j] - rowDot)
				}
			}

			for i := 0; i < qLen; i++ {
				qBase := (b*qLen+i)*dModel + h*dHead
				qRow := c.q.Data[qBase : qBase+dHead]
				dQRow := dQ.Data[qBase : qBase+dHead]
				for j := 0; j < ctxLen; j++ {
					ds := dScores[i*ctxLen+j]
					if ds == 0 {
						continue
					}
					kBase := (b*ctxLen+j)*dModel + h*dHead
					kRow := c.k.Data[kBase : kBase+dHead]
					dKRow := dK.Data[kBase : kBase+dHead]
					for d := 0; d < dHead; d++ {
						dQRow[d] += ds * scale * kRow[d]
						dKRow[d] += ds * scale * qRow[d]
					}
				}
			}
		}
	})

	gradFromQ, err := c.wq.Backward(ctx, dQ)
	if err != nil {
		return nil, nil, fmt.Errorf("nn: CrossAttention wq backward: %w", err)
	}
	gradFromK, err := c.wk.Backward(ctx, dK)
	if err != nil {
		return nil, nil, fmt.Errorf("nn: CrossAttention wk backward: %w", err)
	}
	gradFromV, err := c.wv.Backward(ctx, dV)
	if err != nil {
		return nil, nil, fmt.Errorf("nn: CrossAttention wv backward: %w", err)
	}

	gradContextOut := NewTensor(gradFromK.Shape)
	for i := range gradContextOut.Data {
		gradContextOut.Data[i] = gradFromK.Data[i] + gradFromV.Data[i]
	}
	return gradFromQ, gradContextOut, nil
}

func (c *CrossAttentionLayer) Params() []*Param {
	var params []*Param
	params = append(params, c.wq.Params()...)
	params = append(params, c.wk.Params()...)
	params = append(params, c.wv.Params()...)
	params = append(params, c.wo.Params()...)
	return params
}
