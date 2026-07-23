// nn/rnn.go
package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// RNNLayer is a vanilla (Elman) recurrent layer: h_t = tanh(x_t@Wx +
// h_{t-1}@Wh + b). Input is [batch, seqLen, features]; output is [batch,
// seqLen, hidden] (every timestep's hidden state, not just the last) —
// composable with LastTimestep() to reduce to [batch, hidden] for
// sequence classification, or with another recurrent layer to stack.
// Unlike every other layer in this package, Backward performs full
// backpropagation-through-time (BPTT): it cannot parallelize across
// timesteps (each depends on the previous one's hidden state), only across
// the batch dimension within a timestep.
type RNNLayer struct {
	features, hidden int
	Wx, Wh, B        *Param
	rng              *rand.Rand

	input         *Tensor
	batch, seqLen int
	// hStates[0] is the zero initial state; hStates[t+1] is h_t (the state
	// produced after processing timestep t), each flattened [batch*hidden].
	hStates [][]float32
}

func RNN(rng *rand.Rand, features, hidden int, init Initializer) *RNNLayer {
	if init == nil {
		init = XavierInit()
	}
	return &RNNLayer{
		features: features, hidden: hidden, rng: rng,
		Wx: NewParam(init(rng, []int{features, hidden})),
		Wh: NewParam(init(rng, []int{hidden, hidden})),
		B:  NewParam(NewTensor([]int{hidden})),
	}
}

func (r *RNNLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 3 || inShape[2] != r.features {
		return nil, fmt.Errorf("nn: RNN expects input shape [batch, seqLen, %d], got %v", r.features, inShape)
	}
	return []int{inShape[0], inShape[1], r.hidden}, nil
}

func (r *RNNLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := r.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	batch, seqLen, features, hidden := x.Shape[0], x.Shape[1], r.features, r.hidden
	r.input = x
	r.batch, r.seqLen = batch, seqLen

	out := NewTensor([]int{batch, seqLen, hidden})
	r.hStates = make([][]float32, seqLen+1)
	r.hStates[0] = make([]float32, batch*hidden)

	for t := 0; t < seqLen; t++ {
		hPrev := r.hStates[t]
		hCur := make([]float32, batch*hidden)
		parallelChunks(batch, func(_, bStart, bEnd int) {
			for b := bStart; b < bEnd; b++ {
				xBase := (b*seqLen + t) * features
				hPrevBase := b * hidden
				for h := 0; h < hidden; h++ {
					sum := r.B.Value.Data[h]
					for f := 0; f < features; f++ {
						sum += x.Data[xBase+f] * r.Wx.Value.Data[f*hidden+h]
					}
					for hh := 0; hh < hidden; hh++ {
						sum += hPrev[hPrevBase+hh] * r.Wh.Value.Data[hh*hidden+h]
					}
					hCur[b*hidden+h] = float32(math.Tanh(float64(sum)))
				}
			}
		})
		r.hStates[t+1] = hCur
		for b := 0; b < batch; b++ {
			copy(out.Data[(b*seqLen+t)*hidden:(b*seqLen+t+1)*hidden], hCur[b*hidden:(b+1)*hidden])
		}
	}
	return out, nil
}

// Backward is full BPTT: it walks timesteps in reverse, at each one
// combining the gradient arriving from that timestep's output (gradOut)
// with the gradient already propagated back from timestep t+1 (dhNext),
// then accumulates Wx/Wh/B gradients across every timestep (sum, not
// overwrite — same accumulate-then-reduce pattern LinearLayer uses for a
// Param shared across the parallelized batch dimension, just repeated once
// per timestep here).
func (r *RNNLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, seqLen, features, hidden := r.batch, r.seqLen, r.features, r.hidden
	gradIn := NewTensor(r.input.Shape)
	r.Wx.ZeroGrad()
	r.Wh.ZeroGrad()
	r.B.ZeroGrad()

	numChunks := numParallelChunks(batch)
	dhNext := make([]float32, batch*hidden)

	for t := seqLen - 1; t >= 0; t-- {
		hCur := r.hStates[t+1]
		hPrev := r.hStates[t]
		dz := make([]float32, batch*hidden)

		parallelChunks(batch, func(_, bStart, bEnd int) {
			for b := bStart; b < bEnd; b++ {
				for h := 0; h < hidden; h++ {
					idx := b*hidden + h
					goIdx := (b*seqLen+t)*hidden + h
					dhTotal := gradOut.Data[goIdx] + dhNext[idx]
					hc := hCur[idx]
					dz[idx] = dhTotal * (1 - hc*hc)
				}
			}
		})

		wxPartials := make([][]float32, numChunks)
		whPartials := make([][]float32, numChunks)
		bPartials := make([][]float32, numChunks)
		dhNextNew := make([]float32, batch*hidden)

		parallelChunks(batch, func(chunk, bStart, bEnd int) {
			wxGrad := make([]float32, features*hidden)
			whGrad := make([]float32, hidden*hidden)
			bGrad := make([]float32, hidden)
			wxPartials[chunk], whPartials[chunk], bPartials[chunk] = wxGrad, whGrad, bGrad
			for b := bStart; b < bEnd; b++ {
				xBase := (b*seqLen + t) * features
				hPrevBase := b * hidden
				for h := 0; h < hidden; h++ {
					g := dz[b*hidden+h]
					bGrad[h] += g
					for f := 0; f < features; f++ {
						wxGrad[f*hidden+h] += g * r.input.Data[xBase+f]
						gradIn.Data[xBase+f] += g * r.Wx.Value.Data[f*hidden+h]
					}
					for hh := 0; hh < hidden; hh++ {
						whGrad[hh*hidden+h] += g * hPrev[hPrevBase+hh]
						dhNextNew[b*hidden+hh] += g * r.Wh.Value.Data[hh*hidden+h]
					}
				}
			}
		})
		for chunk := 0; chunk < numChunks; chunk++ {
			for i, v := range wxPartials[chunk] {
				r.Wx.Grad.Data[i] += v
			}
			for i, v := range whPartials[chunk] {
				r.Wh.Grad.Data[i] += v
			}
			for i, v := range bPartials[chunk] {
				r.B.Grad.Data[i] += v
			}
		}
		dhNext = dhNextNew
	}
	return gradIn, nil
}

func (r *RNNLayer) Params() []*Param { return []*Param{r.Wx, r.Wh, r.B} }
