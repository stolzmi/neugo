// nn/lstm.go
package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// LSTMLayer is a standard LSTM cell (Hochreiter & Schmidhuber, 1997) with
// combined gate weights: Wx is [features, 4*hidden] and Wh is [hidden,
// 4*hidden], the four hidden-wide blocks in (input, forget, cell-candidate,
// output) order. Same [batch, seqLen, features] -> [batch, seqLen, hidden]
// shape contract as RNNLayer, and the same BPTT-only-parallel-over-batch
// constraint.
type LSTMLayer struct {
	features, hidden int
	Wx, Wh           *Param
	B                *Param
	rng              *rand.Rand

	input         *Tensor
	batch, seqLen int
	// hStates[0]/cStates[0] are the zero initial states; hStates[t+1]/
	// cStates[t+1] are h_t/c_t. iStates..oStates[t] are the four gate
	// activations computed while producing h_t (length seqLen, not
	// seqLen+1 — there's no "gate 0" before the first timestep).
	hStates, cStates                   [][]float32
	iStates, fStates, gStates, oStates [][]float32
}

func LSTM(rng *rand.Rand, features, hidden int, init Initializer) *LSTMLayer {
	if init == nil {
		init = XavierInit()
	}
	return &LSTMLayer{
		features: features, hidden: hidden, rng: rng,
		Wx: NewParam(init(rng, []int{features, 4 * hidden})),
		Wh: NewParam(init(rng, []int{hidden, 4 * hidden})),
		B:  NewParam(NewTensor([]int{4 * hidden})),
	}
}

func (l *LSTMLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 3 || inShape[2] != l.features {
		return nil, fmt.Errorf("nn: LSTM expects input shape [batch, seqLen, %d], got %v", l.features, inShape)
	}
	return []int{inShape[0], inShape[1], l.hidden}, nil
}

func sigmoid32(x float32) float32 { return float32(1 / (1 + math.Exp(float64(-x)))) }

func (l *LSTMLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := l.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	batch, seqLen, features, hidden := x.Shape[0], x.Shape[1], l.features, l.hidden
	gateW := 4 * hidden
	l.input = x
	l.batch, l.seqLen = batch, seqLen

	out := NewTensor([]int{batch, seqLen, hidden})
	l.hStates = make([][]float32, seqLen+1)
	l.cStates = make([][]float32, seqLen+1)
	l.hStates[0] = make([]float32, batch*hidden)
	l.cStates[0] = make([]float32, batch*hidden)
	l.iStates = make([][]float32, seqLen)
	l.fStates = make([][]float32, seqLen)
	l.gStates = make([][]float32, seqLen)
	l.oStates = make([][]float32, seqLen)

	for t := 0; t < seqLen; t++ {
		hPrev := l.hStates[t]
		cPrev := l.cStates[t]
		hCur := make([]float32, batch*hidden)
		cCur := make([]float32, batch*hidden)
		iCur := make([]float32, batch*hidden)
		fCur := make([]float32, batch*hidden)
		gCur := make([]float32, batch*hidden)
		oCur := make([]float32, batch*hidden)

		parallelChunks(batch, func(_, bStart, bEnd int) {
			for b := bStart; b < bEnd; b++ {
				xBase := (b*seqLen + t) * features
				hPrevBase := b * hidden
				for h := 0; h < hidden; h++ {
					zi := l.B.Value.Data[h]
					zf := l.B.Value.Data[hidden+h]
					zg := l.B.Value.Data[2*hidden+h]
					zo := l.B.Value.Data[3*hidden+h]
					for f := 0; f < features; f++ {
						xv := x.Data[xBase+f]
						row := f * gateW
						zi += xv * l.Wx.Value.Data[row+h]
						zf += xv * l.Wx.Value.Data[row+hidden+h]
						zg += xv * l.Wx.Value.Data[row+2*hidden+h]
						zo += xv * l.Wx.Value.Data[row+3*hidden+h]
					}
					for hh := 0; hh < hidden; hh++ {
						hv := hPrev[hPrevBase+hh]
						row := hh * gateW
						zi += hv * l.Wh.Value.Data[row+h]
						zf += hv * l.Wh.Value.Data[row+hidden+h]
						zg += hv * l.Wh.Value.Data[row+2*hidden+h]
						zo += hv * l.Wh.Value.Data[row+3*hidden+h]
					}
					it := sigmoid32(zi)
					ft := sigmoid32(zf)
					gt := float32(math.Tanh(float64(zg)))
					ot := sigmoid32(zo)
					ct := ft*cPrev[hPrevBase+h] + it*gt
					ht := ot * float32(math.Tanh(float64(ct)))

					idx := b*hidden + h
					iCur[idx], fCur[idx], gCur[idx], oCur[idx] = it, ft, gt, ot
					cCur[idx] = ct
					hCur[idx] = ht
				}
			}
		})

		l.iStates[t], l.fStates[t], l.gStates[t], l.oStates[t] = iCur, fCur, gCur, oCur
		l.cStates[t+1] = cCur
		l.hStates[t+1] = hCur
		for b := 0; b < batch; b++ {
			copy(out.Data[(b*seqLen+t)*hidden:(b*seqLen+t+1)*hidden], hCur[b*hidden:(b+1)*hidden])
		}
	}
	return out, nil
}

// Backward is full BPTT, propagating both the hidden-state gradient
// (dhNext) and the cell-state gradient (dcNext) back through time — the
// two-state generalization of RNNLayer.Backward's single dhNext.
func (l *LSTMLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, seqLen, features, hidden := l.batch, l.seqLen, l.features, l.hidden
	gateW := 4 * hidden
	gradIn := NewTensor(l.input.Shape)
	l.Wx.ZeroGrad()
	l.Wh.ZeroGrad()
	l.B.ZeroGrad()

	numChunks := numParallelChunks(batch)
	dhNext := make([]float32, batch*hidden)
	dcNext := make([]float32, batch*hidden)

	for t := seqLen - 1; t >= 0; t-- {
		hPrev := l.hStates[t]
		cPrev := l.cStates[t]
		iCur, fCur, gCur, oCur := l.iStates[t], l.fStates[t], l.gStates[t], l.oStates[t]
		cCur := l.cStates[t+1]

		dzi := make([]float32, batch*hidden)
		dzf := make([]float32, batch*hidden)
		dzg := make([]float32, batch*hidden)
		dzo := make([]float32, batch*hidden)
		dcPrevNew := make([]float32, batch*hidden)

		parallelChunks(batch, func(_, bStart, bEnd int) {
			for b := bStart; b < bEnd; b++ {
				for h := 0; h < hidden; h++ {
					idx := b*hidden + h
					goIdx := (b*seqLen+t)*hidden + h
					dh := gradOut.Data[goIdx] + dhNext[idx]
					it, ft, gt, ot := iCur[idx], fCur[idx], gCur[idx], oCur[idx]
					ct := cCur[idx]
					tanhCt := float32(math.Tanh(float64(ct)))
					dc := dh*ot*(1-tanhCt*tanhCt) + dcNext[idx]

					dzo[idx] = dh * tanhCt * ot * (1 - ot)
					dzi[idx] = dc * gt * it * (1 - it)
					dzf[idx] = dc * cPrev[idx] * ft * (1 - ft)
					dzg[idx] = dc * it * (1 - gt*gt)
					dcPrevNew[idx] = dc * ft
				}
			}
		})

		wxPartials := make([][]float32, numChunks)
		whPartials := make([][]float32, numChunks)
		bPartials := make([][]float32, numChunks)
		dhPrevNew := make([]float32, batch*hidden)

		parallelChunks(batch, func(chunk, bStart, bEnd int) {
			wxGrad := make([]float32, features*gateW)
			whGrad := make([]float32, hidden*gateW)
			bGrad := make([]float32, gateW)
			wxPartials[chunk], whPartials[chunk], bPartials[chunk] = wxGrad, whGrad, bGrad
			for b := bStart; b < bEnd; b++ {
				xBase := (b*seqLen + t) * features
				hPrevBase := b * hidden
				for h := 0; h < hidden; h++ {
					idx := b*hidden + h
					gi, gf, gg, go_ := dzi[idx], dzf[idx], dzg[idx], dzo[idx]

					bGrad[h] += gi
					bGrad[hidden+h] += gf
					bGrad[2*hidden+h] += gg
					bGrad[3*hidden+h] += go_

					for f := 0; f < features; f++ {
						xv := l.input.Data[xBase+f]
						row := f * gateW
						wxGrad[row+h] += gi * xv
						wxGrad[row+hidden+h] += gf * xv
						wxGrad[row+2*hidden+h] += gg * xv
						wxGrad[row+3*hidden+h] += go_ * xv

						gradIn.Data[xBase+f] += gi*l.Wx.Value.Data[row+h] +
							gf*l.Wx.Value.Data[row+hidden+h] +
							gg*l.Wx.Value.Data[row+2*hidden+h] +
							go_*l.Wx.Value.Data[row+3*hidden+h]
					}
					for hh := 0; hh < hidden; hh++ {
						hv := hPrev[hPrevBase+hh]
						row := hh * gateW
						whGrad[row+h] += gi * hv
						whGrad[row+hidden+h] += gf * hv
						whGrad[row+2*hidden+h] += gg * hv
						whGrad[row+3*hidden+h] += go_ * hv

						dhPrevNew[hPrevBase+hh] += gi*l.Wh.Value.Data[row+h] +
							gf*l.Wh.Value.Data[row+hidden+h] +
							gg*l.Wh.Value.Data[row+2*hidden+h] +
							go_*l.Wh.Value.Data[row+3*hidden+h]
					}
				}
			}
		})

		for chunk := 0; chunk < numChunks; chunk++ {
			for i, v := range wxPartials[chunk] {
				l.Wx.Grad.Data[i] += v
			}
			for i, v := range whPartials[chunk] {
				l.Wh.Grad.Data[i] += v
			}
			for i, v := range bPartials[chunk] {
				l.B.Grad.Data[i] += v
			}
		}
		dhNext = dhPrevNew
		dcNext = dcPrevNew
	}
	return gradIn, nil
}

func (l *LSTMLayer) Params() []*Param { return []*Param{l.Wx, l.Wh, l.B} }
