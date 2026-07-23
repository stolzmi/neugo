// nn/gru.go
package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// GRULayer is a standard GRU cell (Cho et al., 2014) using PyTorch's
// gate-bias convention: the reset gate R gates only the *hidden-side*
// contribution to the candidate N (not a combined x+h bias), which needs
// separate input-side (Bx) and hidden-side (Bh) biases rather than one
// shared bias — unlike RNNLayer/LSTMLayer, which only ever add x-side and
// h-side pre-activations together before the bias is even relevant.
// Wx/Bx and Wh/Bh's 3*hidden blocks are in (reset, update, candidate)
// order. Same [batch, seqLen, features] -> [batch, seqLen, hidden] shape
// contract and BPTT-only-parallel-over-batch constraint as RNN/LSTM.
type GRULayer struct {
	features, hidden int
	Wx, Wh           *Param
	Bx, Bh           *Param
	rng              *rand.Rand

	input         *Tensor
	batch, seqLen int
	// hStates[0] is the zero initial state; hStates[t+1] is h_t. r/z/n
	// States[t] are the three gate activations computed while producing
	// h_t. zhnStates[t] is Whn@h_{t-1}+bhn — the hidden-side candidate
	// pre-activation *before* the reset gate multiplies it in, needed
	// separately in Backward because R's gradient flows through it.
	hStates                   [][]float32
	rStates, zStates, nStates [][]float32
	zhnStates                 [][]float32
}

func GRU(rng *rand.Rand, features, hidden int, init Initializer) *GRULayer {
	if init == nil {
		init = XavierInit()
	}
	return &GRULayer{
		features: features, hidden: hidden, rng: rng,
		Wx: NewParam(init(rng, []int{features, 3 * hidden})),
		Wh: NewParam(init(rng, []int{hidden, 3 * hidden})),
		Bx: NewParam(NewTensor([]int{3 * hidden})),
		Bh: NewParam(NewTensor([]int{3 * hidden})),
	}
}

func (g *GRULayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 3 || inShape[2] != g.features {
		return nil, fmt.Errorf("nn: GRU expects input shape [batch, seqLen, %d], got %v", g.features, inShape)
	}
	return []int{inShape[0], inShape[1], g.hidden}, nil
}

func (g *GRULayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := g.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	batch, seqLen, features, hidden := x.Shape[0], x.Shape[1], g.features, g.hidden
	gateW := 3 * hidden
	g.input = x
	g.batch, g.seqLen = batch, seqLen

	out := NewTensor([]int{batch, seqLen, hidden})
	g.hStates = make([][]float32, seqLen+1)
	g.hStates[0] = make([]float32, batch*hidden)
	g.rStates = make([][]float32, seqLen)
	g.zStates = make([][]float32, seqLen)
	g.nStates = make([][]float32, seqLen)
	g.zhnStates = make([][]float32, seqLen)

	for t := 0; t < seqLen; t++ {
		hPrev := g.hStates[t]
		hCur := make([]float32, batch*hidden)
		rCur := make([]float32, batch*hidden)
		zCur := make([]float32, batch*hidden)
		nCur := make([]float32, batch*hidden)
		zhnCur := make([]float32, batch*hidden)

		parallelChunks(batch, func(_, bStart, bEnd int) {
			for b := bStart; b < bEnd; b++ {
				xBase := (b*seqLen + t) * features
				hPrevBase := b * hidden
				for h := 0; h < hidden; h++ {
					zxR := g.Bx.Value.Data[h]
					zxZ := g.Bx.Value.Data[hidden+h]
					zxN := g.Bx.Value.Data[2*hidden+h]
					zhR := g.Bh.Value.Data[h]
					zhZ := g.Bh.Value.Data[hidden+h]
					zhN := g.Bh.Value.Data[2*hidden+h]
					for f := 0; f < features; f++ {
						xv := x.Data[xBase+f]
						row := f * gateW
						zxR += xv * g.Wx.Value.Data[row+h]
						zxZ += xv * g.Wx.Value.Data[row+hidden+h]
						zxN += xv * g.Wx.Value.Data[row+2*hidden+h]
					}
					for hh := 0; hh < hidden; hh++ {
						hv := hPrev[hPrevBase+hh]
						row := hh * gateW
						zhR += hv * g.Wh.Value.Data[row+h]
						zhZ += hv * g.Wh.Value.Data[row+hidden+h]
						zhN += hv * g.Wh.Value.Data[row+2*hidden+h]
					}
					r := sigmoid32(zxR + zhR)
					z := sigmoid32(zxZ + zhZ)
					n := float32(math.Tanh(float64(zxN + r*zhN)))
					hp := hPrev[hPrevBase+h]
					ht := (1-z)*n + z*hp

					idx := b*hidden + h
					rCur[idx], zCur[idx], nCur[idx] = r, z, n
					zhnCur[idx] = zhN
					hCur[idx] = ht
				}
			}
		})

		g.rStates[t], g.zStates[t], g.nStates[t] = rCur, zCur, nCur
		g.zhnStates[t] = zhnCur
		g.hStates[t+1] = hCur
		for b := 0; b < batch; b++ {
			copy(out.Data[(b*seqLen+t)*hidden:(b*seqLen+t+1)*hidden], hCur[b*hidden:(b+1)*hidden])
		}
	}
	return out, nil
}

// Backward is full BPTT. h_t = (1-Z)*N + Z*h_{t-1} has two paths back to
// h_{t-1}: the direct Z*h_{t-1} term, and through R/Z/N's own dependence
// on h_{t-1} via their gate pre-activations — both are accumulated into
// dhPrevNew below.
func (g *GRULayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	batch, seqLen, features, hidden := g.batch, g.seqLen, g.features, g.hidden
	gateW := 3 * hidden
	gradIn := NewTensor(g.input.Shape)
	g.Wx.ZeroGrad()
	g.Wh.ZeroGrad()
	g.Bx.ZeroGrad()
	g.Bh.ZeroGrad()

	numChunks := numParallelChunks(batch)
	dhNext := make([]float32, batch*hidden)

	for t := seqLen - 1; t >= 0; t-- {
		hPrev := g.hStates[t]
		rCur, zCur, nCur := g.rStates[t], g.zStates[t], g.nStates[t]
		zhnCur := g.zhnStates[t]

		dzxR := make([]float32, batch*hidden) // == dzhR (pre_r = zxR+zhR)
		dzxZ := make([]float32, batch*hidden) // == dzhZ (pre_z = zxZ+zhZ)
		dzxN := make([]float32, batch*hidden)
		dzhN := make([]float32, batch*hidden)
		dhPrevDirect := make([]float32, batch*hidden)

		parallelChunks(batch, func(_, bStart, bEnd int) {
			for b := bStart; b < bEnd; b++ {
				for h := 0; h < hidden; h++ {
					idx := b*hidden + h
					goIdx := (b*seqLen+t)*hidden + h
					dh := gradOut.Data[goIdx] + dhNext[idx]
					r, z, n := rCur[idx], zCur[idx], nCur[idx]
					hp := hPrev[b*hidden+h]

					dN := dh * (1 - z)
					dZ := dh * (hp - n)
					dhPrevDirect[idx] = dh * z

					dPreN := dN * (1 - n*n)
					dzxN[idx] = dPreN
					dzhN[idx] = dPreN * r
					dR := dPreN * zhnCur[idx]

					dPreZ := dZ * z * (1 - z)
					dzxZ[idx] = dPreZ

					dPreR := dR * r * (1 - r)
					dzxR[idx] = dPreR
				}
			}
		})

		wxPartials := make([][]float32, numChunks)
		whPartials := make([][]float32, numChunks)
		bxPartials := make([][]float32, numChunks)
		bhPartials := make([][]float32, numChunks)
		dhPrevNew := make([]float32, batch*hidden)

		parallelChunks(batch, func(chunk, bStart, bEnd int) {
			wxGrad := make([]float32, features*gateW)
			whGrad := make([]float32, hidden*gateW)
			bxGrad := make([]float32, gateW)
			bhGrad := make([]float32, gateW)
			wxPartials[chunk], whPartials[chunk] = wxGrad, whGrad
			bxPartials[chunk], bhPartials[chunk] = bxGrad, bhGrad
			for b := bStart; b < bEnd; b++ {
				xBase := (b*seqLen + t) * features
				hPrevBase := b * hidden
				for h := 0; h < hidden; h++ {
					idx := b*hidden + h
					dr, dz, dn, dhn := dzxR[idx], dzxZ[idx], dzxN[idx], dzhN[idx]

					bxGrad[h] += dr
					bxGrad[hidden+h] += dz
					bxGrad[2*hidden+h] += dn
					bhGrad[h] += dr
					bhGrad[hidden+h] += dz
					bhGrad[2*hidden+h] += dhn

					for f := 0; f < features; f++ {
						xv := g.input.Data[xBase+f]
						row := f * gateW
						wxGrad[row+h] += dr * xv
						wxGrad[row+hidden+h] += dz * xv
						wxGrad[row+2*hidden+h] += dn * xv

						gradIn.Data[xBase+f] += dr*g.Wx.Value.Data[row+h] +
							dz*g.Wx.Value.Data[row+hidden+h] +
							dn*g.Wx.Value.Data[row+2*hidden+h]
					}
					for hh := 0; hh < hidden; hh++ {
						hv := hPrev[hPrevBase+hh]
						row := hh * gateW
						whGrad[row+h] += dr * hv
						whGrad[row+hidden+h] += dz * hv
						whGrad[row+2*hidden+h] += dhn * hv

						dhPrevNew[hPrevBase+hh] += dr*g.Wh.Value.Data[row+h] +
							dz*g.Wh.Value.Data[row+hidden+h] +
							dhn*g.Wh.Value.Data[row+2*hidden+h]
					}
				}
			}
		})

		for chunk := 0; chunk < numChunks; chunk++ {
			for i, v := range wxPartials[chunk] {
				g.Wx.Grad.Data[i] += v
			}
			for i, v := range whPartials[chunk] {
				g.Wh.Grad.Data[i] += v
			}
			for i, v := range bxPartials[chunk] {
				g.Bx.Grad.Data[i] += v
			}
			for i, v := range bhPartials[chunk] {
				g.Bh.Grad.Data[i] += v
			}
		}
		for i := range dhPrevNew {
			dhPrevNew[i] += dhPrevDirect[i]
		}
		dhNext = dhPrevNew
	}
	return gradIn, nil
}

func (g *GRULayer) Params() []*Param { return []*Param{g.Wx, g.Wh, g.Bx, g.Bh} }
