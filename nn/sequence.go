// nn/sequence.go
package nn

import "fmt"

// LastTimestepLayer reduces a recurrent layer's full-sequence output
// [batch, seqLen, hidden] down to just its final timestep [batch, hidden]
// — the usual head for sequence classification, composed after an
// RNN/LSTM/GRU the same way Flatten composes after a conv stack, rather
// than baking a "return only the last state" flag into the recurrent
// layers themselves.
type LastTimestepLayer struct {
	inputShape []int
}

func LastTimestep() *LastTimestepLayer { return &LastTimestepLayer{} }

func (l *LastTimestepLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 3 {
		return nil, fmt.Errorf("nn: LastTimestep expects input shape [batch, seqLen, hidden], got %v", inShape)
	}
	return []int{inShape[0], inShape[2]}, nil
}

func (l *LastTimestepLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := l.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	l.inputShape = x.Shape
	batch, seqLen, hidden := x.Shape[0], x.Shape[1], x.Shape[2]
	out := NewTensor([]int{batch, hidden})
	for b := 0; b < batch; b++ {
		src := (b*seqLen + seqLen - 1) * hidden
		copy(out.Data[b*hidden:(b+1)*hidden], x.Data[src:src+hidden])
	}
	return out, nil
}

func (l *LastTimestepLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(l.inputShape)
	batch, seqLen, hidden := l.inputShape[0], l.inputShape[1], l.inputShape[2]
	for b := 0; b < batch; b++ {
		dst := (b*seqLen + seqLen - 1) * hidden
		copy(gradIn.Data[dst:dst+hidden], gradOut.Data[b*hidden:(b+1)*hidden])
	}
	return gradIn, nil
}

func (l *LastTimestepLayer) Params() []*Param { return nil }
