package nn

import "fmt"

type FlattenLayer struct {
	inputShape []int
}

func Flatten() *FlattenLayer { return &FlattenLayer{} }

func (f *FlattenLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) < 2 {
		return nil, fmt.Errorf("nn: Flatten expects at least [batch, ...], got %v", inShape)
	}
	size := 1
	for _, d := range inShape[1:] {
		size *= d
	}
	return []int{inShape[0], size}, nil
}

func (f *FlattenLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	f.inputShape = append([]int(nil), x.Shape...)
	out, err := f.OutputShape(x.Shape)
	if err != nil {
		return nil, err
	}
	data := append([]float32(nil), x.Data...)
	return &Tensor{Data: data, Shape: out}, nil
}

func (f *FlattenLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	data := append([]float32(nil), gradOut.Data...)
	return &Tensor{Data: data, Shape: append([]int(nil), f.inputShape...)}, nil
}

func (f *FlattenLayer) Params() []*Param { return nil }
