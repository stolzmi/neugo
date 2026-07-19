package Network

import "neugo/tensor"

type Flatten struct {
	InputHeight   int
	InputWidth    int
	InputChannels int
	Input         *tensor.Tensor3D
}

func NewFlatten() *Flatten {
	return &Flatten{}
}

func (f *Flatten) Forward(input *tensor.Tensor3D) []float32 {
	f.Input = input
	f.InputHeight = input.Height
	f.InputWidth = input.Width
	f.InputChannels = input.Channels

	return input.Flatten()
}

func (f *Flatten) Backward(outputGrad []float32) *tensor.Tensor3D {
	return tensor.Unflatten(outputGrad, f.InputHeight, f.InputWidth, f.InputChannels)
}
