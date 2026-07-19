package nn

import "fmt"

// Tensor is the single data type flowing through the network.
// Batch-first: dense is [batch, features], conv is [batch, h, w, channels].
type Tensor struct {
	Data  []float32
	Shape []int
}

func shapeSize(shape []int) int {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return size
}

func NewTensor(shape []int) *Tensor {
	return &Tensor{
		Data:  make([]float32, shapeSize(shape)),
		Shape: append([]int(nil), shape...),
	}
}

func NewTensorFromData(data []float32, shape []int) (*Tensor, error) {
	if want := shapeSize(shape); want != len(data) {
		return nil, fmt.Errorf("nn: data length %d does not match shape %v (size %d)", len(data), shape, want)
	}
	return &Tensor{Data: data, Shape: append([]int(nil), shape...)}, nil
}

func (t *Tensor) Size() int {
	return len(t.Data)
}

func (t *Tensor) Clone() *Tensor {
	d := make([]float32, len(t.Data))
	copy(d, t.Data)
	return &Tensor{Data: d, Shape: append([]int(nil), t.Shape...)}
}
