package nn

import (
	"math"
	"math/rand"
)

// Initializer fills a freshly-shaped Tensor with starting weights.
type Initializer func(rng *rand.Rand, shape []int) *Tensor

// fanInOut supports the two weight-tensor shapes used in this codebase:
// Linear weights are [in, out]; Conv2D kernels are [outC, inC, kh, kw].
func fanInOut(shape []int) (fanIn, fanOut int) {
	switch len(shape) {
	case 2:
		return shape[0], shape[1]
	case 4:
		receptive := shape[2] * shape[3]
		return shape[1] * receptive, shape[0] * receptive
	default:
		n := shapeSize(shape)
		return n, n
	}
}

func XavierInit() Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		fanIn, fanOut := fanInOut(shape)
		limit := float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
		t := NewTensor(shape)
		for i := range t.Data {
			t.Data[i] = (rng.Float32()*2 - 1) * limit
		}
		return t
	}
}

func HeInit() Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		fanIn, _ := fanInOut(shape)
		std := float32(math.Sqrt(2.0 / float64(fanIn)))
		t := NewTensor(shape)
		for i := range t.Data {
			t.Data[i] = float32(rng.NormFloat64()) * std
		}
		return t
	}
}

func ZerosInit() Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		return NewTensor(shape)
	}
}

func UniformInit(low, high float32) Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		t := NewTensor(shape)
		for i := range t.Data {
			t.Data[i] = low + rng.Float32()*(high-low)
		}
		return t
	}
}

func NormalInit(mean, std float32) Initializer {
	return func(rng *rand.Rand, shape []int) *Tensor {
		t := NewTensor(shape)
		for i := range t.Data {
			t.Data[i] = mean + float32(rng.NormFloat64())*std
		}
		return t
	}
}
