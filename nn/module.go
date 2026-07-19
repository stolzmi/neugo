package nn

import "math/rand"

// Param is a trainable tensor pair an Optimizer can see and update.
type Param struct {
	Value *Tensor
	Grad  *Tensor
}

func NewParam(value *Tensor) *Param {
	return &Param{Value: value, Grad: NewTensor(value.Shape)}
}

func (p *Param) ZeroGrad() {
	for i := range p.Grad.Data {
		p.Grad.Data[i] = 0
	}
}

// Mode distinguishes training from inference (Dropout, BatchNorm).
type Mode int

const (
	Inference Mode = iota
	Train
)

// Context threads mode and RNG through forward/backward.
type Context struct {
	Mode Mode
	RNG  *rand.Rand
}

type Module interface {
	Forward(ctx *Context, x *Tensor) (*Tensor, error)
	Backward(ctx *Context, gradOut *Tensor) (*Tensor, error)
	Params() []*Param
	OutputShape(inShape []int) ([]int, error)
}
