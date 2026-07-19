package nn

import "fmt"

type SequentialModel struct {
	modules []Module
}

// Sequential validates the whole chain via OutputShape starting from
// inputShape, returning an error naming the offending module index — no
// runtime shape surprises. This is also where lazily-built modules
// (e.g. Linear(rng, 0, ...)) learn their real input size and allocate
// their Params.
func Sequential(inputShape []int, modules ...Module) (*SequentialModel, error) {
	shape := inputShape
	for i, m := range modules {
		out, err := m.OutputShape(shape)
		if err != nil {
			return nil, fmt.Errorf("nn: Sequential module %d: %w", i, err)
		}
		shape = out
	}
	return &SequentialModel{modules: modules}, nil
}

func (s *SequentialModel) Modules() []Module { return s.modules }

func (s *SequentialModel) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	out := x
	for i, m := range s.modules {
		var err error
		out, err = m.Forward(ctx, out)
		if err != nil {
			return nil, fmt.Errorf("nn: Sequential module %d: %w", i, err)
		}
	}
	return out, nil
}

func (s *SequentialModel) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	grad := gradOut
	for i := len(s.modules) - 1; i >= 0; i-- {
		var err error
		grad, err = s.modules[i].Backward(ctx, grad)
		if err != nil {
			return nil, fmt.Errorf("nn: Sequential module %d: %w", i, err)
		}
	}
	return grad, nil
}

func (s *SequentialModel) Params() []*Param {
	var params []*Param
	for _, m := range s.modules {
		params = append(params, m.Params()...)
	}
	return params
}

func (s *SequentialModel) OutputShape(inShape []int) ([]int, error) {
	shape := inShape
	for i, m := range s.modules {
		out, err := m.OutputShape(shape)
		if err != nil {
			return nil, fmt.Errorf("nn: Sequential module %d: %w", i, err)
		}
		shape = out
	}
	return shape, nil
}
