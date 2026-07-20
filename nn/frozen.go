// nn/frozen.go
package nn

// FrozenModule wraps another Module, forwarding Forward and Backward
// unchanged but reporting no Params — the standard mechanism for
// fine-tuning: an optimizer only ever sees model.Params(), so a frozen
// layer's weights are simply never in that list and never get a Step
// applied, while gradients still flow through Backward to whatever
// trainable layers sit earlier in the chain.
type FrozenModule struct {
	inner Module
}

// Frozen wraps inner so its Params are excluded from training.
func Frozen(inner Module) *FrozenModule {
	return &FrozenModule{inner: inner}
}

func (f *FrozenModule) OutputShape(inShape []int) ([]int, error) {
	return f.inner.OutputShape(inShape)
}

func (f *FrozenModule) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	return f.inner.Forward(ctx, x)
}

func (f *FrozenModule) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	return f.inner.Backward(ctx, gradOut)
}

func (f *FrozenModule) Params() []*Param { return nil }
