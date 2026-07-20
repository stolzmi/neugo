// nn/residual.go
package nn

import "fmt"

// ResidualBlock adds its own input back onto the output of an inner module
// chain (the "main path"), the standard ResNet skip connection. The Module
// interface only supports a single input/output per layer, so rather than
// generalizing Sequential into an arbitrary graph, a residual connection is
// modeled as one more composite Module: it runs inner just like a nested
// Sequential, separately runs shortcut (or passes x through unchanged) to
// bring the input to the same shape as inner's output, and adds the two.
// That keeps every existing single-input/single-output layer, Sequential,
// and the serialization format unchanged — Residual is just another node
// that happens to have two branches internally.
type ResidualBlock struct {
	inner    []Module
	shortcut Module // nil means identity: shortcut output is x unchanged
	input    *Tensor
}

// Residual builds a block that computes inner(x) + shortcut(x) (or
// inner(x) + x if shortcut is nil). Use a nil shortcut when inner preserves
// x's shape (the common case); pass a projection module — e.g.
// Conv2DStrided with a 1x1 kernel — when inner changes channels and/or
// spatial size, so the shortcut's output shape matches inner's.
func Residual(shortcut Module, inner ...Module) *ResidualBlock {
	return &ResidualBlock{inner: inner, shortcut: shortcut}
}

func (r *ResidualBlock) OutputShape(inShape []int) ([]int, error) {
	shape := inShape
	for i, m := range r.inner {
		out, err := m.OutputShape(shape)
		if err != nil {
			return nil, fmt.Errorf("nn: Residual inner module %d: %w", i, err)
		}
		shape = out
	}
	shortShape := inShape
	if r.shortcut != nil {
		out, err := r.shortcut.OutputShape(inShape)
		if err != nil {
			return nil, fmt.Errorf("nn: Residual shortcut: %w", err)
		}
		shortShape = out
	}
	if !shapesEqual(shape, shortShape) {
		return nil, fmt.Errorf("nn: Residual inner output shape %v does not match shortcut output shape %v", shape, shortShape)
	}
	return shape, nil
}

func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (r *ResidualBlock) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	r.input = x
	out := x
	for i, m := range r.inner {
		next, err := m.Forward(ctx, out)
		if err != nil {
			return nil, fmt.Errorf("nn: Residual inner module %d: %w", i, err)
		}
		out = next
	}
	short := x
	if r.shortcut != nil {
		s, err := r.shortcut.Forward(ctx, x)
		if err != nil {
			return nil, fmt.Errorf("nn: Residual shortcut: %w", err)
		}
		short = s
	}
	if !shapesEqual(out.Shape, short.Shape) {
		return nil, fmt.Errorf("nn: Residual inner output shape %v does not match shortcut output shape %v", out.Shape, short.Shape)
	}
	sum := NewTensor(out.Shape)
	parallelChunks(len(sum.Data), func(_, start, end int) {
		for i := start; i < end; i++ {
			sum.Data[i] = out.Data[i] + short.Data[i]
		}
	})
	return sum, nil
}

// Backward distributes gradOut unchanged to both branches — d(a+b) is 1
// w.r.t. each addend — then sums the two resulting input-shaped gradients,
// since both branches trace back to the same block input x.
func (r *ResidualBlock) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradInner := gradOut
	for i := len(r.inner) - 1; i >= 0; i-- {
		g, err := r.inner[i].Backward(ctx, gradInner)
		if err != nil {
			return nil, fmt.Errorf("nn: Residual inner module %d backward: %w", i, err)
		}
		gradInner = g
	}
	gradShort := gradOut
	if r.shortcut != nil {
		g, err := r.shortcut.Backward(ctx, gradOut)
		if err != nil {
			return nil, fmt.Errorf("nn: Residual shortcut backward: %w", err)
		}
		gradShort = g
	}
	gradIn := NewTensor(r.input.Shape)
	parallelChunks(len(gradIn.Data), func(_, start, end int) {
		for i := start; i < end; i++ {
			gradIn.Data[i] = gradInner.Data[i] + gradShort.Data[i]
		}
	})
	return gradIn, nil
}

func (r *ResidualBlock) Params() []*Param {
	var params []*Param
	for _, m := range r.inner {
		params = append(params, m.Params()...)
	}
	if r.shortcut != nil {
		params = append(params, r.shortcut.Params()...)
	}
	return params
}
