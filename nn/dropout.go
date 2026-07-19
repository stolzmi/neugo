package nn

type DropoutLayer struct {
	rate float32
	mask []float32 // per-element scale factor recorded by the last Train-mode Forward (0 or 1/(1-rate)); nil after an Inference Forward
}

func Dropout(rate float32) *DropoutLayer { return &DropoutLayer{rate: rate} }

func (d *DropoutLayer) OutputShape(inShape []int) ([]int, error) { return inShape, nil }

func (d *DropoutLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if ctx.Mode != Train || d.rate == 0 {
		d.mask = nil
		return x.Clone(), nil
	}
	scale := 1.0 / (1.0 - d.rate)
	out := NewTensor(x.Shape)
	d.mask = make([]float32, len(x.Data))
	for i, v := range x.Data {
		if ctx.RNG.Float32() > d.rate {
			out.Data[i] = v * scale
			d.mask[i] = scale
		}
	}
	return out, nil
}

func (d *DropoutLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(gradOut.Shape)
	if d.mask == nil {
		copy(gradIn.Data, gradOut.Data)
		return gradIn, nil
	}
	for i := range gradOut.Data {
		gradIn.Data[i] = gradOut.Data[i] * d.mask[i]
	}
	return gradIn, nil
}

func (d *DropoutLayer) Params() []*Param { return nil }
