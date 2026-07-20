// nn/pooling.go
package nn

import "fmt"

type MaxPool2DLayer struct {
	poolSize, stride int
	input            *Tensor
	outH, outW       int
	maxIdx           []int // flat input index chosen for each output element
}

func MaxPool2D(poolSize, stride int) *MaxPool2DLayer {
	return &MaxPool2DLayer{poolSize: poolSize, stride: stride}
}

func (m *MaxPool2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 {
		return nil, fmt.Errorf("nn: MaxPool2D expects input shape [batch, h, w, c], got %v", inShape)
	}
	if m.poolSize <= 0 || m.stride <= 0 {
		return nil, fmt.Errorf("nn: MaxPool2D poolSize and stride must be positive, got poolSize=%d, stride=%d", m.poolSize, m.stride)
	}
	outH := (inShape[1]-m.poolSize)/m.stride + 1
	outW := (inShape[2]-m.poolSize)/m.stride + 1
	return []int{inShape[0], outH, outW, inShape[3]}, nil
}

func (m *MaxPool2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 4 {
		return nil, fmt.Errorf("nn: MaxPool2D expects input shape [batch, h, w, c], got %v", x.Shape)
	}
	m.input = x
	batch, h, w, ch := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outH := (h-m.poolSize)/m.stride + 1
	outW := (w-m.poolSize)/m.stride + 1
	m.outH, m.outW = outH, outW
	out := NewTensor([]int{batch, outH, outW, ch})
	m.maxIdx = make([]int, batch*outH*outW*ch)

	// Batch-parallel: out and maxIdx writes are per-b disjoint.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					for c := 0; c < ch; c++ {
						best := float32(0)
						bestIdx := -1
						for ph := 0; ph < m.poolSize; ph++ {
							ih := oh*m.stride + ph
							for pw := 0; pw < m.poolSize; pw++ {
								iw := ow*m.stride + pw
								idx := ((b*h+ih)*w+iw)*ch + c
								v := x.Data[idx]
								if bestIdx == -1 || v > best {
									best = v
									bestIdx = idx
								}
							}
						}
						outIdx := ((b*outH+oh)*outW+ow)*ch + c
						out.Data[outIdx] = best
						m.maxIdx[outIdx] = bestIdx
					}
				}
			}
		}
	})
	return out, nil
}

func (m *MaxPool2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(m.input.Shape)
	// Batch-parallel over contiguous per-b ranges of maxIdx: a max index
	// always points inside its own batch element, so gradIn writes from
	// different chunks can never collide (even with overlapping windows).
	batch := m.input.Shape[0]
	perB := m.outH * m.outW * m.input.Shape[3]
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for outIdx := bStart * perB; outIdx < bEnd*perB; outIdx++ {
			gradIn.Data[m.maxIdx[outIdx]] += gradOut.Data[outIdx]
		}
	})
	return gradIn, nil
}

func (m *MaxPool2DLayer) Params() []*Param { return nil }

type AvgPool2DLayer struct {
	poolSize, stride int
	inputShape       []int
	outH, outW       int
}

func AvgPool2D(poolSize, stride int) *AvgPool2DLayer {
	return &AvgPool2DLayer{poolSize: poolSize, stride: stride}
}

func (a *AvgPool2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 {
		return nil, fmt.Errorf("nn: AvgPool2D expects input shape [batch, h, w, c], got %v", inShape)
	}
	if a.poolSize <= 0 || a.stride <= 0 {
		return nil, fmt.Errorf("nn: AvgPool2D poolSize and stride must be positive, got poolSize=%d, stride=%d", a.poolSize, a.stride)
	}
	outH := (inShape[1]-a.poolSize)/a.stride + 1
	outW := (inShape[2]-a.poolSize)/a.stride + 1
	return []int{inShape[0], outH, outW, inShape[3]}, nil
}

func (a *AvgPool2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if len(x.Shape) != 4 {
		return nil, fmt.Errorf("nn: AvgPool2D expects input shape [batch, h, w, c], got %v", x.Shape)
	}
	a.inputShape = x.Shape
	batch, h, w, ch := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outH := (h-a.poolSize)/a.stride + 1
	outW := (w-a.poolSize)/a.stride + 1
	a.outH, a.outW = outH, outW
	out := NewTensor([]int{batch, outH, outW, ch})
	area := float32(a.poolSize * a.poolSize)

	// Batch-parallel: out writes are per-b disjoint.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					for c := 0; c < ch; c++ {
						var sum float32
						for ph := 0; ph < a.poolSize; ph++ {
							ih := oh*a.stride + ph
							for pw := 0; pw < a.poolSize; pw++ {
								iw := ow*a.stride + pw
								sum += x.Data[((b*h+ih)*w+iw)*ch+c]
							}
						}
						out.Data[((b*outH+oh)*outW+ow)*ch+c] = sum / area
					}
				}
			}
		}
	})
	return out, nil
}

func (a *AvgPool2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(a.inputShape)
	h, w, ch := a.inputShape[1], a.inputShape[2], a.inputShape[3]
	area := float32(a.poolSize * a.poolSize)
	batch := a.inputShape[0]

	// Batch-parallel: gradIn writes are per-b disjoint.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for oh := 0; oh < a.outH; oh++ {
				for ow := 0; ow < a.outW; ow++ {
					for c := 0; c < ch; c++ {
						g := gradOut.Data[((b*a.outH+oh)*a.outW+ow)*ch+c] / area
						for ph := 0; ph < a.poolSize; ph++ {
							ih := oh*a.stride + ph
							for pw := 0; pw < a.poolSize; pw++ {
								iw := ow*a.stride + pw
								gradIn.Data[((b*h+ih)*w+iw)*ch+c] += g
							}
						}
					}
				}
			}
		}
	})
	return gradIn, nil
}

func (a *AvgPool2DLayer) Params() []*Param { return nil }
