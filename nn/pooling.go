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

// adaptiveBinRange returns the [start, end) input range that output index i
// (of outSize total) pools over, using the standard adaptive-pooling
// formula (PyTorch's AdaptiveAvgPool2d): start = floor(i*inSize/outSize),
// end = ceil((i+1)*inSize/outSize). Bins always cover [0, inSize) but are
// only non-overlapping when outSize evenly divides inSize — otherwise
// adjacent bins share their boundary element (e.g. inSize=3, outSize=2
// gives bins [0,2) and [1,3)), which is expected/matches PyTorch, and is
// why Backward accumulates into gradIn with += rather than assigning.
func adaptiveBinRange(i, outSize, inSize int) (start, end int) {
	start = i * inSize / outSize
	end = ((i+1)*inSize + outSize - 1) / outSize
	return
}

// AdaptiveAvgPool2DLayer average-pools [batch, h, w, c] down to a fixed
// [batch, outH, outW, c] regardless of input spatial size, dividing h and w
// into outH/outW (possibly unevenly sized) contiguous bins.
type AdaptiveAvgPool2DLayer struct {
	outH, outW int
	inputShape []int
}

func AdaptiveAvgPool2D(outH, outW int) *AdaptiveAvgPool2DLayer {
	return &AdaptiveAvgPool2DLayer{outH: outH, outW: outW}
}

// GlobalAvgPool2D averages every spatial position down to a single value
// per channel — AdaptiveAvgPool2D(1, 1).
func GlobalAvgPool2D() *AdaptiveAvgPool2DLayer { return AdaptiveAvgPool2D(1, 1) }

func (a *AdaptiveAvgPool2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 {
		return nil, fmt.Errorf("nn: AdaptiveAvgPool2D expects input shape [batch, h, w, c], got %v", inShape)
	}
	if a.outH <= 0 || a.outW <= 0 || a.outH > inShape[1] || a.outW > inShape[2] {
		return nil, fmt.Errorf("nn: AdaptiveAvgPool2D target size (%d, %d) must be positive and no larger than input spatial size (%d, %d)", a.outH, a.outW, inShape[1], inShape[2])
	}
	return []int{inShape[0], a.outH, a.outW, inShape[3]}, nil
}

func (a *AdaptiveAvgPool2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := a.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	a.inputShape = x.Shape
	batch, h, w, ch := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	out := NewTensor([]int{batch, a.outH, a.outW, ch})

	// Batch-parallel: out writes are per-b disjoint.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for oh := 0; oh < a.outH; oh++ {
				hStart, hEnd := adaptiveBinRange(oh, a.outH, h)
				for ow := 0; ow < a.outW; ow++ {
					wStart, wEnd := adaptiveBinRange(ow, a.outW, w)
					area := float32((hEnd - hStart) * (wEnd - wStart))
					for c := 0; c < ch; c++ {
						var sum float32
						for ih := hStart; ih < hEnd; ih++ {
							for iw := wStart; iw < wEnd; iw++ {
								sum += x.Data[((b*h+ih)*w+iw)*ch+c]
							}
						}
						out.Data[((b*a.outH+oh)*a.outW+ow)*ch+c] = sum / area
					}
				}
			}
		}
	})
	return out, nil
}

func (a *AdaptiveAvgPool2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(a.inputShape)
	batch, h, w, ch := a.inputShape[0], a.inputShape[1], a.inputShape[2], a.inputShape[3]

	// Batch-parallel: gradIn writes are per-b disjoint across goroutines
	// (bins never cross batch elements); within one b, bins may overlap
	// when outH/outW don't evenly divide h/w, so accumulation into gradIn
	// uses += and relies on that inner loop staying single-threaded.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for oh := 0; oh < a.outH; oh++ {
				hStart, hEnd := adaptiveBinRange(oh, a.outH, h)
				for ow := 0; ow < a.outW; ow++ {
					wStart, wEnd := adaptiveBinRange(ow, a.outW, w)
					area := float32((hEnd - hStart) * (wEnd - wStart))
					for c := 0; c < ch; c++ {
						g := gradOut.Data[((b*a.outH+oh)*a.outW+ow)*ch+c] / area
						for ih := hStart; ih < hEnd; ih++ {
							for iw := wStart; iw < wEnd; iw++ {
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

func (a *AdaptiveAvgPool2DLayer) Params() []*Param { return nil }

// GlobalMaxPool2DLayer max-pools every spatial position down to a single
// value per channel: [batch, h, w, c] -> [batch, 1, 1, c].
type GlobalMaxPool2DLayer struct {
	inputShape []int
	maxIdx     []int // per (batch, channel): flat input index of the max
}

func GlobalMaxPool2D() *GlobalMaxPool2DLayer { return &GlobalMaxPool2DLayer{} }

func (g *GlobalMaxPool2DLayer) OutputShape(inShape []int) ([]int, error) {
	if len(inShape) != 4 {
		return nil, fmt.Errorf("nn: GlobalMaxPool2D expects input shape [batch, h, w, c], got %v", inShape)
	}
	return []int{inShape[0], 1, 1, inShape[3]}, nil
}

func (g *GlobalMaxPool2DLayer) Forward(ctx *Context, x *Tensor) (*Tensor, error) {
	if _, err := g.OutputShape(x.Shape); err != nil {
		return nil, err
	}
	g.inputShape = x.Shape
	batch, h, w, ch := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	out := NewTensor([]int{batch, 1, 1, ch})
	g.maxIdx = make([]int, batch*ch)

	// Batch-parallel: out and maxIdx writes are per-b disjoint.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for c := 0; c < ch; c++ {
				best := float32(0)
				bestIdx := -1
				for ih := 0; ih < h; ih++ {
					for iw := 0; iw < w; iw++ {
						idx := ((b*h+ih)*w+iw)*ch + c
						v := x.Data[idx]
						if bestIdx == -1 || v > best {
							best = v
							bestIdx = idx
						}
					}
				}
				out.Data[b*ch+c] = best
				g.maxIdx[b*ch+c] = bestIdx
			}
		}
	})
	return out, nil
}

func (g *GlobalMaxPool2DLayer) Backward(ctx *Context, gradOut *Tensor) (*Tensor, error) {
	gradIn := NewTensor(g.inputShape)
	batch, ch := g.inputShape[0], g.inputShape[3]

	// Batch-parallel: a global max's source index always lies within its
	// own batch element, so gradIn writes from different chunks never
	// collide.
	parallelChunks(batch, func(_, bStart, bEnd int) {
		for b := bStart; b < bEnd; b++ {
			for c := 0; c < ch; c++ {
				i := b*ch + c
				gradIn.Data[g.maxIdx[i]] += gradOut.Data[i]
			}
		}
	})
	return gradIn, nil
}

func (g *GlobalMaxPool2DLayer) Params() []*Param { return nil }
