package train

import (
	"math"
	"neugo/nn"
)

// Optimizer reads params[i].Grad and mutates params[i].Value.
type Optimizer interface {
	Step(params []*nn.Param)
	SetLR(lr float32)
	GetLR() float32
}

type SGDOptimizer struct {
	LR float32
}

func SGD(lr float32) *SGDOptimizer { return &SGDOptimizer{LR: lr} }

func (o *SGDOptimizer) Step(params []*nn.Param) {
	for _, p := range params {
		for i := range p.Value.Data {
			p.Value.Data[i] -= o.LR * p.Grad.Data[i]
		}
	}
}

func (o *SGDOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *SGDOptimizer) GetLR() float32   { return o.LR }

type MomentumOptimizer struct {
	LR, Beta float32
	velocity map[*nn.Param][]float32
}

func Momentum(lr, beta float32) *MomentumOptimizer {
	return &MomentumOptimizer{LR: lr, Beta: beta, velocity: map[*nn.Param][]float32{}}
}

func (o *MomentumOptimizer) Step(params []*nn.Param) {
	for _, p := range params {
		v, ok := o.velocity[p]
		if !ok {
			v = make([]float32, len(p.Value.Data))
			o.velocity[p] = v
		}
		for i := range p.Value.Data {
			v[i] = o.Beta*v[i] + o.LR*p.Grad.Data[i]
			p.Value.Data[i] -= v[i]
		}
	}
}

func (o *MomentumOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *MomentumOptimizer) GetLR() float32   { return o.LR }

type AdamOptimizer struct {
	LR, Beta1, Beta2, Eps float32
	t                     int
	m, v                  map[*nn.Param][]float32
}

func Adam(lr, beta1, beta2, eps float32) *AdamOptimizer {
	return &AdamOptimizer{
		LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps,
		m: map[*nn.Param][]float32{}, v: map[*nn.Param][]float32{},
	}
}

func (o *AdamOptimizer) Step(params []*nn.Param) {
	o.t++
	b1t := float32(math.Pow(float64(o.Beta1), float64(o.t)))
	b2t := float32(math.Pow(float64(o.Beta2), float64(o.t)))
	for _, p := range params {
		m, ok := o.m[p]
		if !ok {
			m = make([]float32, len(p.Value.Data))
			o.m[p] = m
		}
		v, ok := o.v[p]
		if !ok {
			v = make([]float32, len(p.Value.Data))
			o.v[p] = v
		}
		for i := range p.Value.Data {
			g := p.Grad.Data[i]
			m[i] = o.Beta1*m[i] + (1-o.Beta1)*g
			v[i] = o.Beta2*v[i] + (1-o.Beta2)*g*g
			mHat := m[i] / (1 - b1t)
			vHat := v[i] / (1 - b2t)
			p.Value.Data[i] -= o.LR * mHat / (float32(math.Sqrt(float64(vHat))) + o.Eps)
		}
	}
}

func (o *AdamOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *AdamOptimizer) GetLR() float32   { return o.LR }

type RMSpropOptimizer struct {
	LR, Rho, Eps float32
	sq           map[*nn.Param][]float32
}

func RMSprop(lr, rho, eps float32) *RMSpropOptimizer {
	return &RMSpropOptimizer{LR: lr, Rho: rho, Eps: eps, sq: map[*nn.Param][]float32{}}
}

func (o *RMSpropOptimizer) Step(params []*nn.Param) {
	for _, p := range params {
		sq, ok := o.sq[p]
		if !ok {
			sq = make([]float32, len(p.Value.Data))
			o.sq[p] = sq
		}
		for i := range p.Value.Data {
			g := p.Grad.Data[i]
			sq[i] = o.Rho*sq[i] + (1-o.Rho)*g*g
			p.Value.Data[i] -= o.LR * g / (float32(math.Sqrt(float64(sq[i]))) + o.Eps)
		}
	}
}

func (o *RMSpropOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *RMSpropOptimizer) GetLR() float32   { return o.LR }

// ClipNormOptimizer rescales gradients by their global L2 norm before
// delegating to the wrapped Optimizer, if that norm exceeds maxNorm.
type ClipNormOptimizer struct {
	inner   Optimizer
	maxNorm float32
}

func ClipNorm(inner Optimizer, maxNorm float32) *ClipNormOptimizer {
	return &ClipNormOptimizer{inner: inner, maxNorm: maxNorm}
}

func (o *ClipNormOptimizer) Step(params []*nn.Param) {
	var sumSq float64
	for _, p := range params {
		for _, g := range p.Grad.Data {
			sumSq += float64(g) * float64(g)
		}
	}
	norm := float32(math.Sqrt(sumSq))
	if norm > o.maxNorm {
		scale := o.maxNorm / norm
		for _, p := range params {
			for i := range p.Grad.Data {
				p.Grad.Data[i] *= scale
			}
		}
	}
	o.inner.Step(params)
}

func (o *ClipNormOptimizer) SetLR(lr float32) { o.inner.SetLR(lr) }
func (o *ClipNormOptimizer) GetLR() float32   { return o.inner.GetLR() }
