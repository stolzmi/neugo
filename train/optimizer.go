package train

import (
	"math"
	"github.com/stolzmi/neugo/nn"
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
	lr := o.LR
	for _, p := range params {
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				val[i] -= lr * g[i]
			}
		})
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
	beta, lr := o.Beta, o.LR
	for _, p := range params {
		v, ok := o.velocity[p]
		if !ok {
			v = make([]float32, len(p.Value.Data))
			o.velocity[p] = v
		}
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				v[i] = beta*v[i] + lr*g[i]
				val[i] -= v[i]
			}
		})
	}
}

func (o *MomentumOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *MomentumOptimizer) GetLR() float32   { return o.LR }

type AdamOptimizer struct {
	LR, Beta1, Beta2, Eps float32
	// WeightDecay applies decoupled weight decay (Loshchilov & Hutter,
	// AdamW): p -= LR*WeightDecay*p, applied directly to the parameter
	// value each step, separately from the gradient-based moment update.
	// Zero (the Adam default) disables it entirely.
	WeightDecay float32
	t           int
	m, v        map[*nn.Param][]float32
}

func Adam(lr, beta1, beta2, eps float32) *AdamOptimizer {
	return &AdamOptimizer{
		LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps,
		m: map[*nn.Param][]float32{}, v: map[*nn.Param][]float32{},
	}
}

// AdamW is Adam with decoupled weight decay set from construction.
func AdamW(lr, beta1, beta2, eps, weightDecay float32) *AdamOptimizer {
	o := Adam(lr, beta1, beta2, eps)
	o.WeightDecay = weightDecay
	return o
}

func (o *AdamOptimizer) Step(params []*nn.Param) {
	o.t++
	b1t := float32(math.Pow(float64(o.Beta1), float64(o.t)))
	b2t := float32(math.Pow(float64(o.Beta2), float64(o.t)))
	beta1, beta2, lr, eps, wd := o.Beta1, o.Beta2, o.LR, o.Eps, o.WeightDecay
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
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				gi := g[i]
				m[i] = beta1*m[i] + (1-beta1)*gi
				v[i] = beta2*v[i] + (1-beta2)*gi*gi
				mHat := m[i] / (1 - b1t)
				vHat := v[i] / (1 - b2t)
				if wd != 0 {
					val[i] -= lr * wd * val[i]
				}
				val[i] -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
			}
		})
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
	rho, lr, eps := o.Rho, o.LR, o.Eps
	for _, p := range params {
		sq, ok := o.sq[p]
		if !ok {
			sq = make([]float32, len(p.Value.Data))
			o.sq[p] = sq
		}
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				gi := g[i]
				sq[i] = rho*sq[i] + (1-rho)*gi*gi
				val[i] -= lr * gi / (float32(math.Sqrt(float64(sq[i]))) + eps)
			}
		})
	}
}

func (o *RMSpropOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *RMSpropOptimizer) GetLR() float32   { return o.LR }

// AdagradOptimizer accumulates the sum of squared gradients per parameter
// (never decayed, unlike RMSprop's exponential moving average) and scales
// the learning rate down accordingly — well suited to sparse gradients,
// at the cost of the effective learning rate shrinking monotonically.
type AdagradOptimizer struct {
	LR, Eps float32
	sq      map[*nn.Param][]float32
}

func Adagrad(lr, eps float32) *AdagradOptimizer {
	return &AdagradOptimizer{LR: lr, Eps: eps, sq: map[*nn.Param][]float32{}}
}

func (o *AdagradOptimizer) Step(params []*nn.Param) {
	lr, eps := o.LR, o.Eps
	for _, p := range params {
		sq, ok := o.sq[p]
		if !ok {
			sq = make([]float32, len(p.Value.Data))
			o.sq[p] = sq
		}
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				gi := g[i]
				sq[i] += gi * gi
				val[i] -= lr * gi / (float32(math.Sqrt(float64(sq[i]))) + eps)
			}
		})
	}
}

func (o *AdagradOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *AdagradOptimizer) GetLR() float32   { return o.LR }

// AdadeltaOptimizer (Zeiler, 2012) tracks a decayed average of squared
// gradients and a decayed average of squared updates, using their ratio in
// place of a hand-tuned learning rate — LR is kept only as an extra
// multiplier on top of that ratio (1.0 recovers the paper's original
// formulation) so Adadelta still satisfies the Optimizer interface's
// SetLR/GetLR.
type AdadeltaOptimizer struct {
	LR, Rho, Eps float32
	accGrad      map[*nn.Param][]float32
	accUpdate    map[*nn.Param][]float32
}

func Adadelta(lr, rho, eps float32) *AdadeltaOptimizer {
	return &AdadeltaOptimizer{
		LR: lr, Rho: rho, Eps: eps,
		accGrad: map[*nn.Param][]float32{}, accUpdate: map[*nn.Param][]float32{},
	}
}

func (o *AdadeltaOptimizer) Step(params []*nn.Param) {
	rho, lr, eps := o.Rho, o.LR, o.Eps
	for _, p := range params {
		eg, ok := o.accGrad[p]
		if !ok {
			eg = make([]float32, len(p.Value.Data))
			o.accGrad[p] = eg
		}
		ex, ok := o.accUpdate[p]
		if !ok {
			ex = make([]float32, len(p.Value.Data))
			o.accUpdate[p] = ex
		}
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				gi := g[i]
				eg[i] = rho*eg[i] + (1-rho)*gi*gi
				delta := float32(math.Sqrt(float64(ex[i]+eps))) / float32(math.Sqrt(float64(eg[i]+eps))) * gi
				ex[i] = rho*ex[i] + (1-rho)*delta*delta
				val[i] -= lr * delta
			}
		})
	}
}

func (o *AdadeltaOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *AdadeltaOptimizer) GetLR() float32   { return o.LR }

// NadamOptimizer is Adam with Nesterov momentum (Dozat, 2016): the bias-
// corrected first moment mixes a Nesterov-style lookahead term (beta1*m,
// scaled by the *next* step's bias correction) with the current gradient
// (scaled by the *current* step's), instead of Adam's plain mHat.
type NadamOptimizer struct {
	LR, Beta1, Beta2, Eps float32
	t                     int
	m, v                  map[*nn.Param][]float32
}

func Nadam(lr, beta1, beta2, eps float32) *NadamOptimizer {
	return &NadamOptimizer{
		LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps,
		m: map[*nn.Param][]float32{}, v: map[*nn.Param][]float32{},
	}
}

func (o *NadamOptimizer) Step(params []*nn.Param) {
	o.t++
	b1t := float32(math.Pow(float64(o.Beta1), float64(o.t)))
	b1t1 := float32(math.Pow(float64(o.Beta1), float64(o.t+1)))
	b2t := float32(math.Pow(float64(o.Beta2), float64(o.t)))
	beta1, beta2, lr, eps := o.Beta1, o.Beta2, o.LR, o.Eps
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
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				gi := g[i]
				m[i] = beta1*m[i] + (1-beta1)*gi
				v[i] = beta2*v[i] + (1-beta2)*gi*gi
				mHat := beta1*m[i]/(1-b1t1) + (1-beta1)*gi/(1-b1t)
				vHat := v[i] / (1 - b2t)
				val[i] -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
			}
		})
	}
}

func (o *NadamOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *NadamOptimizer) GetLR() float32   { return o.LR }

// LionOptimizer (Chen et al., 2023, "Symbolic Discovery of Optimization
// Algorithms") steps by the *sign* of an interpolated momentum term rather
// than its raw magnitude, so every parameter moves by the same size step
// each update (scaled only by LR) — it needs just one momentum buffer, no
// second moment.
type LionOptimizer struct {
	LR, Beta1, Beta2 float32
	m                map[*nn.Param][]float32
}

func Lion(lr, beta1, beta2 float32) *LionOptimizer {
	return &LionOptimizer{LR: lr, Beta1: beta1, Beta2: beta2, m: map[*nn.Param][]float32{}}
}

func (o *LionOptimizer) Step(params []*nn.Param) {
	beta1, beta2, lr := o.Beta1, o.Beta2, o.LR
	for _, p := range params {
		m, ok := o.m[p]
		if !ok {
			m = make([]float32, len(p.Value.Data))
			o.m[p] = m
		}
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				gi := g[i]
				c := beta1*m[i] + (1-beta1)*gi
				update := float32(1)
				if c < 0 {
					update = -1
				} else if c == 0 {
					update = 0
				}
				val[i] -= lr * update
				m[i] = beta2*m[i] + (1-beta2)*gi
			}
		})
	}
}

func (o *LionOptimizer) SetLR(lr float32) { o.LR = lr }
func (o *LionOptimizer) GetLR() float32   { return o.LR }

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
		sumSq += sumSquares(p.Grad.Data)
	}
	norm := float32(math.Sqrt(sumSq))
	if norm > o.maxNorm {
		scale := o.maxNorm / norm
		for _, p := range params {
			g := p.Grad.Data
			parallelFor(len(g), func(start, end int) {
				for i := start; i < end; i++ {
					g[i] *= scale
				}
			})
		}
	}
	o.inner.Step(params)
}

func (o *ClipNormOptimizer) SetLR(lr float32) { o.inner.SetLR(lr) }
func (o *ClipNormOptimizer) GetLR() float32   { return o.inner.GetLR() }

// L1RegOptimizer adds an L1 penalty's subgradient (Lambda*sign(w)) to each
// parameter's Grad before delegating Step to the wrapped Optimizer — same
// decorator shape as ClipNormOptimizer (composable with it: e.g.
// L1Reg(ClipNorm(Adam(...), 1.0), 1e-4)).
type L1RegOptimizer struct {
	inner  Optimizer
	Lambda float32
}

func L1Reg(inner Optimizer, lambda float32) *L1RegOptimizer {
	return &L1RegOptimizer{inner: inner, Lambda: lambda}
}

func (o *L1RegOptimizer) Step(params []*nn.Param) {
	lambda := o.Lambda
	for _, p := range params {
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				switch {
				case val[i] > 0:
					g[i] += lambda
				case val[i] < 0:
					g[i] -= lambda
				}
			}
		})
	}
	o.inner.Step(params)
}

func (o *L1RegOptimizer) SetLR(lr float32) { o.inner.SetLR(lr) }
func (o *L1RegOptimizer) GetLR() float32   { return o.inner.GetLR() }

// L2RegOptimizer adds an L2 penalty's gradient (Lambda*w) to each
// parameter's Grad before delegating Step — unlike AdamW's WeightDecay
// (which shrinks the parameter value directly, decoupled from the
// gradient-based moment estimates), this is the classical L2 regularizer
// applied through the gradient itself, so it works with any Optimizer.
type L2RegOptimizer struct {
	inner  Optimizer
	Lambda float32
}

func L2Reg(inner Optimizer, lambda float32) *L2RegOptimizer {
	return &L2RegOptimizer{inner: inner, Lambda: lambda}
}

func (o *L2RegOptimizer) Step(params []*nn.Param) {
	lambda := o.Lambda
	for _, p := range params {
		val, g := p.Value.Data, p.Grad.Data
		parallelFor(len(val), func(start, end int) {
			for i := start; i < end; i++ {
				g[i] += lambda * val[i]
			}
		})
	}
	o.inner.Step(params)
}

func (o *L2RegOptimizer) SetLR(lr float32) { o.inner.SetLR(lr) }
func (o *L2RegOptimizer) GetLR() float32   { return o.inner.GetLR() }
