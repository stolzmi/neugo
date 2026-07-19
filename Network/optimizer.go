package Network

import "math"

// OptimizerType represents the type of optimizer
type OptimizerType int

const (
	SGD OptimizerType = iota
	Adam
	RMSprop
	Momentum
)

// Optimizer interface for different optimization algorithms
type Optimizer interface {
	Update(weights *[][][]float32, gradients [][][]float32, layerIndex, i, j int)
	GetLearningRate() float32
	SetLearningRate(lr float32)
}

// SGDOptimizer implements standard Stochastic Gradient Descent
type SGDOptimizer struct {
	LearningRate float32
}

func NewSGD(learningRate float32) *SGDOptimizer {
	return &SGDOptimizer{LearningRate: learningRate}
}

func (opt *SGDOptimizer) Update(weights *[][][]float32, gradients [][][]float32, layerIndex, i, j int) {
	(*weights)[layerIndex][i][j] -= opt.LearningRate * gradients[layerIndex][i][j]
}

func (opt *SGDOptimizer) GetLearningRate() float32 {
	return opt.LearningRate
}

func (opt *SGDOptimizer) SetLearningRate(lr float32) {
	opt.LearningRate = lr
}

// AdamOptimizer implements Adam optimization algorithm
type AdamOptimizer struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	TimeStep     int

	// First moment estimates (momentum)
	M [][][]float32
	// Second moment estimates (RMSprop)
	V [][][]float32
}

func NewAdam(learningRate, beta1, beta2, epsilon float32) *AdamOptimizer {
	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		TimeStep:     0,
	}
}

func (opt *AdamOptimizer) Initialize(network *Network) {
	// Initialize moment estimates
	opt.M = make([][][]float32, len(network.weights))
	opt.V = make([][][]float32, len(network.weights))

	for i := range network.weights {
		opt.M[i] = make([][]float32, len(network.weights[i]))
		opt.V[i] = make([][]float32, len(network.weights[i]))

		for j := range network.weights[i] {
			opt.M[i][j] = make([]float32, len(network.weights[i][j]))
			opt.V[i][j] = make([]float32, len(network.weights[i][j]))
		}
	}
}

func (opt *AdamOptimizer) Update(weights *[][][]float32, gradients [][][]float32, layerIndex, i, j int) {
	// Increment time step on first call
	if layerIndex == 0 && i == 0 && j == 0 {
		opt.TimeStep++
	}

	gradient := gradients[layerIndex][i][j]

	// Update biased first moment estimate
	opt.M[layerIndex][i][j] = opt.Beta1*opt.M[layerIndex][i][j] + (1-opt.Beta1)*gradient

	// Update biased second raw moment estimate
	opt.V[layerIndex][i][j] = opt.Beta2*opt.V[layerIndex][i][j] + (1-opt.Beta2)*gradient*gradient

	// Compute bias-corrected first moment estimate
	mHat := opt.M[layerIndex][i][j] / (1 - float32(math.Pow(float64(opt.Beta1), float64(opt.TimeStep))))

	// Compute bias-corrected second raw moment estimate
	vHat := opt.V[layerIndex][i][j] / (1 - float32(math.Pow(float64(opt.Beta2), float64(opt.TimeStep))))

	// Update weights
	(*weights)[layerIndex][i][j] -= opt.LearningRate * mHat / (float32(math.Sqrt(float64(vHat))) + opt.Epsilon)
}

func (opt *AdamOptimizer) GetLearningRate() float32 {
	return opt.LearningRate
}

func (opt *AdamOptimizer) SetLearningRate(lr float32) {
	opt.LearningRate = lr
}

// MomentumOptimizer implements SGD with momentum
type MomentumOptimizer struct {
	LearningRate float32
	Momentum     float32
	Velocity     [][][]float32
}

func NewMomentum(learningRate, momentum float32) *MomentumOptimizer {
	return &MomentumOptimizer{
		LearningRate: learningRate,
		Momentum:     momentum,
	}
}

func (opt *MomentumOptimizer) Initialize(network *Network) {
	opt.Velocity = make([][][]float32, len(network.weights))

	for i := range network.weights {
		opt.Velocity[i] = make([][]float32, len(network.weights[i]))
		for j := range network.weights[i] {
			opt.Velocity[i][j] = make([]float32, len(network.weights[i][j]))
		}
	}
}

func (opt *MomentumOptimizer) Update(weights *[][][]float32, gradients [][][]float32, layerIndex, i, j int) {
	gradient := gradients[layerIndex][i][j]

	// Update velocity
	opt.Velocity[layerIndex][i][j] = opt.Momentum*opt.Velocity[layerIndex][i][j] + opt.LearningRate*gradient

	// Update weights
	(*weights)[layerIndex][i][j] -= opt.Velocity[layerIndex][i][j]
}

func (opt *MomentumOptimizer) GetLearningRate() float32 {
	return opt.LearningRate
}

func (opt *MomentumOptimizer) SetLearningRate(lr float32) {
	opt.LearningRate = lr
}

// Add optimizer to network
func (network *Network) SetOptimizer(optimizer Optimizer) {
	// Initialize optimizer-specific state
	switch opt := optimizer.(type) {
	case *AdamOptimizer:
		opt.Initialize(network)
	case *MomentumOptimizer:
		opt.Initialize(network)
	}
}
