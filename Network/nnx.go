package Network

import "math/rand"

// NNX-style API - Flax NNX inspired module system for NeuGo
// Provides a clean, composable, class-based interface similar to Flax NNX

// ============================================================================
// Base Module Interface
// ============================================================================

// Module is the base interface for all neural network modules
type Module interface {
	Forward(x []float32) []float32
	Parameters() int
}

// ============================================================================
// Linear Module
// ============================================================================

// LinearModule represents a fully connected layer
type LinearModule struct {
	InputSize  int
	OutputSize int
	weights    [][]float32
	bias       []float32
	activation ActivationFunction
	layer      Layer
}

// NewLinear creates a new linear layer
func NewLinear(din, dout int, activation ActivationType) *LinearModule {
	layer := NewLayerWithActivation(dout, activation)

	// Initialize weights
	weights := make([][]float32, din)
	scale := float32(1.0) / float32(din)
	for i := 0; i < din; i++ {
		weights[i] = make([]float32, dout)
		for j := 0; j < dout; j++ {
			weights[i][j] = (rand.Float32()*2 - 1) * scale
		}
	}

	bias := make([]float32, dout)
	for i := 0; i < dout; i++ {
		bias[i] = 0.0
	}

	return &LinearModule{
		InputSize:  din,
		OutputSize: dout,
		weights:    weights,
		bias:       bias,
		activation: GetActivationFunction(activation),
		layer:      layer,
	}
}

// Forward performs forward pass
func (l *LinearModule) Forward(x []float32) []float32 {
	output := make([]float32, l.OutputSize)

	// Matrix multiplication + bias
	for j := 0; j < l.OutputSize; j++ {
		sum := l.bias[j]
		for i := 0; i < l.InputSize; i++ {
			sum += x[i] * l.weights[i][j]
		}
		output[j] = l.activation.Apply(sum)
	}

	return output
}

// Parameters returns the number of parameters
func (l *LinearModule) Parameters() int {
	return l.InputSize*l.OutputSize + l.OutputSize
}

// ============================================================================
// Dropout Module
// ============================================================================

// Dropout represents a dropout layer
type Dropout struct {
	Rate      float32
	Training  bool
	lastMask  []bool
}

// NewDropout creates a new dropout layer
func NewDropout(rate float32) *Dropout {
	return &Dropout{
		Rate:     rate,
		Training: true,
	}
}

// Forward performs forward pass with dropout
func (d *Dropout) Forward(x []float32) []float32 {
	if !d.Training || d.Rate == 0 {
		return x
	}

	output := make([]float32, len(x))
	d.lastMask = make([]bool, len(x))
	scale := float32(1.0) / (1.0 - d.Rate)

	for i := 0; i < len(x); i++ {
		if rand.Float32() > d.Rate {
			output[i] = x[i] * scale
			d.lastMask[i] = true
		} else {
			output[i] = 0
			d.lastMask[i] = false
		}
	}

	return output
}

// SetTraining sets training mode
func (d *Dropout) SetTraining(training bool) {
	d.Training = training
}

// Parameters returns 0 (dropout has no parameters)
func (d *Dropout) Parameters() int {
	return 0
}

// ============================================================================
// BatchNorm Module (simplified - running stats not implemented)
// ============================================================================

// BatchNorm represents a batch normalization layer
type BatchNorm struct {
	Size   int
	Gamma  []float32 // scale
	Beta   []float32 // shift
	Eps    float32
}

// NewBatchNorm creates a new batch norm layer
func NewBatchNorm(size int) *BatchNorm {
	gamma := make([]float32, size)
	beta := make([]float32, size)

	for i := 0; i < size; i++ {
		gamma[i] = 1.0
		beta[i] = 0.0
	}

	return &BatchNorm{
		Size:  size,
		Gamma: gamma,
		Beta:  beta,
		Eps:   1e-5,
	}
}

// Forward performs forward pass (simplified, uses input statistics)
func (bn *BatchNorm) Forward(x []float32) []float32 {
	// Calculate mean
	mean := float32(0.0)
	for i := 0; i < len(x); i++ {
		mean += x[i]
	}
	mean /= float32(len(x))

	// Calculate variance
	variance := float32(0.0)
	for i := 0; i < len(x); i++ {
		diff := x[i] - mean
		variance += diff * diff
	}
	variance /= float32(len(x))

	// Normalize
	output := make([]float32, len(x))
	for i := 0; i < len(x); i++ {
		normalized := (x[i] - mean) / float32(sqrt(float64(variance+bn.Eps)))
		output[i] = bn.Gamma[i]*normalized + bn.Beta[i]
	}

	return output
}

// Parameters returns the number of parameters
func (bn *BatchNorm) Parameters() int {
	return bn.Size * 2 // gamma + beta
}

// ============================================================================
// Sequential Module (Container)
// ============================================================================

// SequentialModule represents a sequence of modules
type SequentialModule struct {
	Modules []Module
}

// NewSequentialModule creates a new sequential module
func NewSequentialModule(modules ...Module) *SequentialModule {
	return &SequentialModule{
		Modules: modules,
	}
}

// Forward performs forward pass through all modules
func (s *SequentialModule) Forward(x []float32) []float32 {
	output := x
	for _, module := range s.Modules {
		output = module.Forward(output)
	}
	return output
}

// Parameters returns total parameters
func (s *SequentialModule) Parameters() int {
	total := 0
	for _, module := range s.Modules {
		total += module.Parameters()
	}
	return total
}

// ============================================================================
// MLP Builder (NNX-style)
// ============================================================================

// MLPModule represents a multi-layer perceptron built with modules
type MLPModule struct {
	Linear1 *LinearModule
	Dropout *Dropout
	BN      *BatchNorm
	Linear2 *LinearModule
}

// NewMLPModule creates a new MLP (NNX-style constructor)
func NewMLPModule(din, dmid, dout int, dropoutRate float32) *MLPModule {
	return &MLPModule{
		Linear1: NewLinear(din, dmid, Linear),
		Dropout: NewDropout(dropoutRate),
		BN:      NewBatchNorm(dmid),
		Linear2: NewLinear(dmid, dout, Linear),
	}
}

// Forward performs forward pass (NNX-style __call__)
func (mlp *MLPModule) Forward(x []float32) []float32 {
	// x = gelu(dropout(bn(linear1(x))))
	x = mlp.Linear1.Forward(x)
	x = mlp.BN.Forward(x)
	x = mlp.Dropout.Forward(x)
	x = GELUFunc(x)

	// return linear2(x)
	return mlp.Linear2.Forward(x)
}

// Parameters returns total parameters
func (mlp *MLPModule) Parameters() int {
	return mlp.Linear1.Parameters() + mlp.Linear2.Parameters() + mlp.BN.Parameters()
}

// SetTraining sets training mode for dropout
func (mlp *MLPModule) SetTraining(training bool) {
	mlp.Dropout.SetTraining(training)
}

// ============================================================================
// Activation Functions
// ============================================================================

// GELUFunc applies GELU activation element-wise
func GELUFunc(x []float32) []float32 {
	output := make([]float32, len(x))
	for i := 0; i < len(x); i++ {
		// GELU approximation: x * sigmoid(1.702 * x)
		output[i] = x[i] * sigmoid(1.702 * x[i])
	}
	return output
}

// ReLUFunc applies ReLU element-wise
func ReLUFunc(x []float32) []float32 {
	output := make([]float32, len(x))
	for i := 0; i < len(x); i++ {
		if x[i] > 0 {
			output[i] = x[i]
		} else {
			output[i] = 0
		}
	}
	return output
}

// SigmoidFunc applies Sigmoid element-wise
func SigmoidFunc(x []float32) []float32 {
	output := make([]float32, len(x))
	for i := 0; i < len(x); i++ {
		output[i] = sigmoid(x[i])
	}
	return output
}

// Helper sigmoid function
func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(exp(float64(-x))))
}

func exp(x float64) float64 {
	// Use math.Exp or simple approximation
	result := float64(1.0)
	term := float64(1.0)
	for i := 1; i < 20; i++ {
		term *= x / float64(i)
		result += term
	}
	return result
}

// ============================================================================
// Custom MLP Builder Functions
// ============================================================================

// MLPClassifier creates an MLP for classification
func MLPClassifier(din, dmid, dout int, dropoutRate float32, useNorm bool) *SequentialModule {
	modules := []Module{
		NewLinear(din, dmid, ReLU),
	}

	if useNorm {
		modules = append(modules, NewBatchNorm(dmid))
	}

	if dropoutRate > 0 {
		modules = append(modules, NewDropout(dropoutRate))
	}

	modules = append(modules, NewLinear(dmid, dout, Softmax))

	return NewSequentialModule(modules...)
}

// SimpleMLP creates a simple MLP without normalization
func SimpleMLP(din, dmid, dout int) *SequentialModule {
	return NewSequentialModule(
		NewLinear(din, dmid, ReLU),
		NewLinear(dmid, dout, Sigmoid),
	)
}

// DeepMLP creates a deep MLP with multiple hidden layers
func DeepMLP(din int, hiddenSizes []int, dout int, dropoutRate float32) *SequentialModule {
	modules := []Module{}

	// First layer
	modules = append(modules, NewLinear(din, hiddenSizes[0], ReLU))

	if dropoutRate > 0 {
		modules = append(modules, NewDropout(dropoutRate))
	}

	// Hidden layers
	for i := 0; i < len(hiddenSizes)-1; i++ {
		modules = append(modules, NewLinear(hiddenSizes[i], hiddenSizes[i+1], ReLU))
		modules = append(modules, NewBatchNorm(hiddenSizes[i+1]))

		if dropoutRate > 0 {
			modules = append(modules, NewDropout(dropoutRate))
		}
	}

	// Output layer
	modules = append(modules, NewLinear(hiddenSizes[len(hiddenSizes)-1], dout, Linear))

	return NewSequentialModule(modules...)
}

// ============================================================================
// Helper Functions
// ============================================================================

func sqrt(x float64) float32 {
	// Simple sqrt approximation using Newton's method
	if x == 0 {
		return 0
	}
	z := float64(1.0)
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return float32(z)
}
