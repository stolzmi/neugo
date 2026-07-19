package Network

import "math"

// ActivationType represents the type of activation function
type ActivationType int

const (
	Sigmoid ActivationType = iota
	ReLU
	Tanh
	Linear
	LeakyReLU
	Softmax
)

// ActivationFunction applies an activation function and its derivative
type ActivationFunction struct {
	Type       ActivationType
	Apply      func(float32) float32
	Derivative func(float32) float32
}

// Sigmoid activation: f(x) = 1 / (1 + e^-x)
func sigmoidFunc(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

func sigmoidDerivFunc(x float32) float32 {
	// For post-activation derivative: f'(a) = a * (1 - a)
	return x * (1 - x)
}

// ReLU activation: f(x) = max(0, x)
func reluFunc(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

func reluDerivFunc(x float32) float32 {
	// For post-activation derivative
	if x > 0 {
		return 1
	}
	return 0
}

// Tanh activation: f(x) = tanh(x)
func tanhFunc(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func tanhDerivFunc(x float32) float32 {
	// For post-activation derivative: f'(a) = 1 - a^2
	return 1 - x*x
}

// Linear activation: f(x) = x
func linearFunc(x float32) float32 {
	return x
}

func linearDerivFunc(x float32) float32 {
	return 1
}

// LeakyReLU activation: f(x) = max(0.01*x, x)
func leakyReluFunc(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}

func leakyReluDerivFunc(x float32) float32 {
	// For post-activation derivative
	if x > 0 {
		return 1
	}
	return 0.01
}

// GetActivationFunction returns the activation function for a given type
func GetActivationFunction(activationType ActivationType) ActivationFunction {
	switch activationType {
	case Sigmoid:
		return ActivationFunction{
			Type:       Sigmoid,
			Apply:      sigmoidFunc,
			Derivative: sigmoidDerivFunc,
		}
	case ReLU:
		return ActivationFunction{
			Type:       ReLU,
			Apply:      reluFunc,
			Derivative: reluDerivFunc,
		}
	case Tanh:
		return ActivationFunction{
			Type:       Tanh,
			Apply:      tanhFunc,
			Derivative: tanhDerivFunc,
		}
	case Linear:
		return ActivationFunction{
			Type:       Linear,
			Apply:      linearFunc,
			Derivative: linearDerivFunc,
		}
	case LeakyReLU:
		return ActivationFunction{
			Type:       LeakyReLU,
			Apply:      leakyReluFunc,
			Derivative: leakyReluDerivFunc,
		}
	default:
		// Default to Sigmoid
		return ActivationFunction{
			Type:       Sigmoid,
			Apply:      sigmoidFunc,
			Derivative: sigmoidDerivFunc,
		}
	}
}

// ApplySoftmax applies softmax to a slice of values (for output layer)
func ApplySoftmax(values []float32) []float32 {
	result := make([]float32, len(values))

	// Find max for numerical stability
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}

	// Calculate exp(x - max) and sum
	sum := float32(0)
	for i, v := range values {
		result[i] = float32(math.Exp(float64(v - max)))
		sum += result[i]
	}

	// Normalize
	for i := range result {
		result[i] /= sum
	}

	return result
}
