package Network

import "math"

type Neuron struct {
	bias       float32
	activation float32
	gradient   float32
}

func NewNeuron(bias float32, activation float32) *Neuron {
	return &Neuron{bias: bias, activation: activation}
}

func (neuron *Neuron) Bias() float32 {
	return neuron.bias
}

func (neuron *Neuron) Activation() float32 {
	return neuron.activation
}

func (neuron *Neuron) setBias(bias float32) {
	neuron.bias = bias
}
func (neuron *Neuron) setActivation(activation float32) {
	neuron.activation = activation
}

func (neuron *Neuron) Gradient() float32 {
	return neuron.gradient
}

func (neuron *Neuron) setGradient(gradient float32) {
	neuron.gradient = gradient
}

func (neuron *Neuron) addGradient(gradient float32) {
	neuron.gradient += gradient
}

// Sigmoid activation function
func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

// Derivative of sigmoid
func sigmoidDerivative(x float32) float32 {
	s := sigmoid(x)
	return s * (1 - s)
}
