package Network

type Neuron struct {
	activation float32
	gradient   float32
}

func NewNeuron(activation float32) *Neuron {
	return &Neuron{activation: activation}
}

func (neuron *Neuron) Activation() float32 {
	return neuron.activation
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
