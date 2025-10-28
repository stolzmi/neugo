package Network

type Layer struct {
	neurons []Neuron
}

func NewLayer(size int) Layer {
	neurons := make([]Neuron, size)
	for i := 0; i < size; i++ {
		neurons[i] = *NewNeuron(0.0, 0.0)
	}
	return Layer{neurons: neurons}
}

func (layer Layer) Neurons() []Neuron {
	return layer.neurons
}

func (layer Layer) Size() int {
	return len(layer.neurons)
}
