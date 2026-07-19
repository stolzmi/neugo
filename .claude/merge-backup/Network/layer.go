package Network

type Layer struct {
	neurons       []Neuron
	activation    ActivationFunction
	hasBias       bool
	regularSize   int // Size without bias node
}

func NewLayer(size int) Layer {
	return NewLayerWithActivation(size, Sigmoid)
}

func NewLayerWithActivation(size int, activationType ActivationType) Layer {
	return newLayerInternal(size, activationType, false)
}

// newLayerInternal creates a layer with optional bias node
// For hidden and output layers, bias will be added during network construction
func newLayerInternal(size int, activationType ActivationType, withBias bool) Layer {
	actualSize := size
	if withBias {
		actualSize = size + 1 // Add one neuron for bias
	}

	neurons := make([]Neuron, actualSize)
	for i := 0; i < actualSize; i++ {
		neurons[i] = *NewNeuron(0.0)
	}

	// If this layer has bias, set the last neuron's activation to 1.0 (constant)
	if withBias {
		neurons[size].setActivation(1.0)
	}

	return Layer{
		neurons:     neurons,
		activation:  GetActivationFunction(activationType),
		hasBias:     withBias,
		regularSize: size,
	}
}

func (layer Layer) Activation() ActivationFunction {
	return layer.activation
}

func (layer Layer) Neurons() []Neuron {
	return layer.neurons
}

func (layer Layer) Size() int {
	return len(layer.neurons)
}

func (layer Layer) RegularSize() int {
	return layer.regularSize
}

func (layer Layer) HasBias() bool {
	return layer.hasBias
}
