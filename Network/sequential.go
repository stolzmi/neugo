package Network

// Sequential provides a clean, fluent API for building neural networks
// Inspired by PyTorch's nn.Sequential and Flax's functional approach
type Sequential struct {
	layers []Layer
	loss   LossType
}

// NewSequential creates a new sequential model builder
func NewSequential() *Sequential {
	return &Sequential{
		layers: []Layer{},
		loss:   MSE, // default loss
	}
}

// Add adds a layer to the model
func (s *Sequential) Add(size int, activation ActivationType) *Sequential {
	s.layers = append(s.layers, NewLayerWithActivation(size, activation))
	return s
}

// Dense adds a dense (fully connected) layer - alias for Add for clarity
func (s *Sequential) Dense(size int, activation ActivationType) *Sequential {
	return s.Add(size, activation)
}

// Input adds an input layer (linear activation by default)
func (s *Sequential) Input(size int) *Sequential {
	s.layers = append(s.layers, NewLayerWithActivation(size, Linear))
	return s
}

// WithLoss sets the loss function for the model
func (s *Sequential) WithLoss(lossType LossType) *Sequential {
	s.loss = lossType
	return s
}

// Build constructs the final Network from the sequential specification
func (s *Sequential) Build() Network {
	if len(s.layers) == 0 {
		panic("Cannot build empty network - add layers first")
	}
	return NewNetworkWithLoss(s.layers, s.loss)
}

// Compile is an alias for Build (PyTorch-style)
func (s *Sequential) Compile() Network {
	return s.Build()
}

// Helper constructors for common patterns

// MLP creates a Multi-Layer Perceptron with specified layer sizes
// The last layer uses the specified output activation
func MLP(sizes []int, hiddenActivation, outputActivation ActivationType) Network {
	if len(sizes) < 2 {
		panic("MLP requires at least input and output sizes")
	}

	seq := NewSequential()

	// Input layer
	seq.Input(sizes[0])

	// Hidden layers
	for i := 1; i < len(sizes)-1; i++ {
		seq.Dense(sizes[i], hiddenActivation)
	}

	// Output layer
	seq.Dense(sizes[len(sizes)-1], outputActivation)

	return seq.Build()
}

// BinaryClassifier creates a binary classification network
// Input -> Hidden Layers (ReLU) -> Output (Sigmoid)
func BinaryClassifier(inputSize int, hiddenSizes []int) Network {
	sizes := append([]int{inputSize}, hiddenSizes...)
	sizes = append(sizes, 1)

	return NewSequential().
		Input(inputSize).
		addHiddenLayers(hiddenSizes, ReLU).
		Dense(1, Sigmoid).
		WithLoss(BinaryCrossEntropy).
		Build()
}

// MultiClassClassifier creates a multi-class classification network
// Input -> Hidden Layers (ReLU) -> Output (Softmax)
func MultiClassClassifier(inputSize, numClasses int, hiddenSizes []int) Network {
	return NewSequential().
		Input(inputSize).
		addHiddenLayers(hiddenSizes, ReLU).
		Dense(numClasses, Softmax).
		WithLoss(CategoricalCrossEntropy).
		Build()
}

// Regressor creates a regression network
// Input -> Hidden Layers (ReLU) -> Output (Linear)
func Regressor(inputSize, outputSize int, hiddenSizes []int) Network {
	return NewSequential().
		Input(inputSize).
		addHiddenLayers(hiddenSizes, ReLU).
		Dense(outputSize, Linear).
		WithLoss(MSE).
		Build()
}

// Helper to add multiple hidden layers
func (s *Sequential) addHiddenLayers(sizes []int, activation ActivationType) *Sequential {
	for _, size := range sizes {
		s.Dense(size, activation)
	}
	return s
}