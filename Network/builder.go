package Network

// Builder provides functional composition utilities for creating networks
// Inspired by Flax's functional approach

// Layer builder functions
type LayerSpec struct {
	Size       int
	Activation ActivationType
}

// L creates a layer specification (shorthand)
func L(size int, activation ActivationType) LayerSpec {
	return LayerSpec{Size: size, Activation: activation}
}

// Stack builds a network from layer specifications
func Stack(specs ...LayerSpec) Network {
	if len(specs) == 0 {
		panic("Cannot stack zero layers")
	}

	seq := NewSequential()
	for _, spec := range specs {
		seq.Dense(spec.Size, spec.Activation)
	}

	return seq.Build()
}

// StackWithLoss builds a network from layer specs with a specified loss
func StackWithLoss(lossType LossType, specs ...LayerSpec) Network {
	if len(specs) == 0 {
		panic("Cannot stack zero layers")
	}

	seq := NewSequential()
	for _, spec := range specs {
		seq.Dense(spec.Size, spec.Activation)
	}

	return seq.WithLoss(lossType).Build()
}

// Common layer patterns

// Input creates an input layer spec
func Input(size int) LayerSpec {
	return L(size, Linear)
}

// Dense creates a dense layer spec with specified activation
func Dense(size int, activation ActivationType) LayerSpec {
	return L(size, activation)
}

// ReLULayer creates a ReLU activated layer spec
func ReLULayer(size int) LayerSpec {
	return L(size, ReLU)
}

// SigmoidLayer creates a Sigmoid activated layer spec
func SigmoidLayer(size int) LayerSpec {
	return L(size, Sigmoid)
}

// TanhLayer creates a Tanh activated layer spec
func TanhLayer(size int) LayerSpec {
	return L(size, Tanh)
}

// LinearLayer creates a Linear activated layer spec
func LinearLayer(size int) LayerSpec {
	return L(size, Linear)
}

// SoftmaxLayer creates a Softmax output layer spec
func SoftmaxLayer(size int) LayerSpec {
	return L(size, Softmax)
}

// Quick builders for common architectures

// Quick creates a network from sizes, using ReLU for hidden layers and specified output activation
func Quick(outputActivation ActivationType, sizes ...int) Network {
	if len(sizes) < 2 {
		panic("Quick requires at least 2 sizes (input and output)")
	}

	seq := NewSequential()
	seq.Input(sizes[0])

	for i := 1; i < len(sizes)-1; i++ {
		seq.Dense(sizes[i], ReLU)
	}

	seq.Dense(sizes[len(sizes)-1], outputActivation)

	return seq.Build()
}

// QuickBinary creates a binary classifier: sizes... -> 1 (Sigmoid) with BCE loss
func QuickBinary(sizes ...int) Network {
	if len(sizes) < 1 {
		panic("QuickBinary requires at least input size")
	}

	seq := NewSequential()
	seq.Input(sizes[0])

	for i := 1; i < len(sizes); i++ {
		seq.Dense(sizes[i], ReLU)
	}

	seq.Dense(1, Sigmoid)

	return seq.WithLoss(BinaryCrossEntropy).Build()
}

// QuickMultiClass creates a multi-class classifier with Softmax and CCE loss
func QuickMultiClass(numClasses int, sizes ...int) Network {
	if len(sizes) < 1 {
		panic("QuickMultiClass requires at least input size")
	}

	seq := NewSequential()
	seq.Input(sizes[0])

	for i := 1; i < len(sizes); i++ {
		seq.Dense(sizes[i], ReLU)
	}

	seq.Dense(numClasses, Softmax)

	return seq.WithLoss(CategoricalCrossEntropy).Build()
}