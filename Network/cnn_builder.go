package Network

// CNNBuilder provides a clean, fluent API for building CNNs
type CNNBuilder struct {
	cnn *CNN
}

// NewCNNBuilder creates a new CNN builder
func NewCNNBuilder(height, width, channels int) *CNNBuilder {
	return &CNNBuilder{
		cnn: NewCNN(height, width, channels, BinaryCrossEntropy),
	}
}

// Conv2D adds a convolutional layer
func (b *CNNBuilder) Conv2D(filters, kernelSize int, activation ActivationType) *CNNBuilder {
	b.cnn.AddConv2D(filters, kernelSize, 1, 1, activation)
	return b
}

// Conv2DWithStride adds a convolutional layer with custom stride
func (b *CNNBuilder) Conv2DWithStride(filters, kernelSize, stride int, activation ActivationType) *CNNBuilder {
	b.cnn.AddConv2D(filters, kernelSize, stride, 1, activation)
	return b
}

// Conv2DFull adds a convolutional layer with full control
func (b *CNNBuilder) Conv2DFull(filters, kernelSize, stride, padding int, activation ActivationType) *CNNBuilder {
	b.cnn.AddConv2D(filters, kernelSize, stride, padding, activation)
	return b
}

// MaxPool adds a max pooling layer
func (b *CNNBuilder) MaxPool(poolSize int) *CNNBuilder {
	b.cnn.AddMaxPool2D(poolSize, poolSize)
	return b
}

// MaxPoolWithStride adds a max pooling layer with custom stride
func (b *CNNBuilder) MaxPoolWithStride(poolSize, stride int) *CNNBuilder {
	b.cnn.AddMaxPool2D(poolSize, stride)
	return b
}

// AvgPool adds an average pooling layer
func (b *CNNBuilder) AvgPool(poolSize int) *CNNBuilder {
	b.cnn.AddAvgPool2D(poolSize, poolSize)
	return b
}

// Flatten adds a flatten layer (automatically called by Dense if needed)
func (b *CNNBuilder) Flatten() *CNNBuilder {
	b.cnn.AddFlatten()
	return b
}

// GetFlattenedSize returns the current flattened size (for informational purposes)
func (b *CNNBuilder) GetFlattenedSize() int {
	return b.cnn.CurrentHeight * b.cnn.CurrentWidth * b.cnn.CurrentChannels
}

// Dense adds dense (fully connected) layers after flattening
// Automatically handles the flattening and creates the dense network
func (b *CNNBuilder) Dense(sizes []int, outputActivation ActivationType) *CNNBuilder {
	if b.cnn.FlattenLayer == nil {
		b.Flatten()
	}

	flattenedSize := b.cnn.CurrentHeight * b.cnn.CurrentWidth * b.cnn.CurrentChannels

	layers := make([]Layer, 0, len(sizes)+1)
	layers = append(layers, NewLayerWithActivation(flattenedSize, Linear))

	for i, size := range sizes {
		if i < len(sizes)-1 {
			layers = append(layers, NewLayerWithActivation(size, ReLU))
		} else {
			layers = append(layers, NewLayerWithActivation(size, outputActivation))
		}
	}

	b.cnn.SetDenseNetwork(layers)
	return b
}

// DenseCustom adds dense layers with custom activation for hidden layers
func (b *CNNBuilder) DenseCustom(sizes []int, hiddenActivation, outputActivation ActivationType) *CNNBuilder {
	if b.cnn.FlattenLayer == nil {
		b.Flatten()
	}

	flattenedSize := b.cnn.CurrentHeight * b.cnn.CurrentWidth * b.cnn.CurrentChannels

	layers := make([]Layer, 0, len(sizes)+1)
	layers = append(layers, NewLayerWithActivation(flattenedSize, Linear))

	for i, size := range sizes {
		if i < len(sizes)-1 {
			layers = append(layers, NewLayerWithActivation(size, hiddenActivation))
		} else {
			layers = append(layers, NewLayerWithActivation(size, outputActivation))
		}
	}

	b.cnn.SetDenseNetwork(layers)
	return b
}

// WithLoss sets the loss function
func (b *CNNBuilder) WithLoss(lossType LossType) *CNNBuilder {
	b.cnn.Loss = GetLossFunction(lossType)
	if b.cnn.DenseNetwork != nil {
		b.cnn.DenseNetwork.loss = b.cnn.Loss
	}
	return b
}

// Build returns the constructed CNN
func (b *CNNBuilder) Build() *CNN {
	return b.cnn
}

// Quick CNN builders for common patterns

// QuickCNN creates a simple CNN for binary classification
// Example: QuickCNN(28, 28, 1, []int{32, 64}, []int{128, 1})
// Creates: Conv(32)->MaxPool->Conv(64)->MaxPool->Flatten->Dense(128)->Dense(1)
func QuickCNN(height, width, channels int, convFilters, denseSizes []int) *CNN {
	builder := NewCNNBuilder(height, width, channels)

	// Add conv layers with pooling after each
	for _, filters := range convFilters {
		builder.Conv2D(filters, 3, ReLU).MaxPool(2)
	}

	// Flatten and add dense layers
	builder.Flatten()
	builder.Dense(denseSizes, Sigmoid)
	builder.WithLoss(BinaryCrossEntropy)

	return builder.Build()
}

// QuickCNNMultiClass creates a simple CNN for multi-class classification
func QuickCNNMultiClass(height, width, channels, numClasses int, convFilters, denseSizes []int) *CNN {
	builder := NewCNNBuilder(height, width, channels)

	// Add conv layers with pooling after each
	for _, filters := range convFilters {
		builder.Conv2D(filters, 3, ReLU).MaxPool(2)
	}

	// Flatten and add dense layers
	builder.Flatten()
	denseSizes = append(denseSizes, numClasses)
	builder.Dense(denseSizes, Softmax)
	builder.WithLoss(CategoricalCrossEntropy)

	return builder.Build()
}

// ImageClassifierCNN creates a CNN following common architecture patterns
// for image classification (similar to simple versions of VGG/ResNet style)
func ImageClassifierCNN(height, width, channels, numClasses int) *CNN {
	return NewCNNBuilder(height, width, channels).
		Conv2D(32, 3, ReLU).
		Conv2D(32, 3, ReLU).
		MaxPool(2).
		Conv2D(64, 3, ReLU).
		Conv2D(64, 3, ReLU).
		MaxPool(2).
		Flatten().
		Dense([]int{128, numClasses}, Softmax).
		WithLoss(CategoricalCrossEntropy).
		Build()
}
