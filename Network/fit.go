package Network

import "neugo/tensor"

// Fit provides a clean, high-level training interface with sensible defaults
// Inspired by Keras's fit() and PyTorch Lightning

// FitConfig provides a simplified training configuration with smart defaults
type FitConfig struct {
	Epochs       int
	LearningRate float32
	BatchSize    int
	Validation   *ValidationData
	Verbose      bool
	// Optional advanced settings
	L2           float32
	Dropout      float32
	EarlyStopping *EarlyStopping
	Scheduler    LRScheduler
	Callbacks    *CallbackList
}

// ValidationData holds validation dataset
type ValidationData struct {
	X [][]float32
	Y [][]float32
}

// NewFitConfig creates a fit configuration with sensible defaults
func NewFitConfig(epochs int) *FitConfig {
	return &FitConfig{
		Epochs:       epochs,
		LearningRate: 0.01,
		BatchSize:    32,
		Verbose:      true,
		L2:           0.0,
		Dropout:      0.0,
	}
}

// WithLearningRate sets the learning rate
func (c *FitConfig) WithLearningRate(lr float32) *FitConfig {
	c.LearningRate = lr
	return c
}

// WithBatchSize sets the batch size
func (c *FitConfig) WithBatchSize(size int) *FitConfig {
	c.BatchSize = size
	return c
}

// WithValidation sets validation data
func (c *FitConfig) WithValidation(x, y [][]float32) *FitConfig {
	c.Validation = &ValidationData{X: x, Y: y}
	return c
}

// WithL2 sets L2 regularization
func (c *FitConfig) WithL2(lambda float32) *FitConfig {
	c.L2 = lambda
	return c
}

// WithDropout sets dropout rate
func (c *FitConfig) WithDropout(rate float32) *FitConfig {
	c.Dropout = rate
	return c
}

// WithEarlyStopping enables early stopping
func (c *FitConfig) WithEarlyStopping(patience int, minDelta float32) *FitConfig {
	c.EarlyStopping = NewEarlyStopping(patience, minDelta)
	return c
}

// WithScheduler sets a learning rate scheduler
func (c *FitConfig) WithScheduler(scheduler LRScheduler) *FitConfig {
	c.Scheduler = scheduler
	return c
}

// WithCallbacks sets custom callbacks
func (c *FitConfig) WithCallbacks(callbacks *CallbackList) *FitConfig {
	c.Callbacks = callbacks
	return c
}

// Silent disables verbose output
func (c *FitConfig) Silent() *FitConfig {
	c.Verbose = false
	return c
}

// QuickFit provides the simplest possible training interface
// Just specify data, epochs, and optionally learning rate
func (network *Network) QuickFit(x, y [][]float32, epochs int, lr ...float32) *History {
	learningRate := float32(0.01)
	if len(lr) > 0 {
		learningRate = lr[0]
	}

	config := NewFitConfig(epochs).
		WithLearningRate(learningRate).
		WithBatchSize(32)

	return network.FitWithConfig(x, y, config)
}

// FitWithConfig trains the network using a FitConfig
func (network *Network) FitWithConfig(x, y [][]float32, config *FitConfig) *History {
	// Build TrainingConfig from FitConfig
	trainingConfig := &TrainingConfig{
		Epochs:       config.Epochs,
		BatchSize:    config.BatchSize,
		LearningRate: config.LearningRate,
		L2Lambda:     config.L2,
		DropoutRate:  config.Dropout,
		Verbose:      config.Verbose,
		Threshold:    0.5,
	}

	// Add validation data if provided
	if config.Validation != nil {
		trainingConfig.ValidationData = config.Validation.X
		trainingConfig.ValidationLabels = config.Validation.Y
	}

	// Add optional components
	trainingConfig.Scheduler = config.Scheduler
	trainingConfig.EarlyStopping = config.EarlyStopping

	// Setup callbacks
	if config.Callbacks != nil {
		trainingConfig.Callbacks = config.Callbacks
	} else {
		trainingConfig.Callbacks = NewCallbackList()
	}

	// Add progress bar by default if verbose
	if config.Verbose {
		progress := NewProgressBar(config.Epochs, 10, true)
		trainingConfig.Callbacks.Add(progress)
	}

	return network.Train(x, y, trainingConfig)
}

// CNNFitConfig provides simplified training configuration for CNNs
type CNNFitConfig struct {
	*FitConfig
}

// NewCNNFitConfig creates a CNN fit configuration
func NewCNNFitConfig(epochs int) *CNNFitConfig {
	return &CNNFitConfig{
		FitConfig: NewFitConfig(epochs),
	}
}

// QuickFit for CNN - similar to Network.QuickFit but handles tensor inputs
func (cnn *CNN) QuickFit(images []*tensor.Tensor3D, labels [][]float32, epochs int, lr ...float32) {
	learningRate := float32(0.01)
	if len(lr) > 0 {
		learningRate = lr[0]
	}

	println("Training CNN for", epochs, "epochs with LR:", learningRate)

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0.0)

		for i := 0; i < len(images); i++ {
			cnn.ForwardPass(images[i])
			output := []float32{cnn.DenseNetwork.GetOutput()[0].Activation()}
			loss := cnn.Loss.Calculate(output, labels[i])
			epochLoss += loss
			cnn.BackPropagation(images[i], labels[i], learningRate)
		}

		avgLoss := epochLoss / float32(len(images))

		if epoch%10 == 0 || epoch == epochs-1 {
			println("Epoch", epoch+1, "Loss:", avgLoss)
		}
	}
}

// Helper functions for common training patterns

// QuickTrain trains a network with minimal configuration
func QuickTrain(network *Network, x, y [][]float32, epochs int) *History {
	return network.QuickFit(x, y, epochs)
}

// TrainWithValidation trains with validation data and early stopping
func TrainWithValidation(network *Network, trainX, trainY, valX, valY [][]float32, epochs int) *History {
	config := NewFitConfig(epochs).
		WithValidation(valX, valY).
		WithEarlyStopping(10, 0.001)

	return network.FitWithConfig(trainX, trainY, config)
}

// TrainWithScheduler trains with a learning rate scheduler
func TrainWithScheduler(network *Network, x, y [][]float32, epochs int, scheduler LRScheduler) *History {
	config := NewFitConfig(epochs).
		WithScheduler(scheduler)

	return network.FitWithConfig(x, y, config)
}
