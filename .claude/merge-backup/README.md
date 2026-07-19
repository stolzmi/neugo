# NeuGo - Neural Network Library in Go

A simple, flexible neural network implementation in Go with support for multiple activation functions and loss functions.

## Features

### Core Neural Network
- 🧠 **Flexible Architecture**: Create networks with any number of layers and neurons
- ⚡ **Multiple Activation Functions**: Sigmoid, ReLU, Tanh, Linear, LeakyReLU, Softmax
- 📊 **Multiple Loss Functions**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy, MAE
- 🎲 **Smart Weight Initialization**: Xavier/Glorot initialization for better convergence
- 🔄 **Backpropagation**: Automatic gradient computation and weight updates

### Training & Optimization (Phase 1 & 2)
- 🎯 **Batch Training**: Mini-batch gradient descent for efficient learning
- 🚀 **Advanced Optimizers**: Adam, Momentum, SGD
- 📉 **Learning Rate Scheduling**: Step Decay, Exponential Decay, Cosine Annealing, Warmup, Reduce on Plateau
- 🛡️ **Regularization**: L2, L1, Dropout to prevent overfitting
- ⏹️ **Early Stopping**: Automatic training termination based on validation metrics
- 💾 **Model Serialization**: Save and load trained models as JSON
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

### Data Utilities (Phase 3)
- 📂 **CSV Loading**: Flexible CSV loader with custom delimiters and configurations
- 📏 **Normalization**: Z-score and Min-Max normalization
- ✂️ **Data Splitting**: Train/validation/test splits with shuffling
- ⚖️ **Class Balancing**: Oversampling and undersampling for imbalanced datasets
- 📊 **Distribution Analysis**: Automatic class distribution analysis
- 🔀 **Data Shuffling**: Reproducible shuffling with seed control

### Training Enhancements (Phase 4)
- 🔔 **Callbacks System**: Extensible callback interface for training events
- 📈 **History Tracking**: Automatic tracking of loss, accuracy, F1 over epochs
- 💾 **Model Checkpointing**: Auto-save best models during training
- 📊 **Progress Monitoring**: Built-in progress bars and logging
- 🔄 **Cross-Validation**: K-Fold and Stratified K-Fold with statistics
- ⚙️ **Training Config**: Unified configuration object for all training parameters
- 🎨 **Custom Callbacks**: User-defined callbacks for specialized behavior

## Project Structure

```
neugo/
├── Network/              # Core neural network package
│   ├── activation.go     # Activation functions (Sigmoid, ReLU, Tanh, etc.)
│   ├── layer.go          # Layer implementation
│   ├── loss.go           # Loss functions (MSE, BCE, CCE, MAE)
│   ├── network.go        # Network implementation
│   ├── neuron.go         # Neuron implementation
│   ├── batch.go          # Batch training (Phase 1)
│   ├── serialization.go  # Model save/load (Phase 1)
│   ├── metrics.go        # Evaluation metrics (Phase 1)
│   ├── optimizer.go      # Optimizers: Adam, Momentum, SGD (Phase 2)
│   ├── regularization.go # L2, L1, Dropout (Phase 2)
│   ├── scheduler.go      # Learning rate schedulers (Phase 2)
│   ├── callbacks.go      # Callback system (Phase 4)
│   ├── training.go       # Enhanced training with config (Phase 4)
│   ├── crossval.go       # Cross-validation utilities (Phase 4)
│   └── doc.go            # Package documentation
├── data/                 # Data utilities package (Phase 3)
│   ├── loader.go         # CSV loading with flexible config
│   ├── preprocessing.go  # Normalization, splitting, shuffling
│   ├── balancing.go      # Class balancing (over/undersampling)
│   └── doc.go            # Package documentation
├── examples/             # Example applications
│   ├── train.go          # Basic XOR training
│   ├── showcase.go       # Full Phase 1 & 2 feature demo
│   ├── wine_quality.go   # Wine quality classifier (original)
│   ├── wine_quality_clean.go  # Using data utilities
│   └── fraud_detection.go     # Fraud detection (imbalanced data)
├── dataset/              # Example datasets
│   ├── wine_quality/     # Wine quality dataset
│   └── fraud/            # Fraud detection dataset
├── FEATURE_GUIDE.md      # Complete API reference
└── README.md             # This file
```

## Quick Start

### 🚀 NNX-Style API (NEW - Flax NNX-Inspired)

**Module-based approach - most similar to Flax NNX and PyTorch:**

```go
package main

import "neugo/Network"

// Define custom module (like Flax NNX)
type MLP struct {
    Linear1 *Network.LinearModule
    Dropout *Network.Dropout
    BN      *Network.BatchNorm
    Linear2 *Network.LinearModule
}

func NewMLP(din, dmid, dout int, dropoutRate float32) *MLP {
    return &MLP{
        Linear1: Network.NewLinear(din, dmid, Network.Linear),
        Dropout: Network.NewDropout(dropoutRate),
        BN:      Network.NewBatchNorm(dmid),
        Linear2: Network.NewLinear(dmid, dout, Network.Linear),
    }
}

func (mlp *MLP) Forward(x []float32) []float32 {
    x = mlp.Linear1.Forward(x)
    x = mlp.BN.Forward(x)
    x = mlp.Dropout.Forward(x)
    x = Network.GELUFunc(x)
    return mlp.Linear2.Forward(x)
}

func main() {
    model := NewMLP(10, 64, 1, 0.1)
    output := model.Forward(input)
}
```

**Or use built-in modules:**

```go
// Sequential composition
model := Network.NewSequentialModule(
    Network.NewLinear(10, 64, Network.ReLU),
    Network.NewBatchNorm(64),
    Network.NewDropout(0.2),
    Network.NewLinear(64, 1, Network.Sigmoid),
)

output := model.Forward(input)
```

**See [NNX_API_GUIDE.md](NNX_API_GUIDE.md) for complete NNX-style documentation!**

### 🎨 Clean API (PyTorch/Flax-Inspired)

```go
package main

import "neugo/Network"

func main() {
    // Build model (4 different styles available!)
    model := Network.QuickBinary(2, 8, 4)  // 2 inputs → 8 → 4 → 1 output

    // Or use Sequential API (PyTorch-like)
    model = Network.NewSequential().
        Input(2).
        Dense(8, Network.ReLU).
        Dense(4, Network.ReLU).
        Dense(1, Network.Sigmoid).
        WithLoss(Network.BinaryCrossEntropy).
        Build()

    // Train with one line
    history := model.QuickFit(trainX, trainY, 1000, 0.1)

    // Evaluate
    metrics := model.Evaluate(testX, testY, 0.5)
    println("Accuracy:", metrics.Accuracy)
}
```

**See [CLEAN_API_GUIDE.md](CLEAN_API_GUIDE.md) for complete documentation of the new API!**

### Classic API (Still Supported)

```go
package main

import (
    "neugo/Network"
)

func main() {
    // Create layers
    inputLayer := Network.NewLayer(2)  // Default Sigmoid activation
    hiddenLayer := Network.NewLayerWithActivation(4, Network.ReLU)
    outputLayer := Network.NewLayerWithActivation(1, Network.Sigmoid)

    // Create network
    layers := []Network.Layer{inputLayer, hiddenLayer, outputLayer}
    network := Network.NewNetwork(layers)  // Default MSE loss

    // Train
    input := []float32{1.0, 0.5}
    labels := []float32{1.0}
    learningRate := float32(0.1)

    network.ForwardPass(input)
    loss := network.CalculateLoss(labels)
    network.BackPropagation(labels, learningRate)
}
```

## Examples

### Clean API Demo (NEW!)

```bash
go run examples/clean_api_demo.go
```

Showcases all 4 model building styles and the new simplified training API.

### XOR Problem

```bash
go run examples/train.go
```

### Test Different Activations

```bash
go run examples/test_activations.go
```

### Test Different Loss Functions

```bash
go run examples/test_losses.go
```

### Wine Quality Classifier (with Data Utilities)

```bash
go run examples/wine_quality_clean.go
```

### Comprehensive Feature Showcase

```bash
go run examples/showcase.go
```

## Complete Example with Data Utilities

```go
package main

import (
	"fmt"
	"neugo/Network"
	"neugo/data"
)

func main() {
	// Load dataset
	dataset, _ := data.QuickLoadBinaryCSV("wine.csv", ';', 6.0)
	fmt.Printf("Loaded %d samples with %d features\n",
		dataset.NumSamples, dataset.NumFeatures)

	// Analyze class distribution
	dist := data.AnalyzeClassDistribution(dataset.Labels, 0.5)
	fmt.Printf("Class 0: %.2f%%, Class 1: %.2f%%\n",
		dist.Percentages[0], dist.Percentages[1])

	// Normalize features
	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	// Split into train/val/test
	split := data.SplitData(normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7,
		ValRatio:   0.15,
		TestRatio:  0.15,
		Shuffle:    true,
		Seed:       42,
	})

	// Balance training data
	trainX, trainY := data.OversampleMinorityClass(
		split.TrainX, split.TrainY,
		data.OversampleConfig{
			TargetRatio: 0.4,  // 40% minority class
			Strategy:    "duplicate",
			Seed:        42,
		},
	)

	// Build network
	layers := []Network.Layer{
		Network.NewLayerWithActivation(dataset.NumFeatures, Network.Linear),
		Network.NewLayerWithActivation(16, Network.ReLU),
		Network.NewLayerWithActivation(8, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)

	// Setup training
	scheduler := Network.NewCosineAnnealing(0.1, 0.001, 100)
	earlyStopping := Network.NewEarlyStopping(20, 0.0001)

	// Train with all features
	for epoch := 0; epoch < 100; epoch++ {
		lr := scheduler.GetLearningRate(epoch)

		// Train on batches
		for i := 0; i < len(trainX); i += 32 {
			end := min(i+32, len(trainX))
			loss := network.TrainBatchWithRegularization(
				trainX[i:end], trainY[i:end],
				lr, 0.001, 0.2,
			)
		}

		// Validate
		valMetrics := network.Evaluate(split.ValX, split.ValY, 0.5)
		earlyStopping.Update(valMetrics.Loss, &network)

		if earlyStopping.ShouldStop {
			earlyStopping.RestoreBestWeights(&network)
			break
		}

		scheduler.Step()
	}

	// Evaluate on test set
	testMetrics := network.Evaluate(split.TestX, split.TestY, 0.5)
	fmt.Printf("Test Accuracy: %.2f%%\n", testMetrics.Accuracy)
	fmt.Printf("Test F1 Score: %.4f\n", testMetrics.F1Score)

	// Save model
	network.SaveToFile("model.json")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

## API Reference

### Layer

- `NewLayer(size int) Layer` - Create layer with Sigmoid activation
- `NewLayerWithActivation(size int, activationType ActivationType) Layer` - Create layer with custom activation

### Network

- `NewNetwork(layers []Layer) Network` - Create network with MSE loss
- `NewNetworkWithLoss(layers []Layer, lossType LossType) Network` - Create network with custom loss
- `network.ForwardPass(input []float32)` - Perform forward pass
- `network.BackPropagation(labels []float32, learningRate float32)` - Perform backpropagation
- `network.CalculateLoss(labels []float32) float32` - Calculate loss

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Data Utilities API

### Loading Data

```go
// Quick load for binary classification
dataset, err := data.QuickLoadBinaryCSV("data.csv", ',', 0.5)

// Quick load for regression
dataset, err := data.QuickLoadRegressionCSV("data.csv", ',')

// Custom configuration
config := data.CSVConfig{
	Delimiter:       ';',
	HasHeader:       true,
	LabelColumn:     -1, // -1 = last column
	LabelType:       "binary",
	BinaryThreshold: 6.0,
}
dataset, err := data.LoadCSV("data.csv", config)
```

### Preprocessing

```go
// Calculate statistics
stats := data.CalculateStats(features)

// Z-score normalization (mean=0, std=1)
normalized := data.NormalizeZScore(features, stats)

// Min-max normalization (scale to [0, 1])
normalized := data.NormalizeMinMax(features, stats)

// Shuffle data
shuffledX, shuffledY := data.ShuffleData(features, labels, 42)

// Split data
split := data.SplitData(features, labels, data.SplitConfig{
	TrainRatio: 0.7,
	ValRatio:   0.15,
	TestRatio:  0.15,
	Shuffle:    true,
	Seed:       42,
})
```

### Class Balancing

```go
// Analyze distribution
dist := data.AnalyzeClassDistribution(labels, 0.5)
fmt.Printf("Balanced: %v\n", dist.IsBalanced)

// Oversample minority class
balancedX, balancedY := data.OversampleMinorityClass(
	features, labels,
	data.OversampleConfig{
		TargetRatio: 0.4,
		Strategy:    "duplicate", // or "random"
		Seed:        42,
	},
)

// Undersample majority class
balancedX, balancedY := data.UndersampleMajorityClass(
	features, labels,
	data.UndersampleConfig{
		TargetRatio: 0.4,
		Strategy:    "random", // or "systematic"
		Seed:        42,
	},
)

// Automatic balancing
balancedX, balancedY := data.BalanceDataset(
	features, labels,
	0.4,   // target ratio
	true,  // prefer oversample over undersample
)
```

## Completed Features

### Phase 1: Core Functionality ✅
- [x] Batch training support
- [x] Model serialization (save/load)
- [x] Validation & Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- [x] Early stopping

### Phase 2: Advanced Training ✅
- [x] Optimizers (Adam, Momentum, SGD)
- [x] Regularization (L2, L1, Dropout)
- [x] Learning Rate Scheduling (5 strategies)

### Phase 3: Data Utilities ✅
- [x] CSV loading with flexible configuration
- [x] Normalization (Z-score, Min-Max)
- [x] Data splitting with shuffling
- [x] Class balancing (over/undersampling)
- [x] Distribution analysis

### Phase 4: Training Enhancements ✅
- [x] Callbacks system (extensible interface)
- [x] History tracking (loss, metrics over time)
- [x] Model checkpointing (auto-save best models)
- [x] Progress monitoring (progress bars, logging)
- [x] Cross-validation (K-Fold, Stratified K-Fold)
- [x] Training configuration (unified config object)
- [x] Custom callbacks (user-defined behavior)

## Phase 4 API Examples

### Callbacks

```go
// Create callbacks
history := Network.NewHistory()
checkpoint := Network.NewModelCheckpoint(
    "best_model.json",
    "f1",    // Monitor F1 score
    "max",   // Maximize
    true,    // Save best only
    true,    // Verbose
)
progress := Network.NewProgressBar(100, 10, true)

// Custom callback
custom := Network.NewCustomCallback()
custom.OnEpochEndFunc = func(epoch int, net *Network.Network, metrics *Network.Metrics) {
    if epoch == 50 {
        fmt.Println("Halfway there!")
    }
}

callbacks := Network.NewCallbackList(history, checkpoint, progress, custom)
```

### Training with Configuration

```go
config := &Network.TrainingConfig{
    Epochs:           100,
    BatchSize:        32,
    LearningRate:     0.1,
    L2Lambda:         0.001,
    DropoutRate:      0.2,
    Scheduler:        Network.NewCosineAnnealing(0.1, 0.001, 100),
    EarlyStopping:    Network.NewEarlyStopping(15, 0.0001),
    Callbacks:        callbacks,
    ValidationData:   valX,
    ValidationLabels: valY,
    Threshold:        0.5,
    Verbose:          true,
}

history := network.Train(trainX, trainY, config)

// Access training history
fmt.Printf("Duration: %v\n", history.Duration())
fmt.Printf("Final loss: %.4f\n", history.TrainLoss[len(history.TrainLoss)-1])
```

### Cross-Validation

```go
// Create network factory
createNetwork := func() Network.Network {
    layers := []Network.Layer{
        Network.NewLayerWithActivation(11, Network.Linear),
        Network.NewLayerWithActivation(16, Network.ReLU),
        Network.NewLayerWithActivation(1, Network.Sigmoid),
    }
    return Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)
}

// K-Fold CV
result := Network.CrossValidate(
    createNetwork,
    features, labels,
    5, // K=5 folds
    config,
    true, // Verbose
)

fmt.Printf("Mean Accuracy: %.2f%% ± %.2f%%\n",
    result.MeanAccuracy, result.StdAccuracy)

// Stratified K-Fold (preserves class distribution)
stratifiedResult := Network.CrossValidateStratified(
    createNetwork,
    features, labels,
    5,
    config,
    true,
)
```

## Future Improvements

- [ ] Convolutional layers for image processing
- [ ] Recurrent layers (LSTM, GRU) for sequences
- [ ] GPU acceleration
- [ ] Hyperparameter tuning utilities
- [ ] Model visualization tools
- [ ] More advanced balancing (SMOTE)

## License

MIT License - feel free to use this project for learning and development.
