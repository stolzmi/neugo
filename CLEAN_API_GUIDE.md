# NeuGo Clean API Guide

A modern, intuitive API for building neural networks in Go, inspired by PyTorch and Flax.

## Table of Contents
- [Philosophy](#philosophy)
- [Quick Start](#quick-start)
- [Model Building Styles](#model-building-styles)
- [Training API](#training-api)
- [CNN Builder](#cnn-builder)
- [Complete Examples](#complete-examples)

## Philosophy

The clean API is designed around three principles:

1. **Readability**: Code should read like the architecture diagram
2. **Chainability**: Fluent interfaces for building models step by step
3. **Sensible Defaults**: Common patterns should be simple, advanced features still accessible

## Quick Start

### 30-Second Example

```go
import "neugo/Network"

// Build a model
model := Network.QuickBinary(2, 8, 4) // 2 inputs → 8 → 4 → 1 output

// Train it
history := model.QuickFit(trainX, trainY, 1000, 0.1)

// Evaluate
metrics := model.Evaluate(testX, testY, 0.5)
```

That's it! Three lines for a complete workflow.

## Model Building Styles

### Style 1: Sequential API (PyTorch-like)

**Best for**: Most use cases, maximum flexibility

```go
model := Network.NewSequential().
    Input(784).                           // Input layer
    Dense(128, Network.ReLU).             // Hidden layer
    Dense(64, Network.ReLU).              // Hidden layer
    Dense(10, Network.Softmax).           // Output layer
    WithLoss(Network.CategoricalCrossEntropy).
    Build()
```

**Features:**
- Chainable method calls
- Clear, linear structure
- Easy to modify

### Style 2: Functional API (Flax-like)

**Best for**: Concise definitions, functional programming fans

```go
model := Network.Stack(
    Network.Input(784),
    Network.ReLULayer(128),
    Network.ReLULayer(64),
    Network.SoftmaxLayer(10),
)
```

Or with explicit loss:

```go
model := Network.StackWithLoss(
    Network.CategoricalCrossEntropy,
    Network.Input(784),
    Network.ReLULayer(128),
    Network.SoftmaxLayer(10),
)
```

**Features:**
- Most concise
- Pure function composition
- Immutable style

### Style 3: Quick Builders (Keras-like)

**Best for**: Rapid prototyping, common patterns

```go
// Binary classification
model := Network.QuickBinary(inputSize, hiddenSize1, hiddenSize2, ...)

// Multi-class classification
model := Network.QuickMultiClass(numClasses, inputSize, hiddenSize1, ...)

// Regression
model := Network.Quick(Network.Linear, inputSize, hiddenSize1, ..., outputSize)
```

**Features:**
- Minimal code
- Automatic architecture selection
- Perfect for experimentation

### Style 4: High-Level Constructors

**Best for**: Domain-specific applications

```go
// Binary classifier with custom hidden layers
model := Network.BinaryClassifier(inputSize, []int{128, 64})

// Multi-class classifier
model := Network.MultiClassClassifier(inputSize, numClasses, []int{128, 64})

// Regressor
model := Network.Regressor(inputSize, outputSize, []int{128, 64})
```

**Features:**
- Task-oriented
- Encapsulates best practices
- Self-documenting code

## Training API

### QuickFit - Simplest Training

```go
// Minimal arguments
history := model.QuickFit(trainX, trainY, epochs)

// With custom learning rate
history := model.QuickFit(trainX, trainY, epochs, learningRate)
```

### FitConfig - Advanced Training

For full control with readable configuration:

```go
config := Network.NewFitConfig(epochs).
    WithLearningRate(0.01).
    WithBatchSize(32).
    WithValidation(valX, valY).
    WithL2(0.001).
    WithDropout(0.2).
    WithEarlyStopping(patience, minDelta).
    WithScheduler(Network.NewCosineAnnealing(0.1, 0.001, epochs))

history := model.FitWithConfig(trainX, trainY, config)
```

**All available options:**

```go
config := Network.NewFitConfig(epochs).
    WithLearningRate(0.01).              // Learning rate
    WithBatchSize(32).                    // Batch size
    WithValidation(valX, valY).           // Validation data
    WithL2(0.001).                        // L2 regularization
    WithDropout(0.2).                     // Dropout rate
    WithEarlyStopping(10, 0.001).        // Early stopping
    WithScheduler(scheduler).             // LR scheduler
    WithCallbacks(callbacks).             // Custom callbacks
    Silent()                              // Disable verbose output
```

### Helper Training Functions

```go
// Quick training
history := Network.QuickTrain(model, trainX, trainY, epochs)

// With validation and early stopping
history := Network.TrainWithValidation(
    model, trainX, trainY, valX, valY, epochs,
)

// With scheduler
scheduler := Network.NewCosineAnnealing(0.1, 0.001, epochs)
history := Network.TrainWithScheduler(model, trainX, trainY, epochs, scheduler)
```

## CNN Builder

### Old Way (Verbose)

```go
cnn := Network.NewCNN(28, 28, 1, Network.BinaryCrossEntropy)
cnn.AddConv2D(32, 3, 1, 1, Network.ReLU)
cnn.AddMaxPool2D(2, 2)
cnn.AddConv2D(64, 3, 1, 1, Network.ReLU)
cnn.AddMaxPool2D(2, 2)
cnn.AddFlatten()
flattenedSize := 7 * 7 * 64  // Manual calculation!
layers := []Network.Layer{
    Network.NewLayerWithActivation(flattenedSize, Network.Linear),
    Network.NewLayerWithActivation(128, Network.ReLU),
    Network.NewLayerWithActivation(1, Network.Sigmoid),
}
cnn.SetDenseNetwork(layers)
```

### New Way (Clean)

```go
cnn := Network.NewCNNBuilder(28, 28, 1).
    Conv2D(32, 3, Network.ReLU).
    MaxPool(2).
    Conv2D(64, 3, Network.ReLU).
    MaxPool(2).
    Dense([]int{128, 1}, Network.Sigmoid).
    WithLoss(Network.BinaryCrossEntropy).
    Build()
```

**No manual size calculation needed!** The builder tracks dimensions automatically.

### Quick CNN Builders

```go
// Binary classification CNN
cnn := Network.QuickCNN(
    28, 28, 1,                    // height, width, channels
    []int{32, 64},                // conv filters
    []int{128, 1},                // dense sizes
)

// Multi-class classification CNN
cnn := Network.QuickCNNMultiClass(
    28, 28, 1, 10,                // height, width, channels, classes
    []int{32, 64},                // conv filters
    []int{128},                   // dense hidden layers (output added automatically)
)

// Image classifier with common architecture
cnn := Network.ImageClassifierCNN(28, 28, 1, 10)
```

### CNN Builder Methods

```go
builder := Network.NewCNNBuilder(height, width, channels)

// Convolutional layers
builder.Conv2D(filters, kernelSize, activation)
builder.Conv2DWithStride(filters, kernelSize, stride, activation)
builder.Conv2DFull(filters, kernelSize, stride, padding, activation)

// Pooling layers
builder.MaxPool(poolSize)
builder.MaxPoolWithStride(poolSize, stride)
builder.AvgPool(poolSize)

// Flattening
flatSize := builder.Flatten()  // Returns flattened size

// Dense layers (automatically handles flattening)
builder.Dense([]int{128, 64, 10}, Network.Softmax)
builder.DenseCustom([]int{128, 10}, Network.ReLU, Network.Softmax)

// Loss function
builder.WithLoss(Network.CategoricalCrossEntropy)

// Build
cnn := builder.Build()
```

## Complete Examples

### Example 1: XOR Problem

```go
package main

import (
    "neugo/Network"
)

func main() {
    // Data
    trainX := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    trainY := [][]float32{{0}, {1}, {1}, {0}}

    // Model
    model := Network.QuickBinary(2, 8, 4)

    // Train
    model.QuickFit(trainX, trainY, 1000, 0.1)

    // Evaluate
    metrics := model.Evaluate(trainX, trainY, 0.5)
    println("Accuracy:", metrics.Accuracy)
}
```

### Example 2: MNIST-Style Classifier

```go
package main

import (
    "neugo/Network"
    "neugo/data"
)

func main() {
    // Load data (assuming MNIST format)
    dataset, _ := data.LoadCSV("mnist.csv", data.CSVConfig{
        Delimiter: ',',
        HasHeader: true,
        LabelColumn: 0,
        LabelType: "multiclass",
    })

    // Normalize
    stats := data.CalculateStats(dataset.Features)
    normalized := data.NormalizeZScore(dataset.Features, stats)

    // Split
    split := data.SplitData(normalized, dataset.Labels, data.SplitConfig{
        TrainRatio: 0.7,
        ValRatio: 0.15,
        TestRatio: 0.15,
        Shuffle: true,
    })

    // Model - super clean!
    model := Network.MultiClassClassifier(784, 10, []int{128, 64})

    // Training with validation and early stopping
    config := Network.NewFitConfig(100).
        WithLearningRate(0.01).
        WithBatchSize(32).
        WithValidation(split.ValX, split.ValY).
        WithL2(0.001).
        WithDropout(0.2).
        WithEarlyStopping(10, 0.001)

    history := model.FitWithConfig(split.TrainX, split.TrainY, config)

    // Evaluate
    testMetrics := model.Evaluate(split.TestX, split.TestY, 0.5)
    println("Test Accuracy:", testMetrics.Accuracy)

    // Save
    model.SaveToFile("mnist_model.json")
}
```

### Example 3: CNN for Images

```go
package main

import (
    "neugo/Network"
)

func main() {
    // Build CNN
    cnn := Network.NewCNNBuilder(28, 28, 1).
        Conv2D(32, 3, Network.ReLU).
        Conv2D(32, 3, Network.ReLU).
        MaxPool(2).
        Conv2D(64, 3, Network.ReLU).
        Conv2D(64, 3, Network.ReLU).
        MaxPool(2).
        Dense([]int{128, 10}, Network.Softmax).
        WithLoss(Network.CategoricalCrossEntropy).
        Build()

    // Train (assuming you have image data)
    cnn.QuickFit(trainImages, trainLabels, 50, 0.01)

    // Evaluate
    metrics := cnn.Evaluate(testImages, testLabels, 0.5)
    println("Accuracy:", metrics.Accuracy)
}
```

### Example 4: Custom Training Loop

```go
package main

import (
    "neugo/Network"
)

func main() {
    model := Network.QuickBinary(10, 16, 8)

    // Create custom callbacks
    history := Network.NewHistory()
    checkpoint := Network.NewModelCheckpoint(
        "best_model.json",
        "f1",
        "max",
        true,
        true,
    )
    progress := Network.NewProgressBar(100, 10, true)

    callbacks := Network.NewCallbackList(history, checkpoint, progress)

    // Configure training
    config := Network.NewFitConfig(100).
        WithLearningRate(0.01).
        WithValidation(valX, valY).
        WithCallbacks(callbacks)

    // Train
    history = model.FitWithConfig(trainX, trainY, config)

    // Access history
    println("Training took:", history.Duration())
    println("Final loss:", history.TrainLoss[len(history.TrainLoss)-1])
}
```

## Comparison: Before vs After

### Before (Original API)

```go
// Creating model
layer1 := Network.NewLayerWithActivation(784, Network.Linear)
layer2 := Network.NewLayerWithActivation(128, Network.ReLU)
layer3 := Network.NewLayerWithActivation(64, Network.ReLU)
layer4 := Network.NewLayerWithActivation(10, Network.Softmax)
layers := []Network.Layer{layer1, layer2, layer3, layer4}
model := Network.NewNetworkWithLoss(layers, Network.CategoricalCrossEntropy)

// Training
losses := model.Fit(trainX, trainY, epochs, batchSize, learningRate, verbose)
```

### After (Clean API)

```go
// Creating model
model := Network.QuickMultiClass(10, 784, 128, 64)

// Training
history := model.QuickFit(trainX, trainY, epochs, learningRate)
```

**Result**: ~60% less code, infinitely more readable!

## Migration Guide

Old code will continue to work! The clean API is additive, not breaking.

To migrate gradually:

1. **New models**: Use the clean API
2. **Existing models**: Can keep using old API or migrate when you refactor
3. **Mixed usage**: Perfectly fine! Clean API builds standard `Network` objects

## Best Practices

### 1. Choose the Right Style for Your Use Case

- **Prototyping?** → Use `QuickBinary`, `QuickMultiClass`, etc.
- **Production?** → Use `Sequential` or `Stack` for clarity
- **Teaching?** → Use `Sequential` - most readable
- **Research?** → Use `Stack` - most flexible

### 2. Start Simple, Add Complexity

```go
// Start
model := Network.QuickBinary(inputSize, 64, 32)
history := model.QuickFit(trainX, trainY, 100)

// Add validation
config := Network.NewFitConfig(100).WithValidation(valX, valY)
history := model.FitWithConfig(trainX, trainY, config)

// Add regularization
config := Network.NewFitConfig(100).
    WithValidation(valX, valY).
    WithL2(0.001).
    WithDropout(0.2).
    WithEarlyStopping(10, 0.001)
```

### 3. Use Type Inference

```go
// Good - let Go infer the type
model := Network.QuickBinary(10, 64, 32)

// Also good - explicit when it aids readability
var classifier Network.Network = Network.MultiClassClassifier(10, 5, []int{64})
```

## Performance Notes

The clean API has **zero performance overhead**. It's just a nicer way to construct the same underlying `Network` and `CNN` types.

## Summary

The clean API provides:

✅ **4 different styles** to match your preference
✅ **Fluent, chainable interfaces**
✅ **Sensible defaults**
✅ **No manual size calculations** (for CNNs)
✅ **Clean, readable code**
✅ **100% backwards compatible**
✅ **Zero performance overhead**

Choose the style that makes your code most readable and maintainable. Happy building!
