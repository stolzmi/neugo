# Convolutional Neural Networks (CNN) in NeuGo

## Overview

NeuGo now supports Convolutional Neural Networks with the following features:

### Components

1. **Tensor Operations** (`tensor/tensor.go`)
   - 3D tensor support for image data (Height × Width × Channels)
   - Flatten/Unflatten operations
   - Tensor arithmetic (Add, Scale, Clone)

2. **Conv2D Layer** (`Network/conv2d.go`)
   - Configurable kernel size, stride, and padding
   - Multiple output channels (filters)
   - He initialization for weights
   - Forward and backward propagation
   - Supports any activation function

3. **Pooling Layers** (`Network/pooling.go`)
   - MaxPool2D: Max pooling with configurable pool size and stride
   - AvgPool2D: Average pooling with configurable pool size and stride
   - Gradient backpropagation for both

4. **Flatten Layer** (`Network/flatten.go`)
   - Converts 3D tensor to 1D array
   - Connects CNN layers to dense layers
   - Backward pass reshapes gradients back to 3D

5. **CNN Architecture** (`Network/cnn.go`)
   - Combines convolutional and dense layers
   - Automatic dimension tracking
   - Training and evaluation methods
   - Full backpropagation support

6. **Image Data Utilities** (`data/image.go`)
   - Load MNIST-format CSV data
   - Binary classification from images
   - Train/val/test splitting for images
   - Automatic normalization (0-255 → 0-1)

## Usage Example

```go
package main

import (
    "neugo/Network"
    "neugo/tensor"
)

func main() {
    // Create CNN with input dimensions
    cnn := Network.NewCNN(28, 28, 1, Network.BinaryCrossEntropy)

    // Add convolutional layers
    cnn.AddConv2D(32, 3, 1, 1, Network.ReLU)  // 32 filters, 3x3 kernel
    cnn.AddMaxPool2D(2, 2)                     // 2x2 pooling
    cnn.AddConv2D(64, 3, 1, 1, Network.ReLU)  // 64 filters
    cnn.AddMaxPool2D(2, 2)

    // Flatten for dense layers
    cnn.AddFlatten()

    // Add dense layers
    denseLayers := []Network.Layer{
        Network.NewLayerWithActivation(7*7*64, Network.Linear),
        Network.NewLayerWithActivation(128, Network.ReLU),
        Network.NewLayerWithActivation(1, Network.Sigmoid),
    }
    cnn.SetDenseNetwork(denseLayers)

    // Train
    losses := cnn.Train(trainImages, trainLabels, 50, 0.01)

    // Evaluate
    metrics := cnn.Evaluate(testImages, testLabels, 0.5)
    println("Test Accuracy:", metrics.Accuracy)
}
```

## Architecture Details

### Conv2D Parameters

- `inputChannels`: Number of input channels (1 for grayscale, 3 for RGB)
- `outputChannels`: Number of filters to learn
- `kernelSize`: Size of convolution kernel (e.g., 3 for 3×3)
- `stride`: Step size for convolution
- `padding`: Zero-padding around input
- `activation`: Activation function (ReLU, Sigmoid, Tanh, etc.)

### Output Dimensions

For Conv2D:
```
outputHeight = (inputHeight + 2×padding - kernelSize) / stride + 1
outputWidth = (inputWidth + 2×padding - kernelSize) / stride + 1
```

For MaxPool2D/AvgPool2D:
```
outputHeight = (inputHeight - poolSize) / stride + 1
outputWidth = (inputWidth - poolSize) / stride + 1
```

## Training Features

1. **Forward Pass**: Automatic propagation through conv → pool → flatten → dense
2. **Backward Pass**: Full gradient backpropagation from output to input
3. **Weight Updates**: Gradients applied to both kernels and biases
4. **Batch Training**: Process one image at a time (batch support coming soon)

## Image Data Format

Images are represented as 3D tensors:
- **Dimensions**: [Channels][Height][Width]
- **Grayscale**: 1 channel
- **RGB**: 3 channels
- **Values**: Normalized to [0, 1]

## Performance

The demo shows:
- **Pattern Recognition**: 100% accuracy on vertical/horizontal patterns
- **Training Speed**: 50 epochs in seconds
- **Convergence**: Loss drops from 0.67 → 0.02

## Files Added

- `tensor/tensor.go` - 3D tensor operations
- `Network/conv2d.go` - Convolutional layer
- `Network/pooling.go` - MaxPool2D and AvgPool2D
- `Network/flatten.go` - Flatten layer
- `Network/cnn.go` - CNN architecture
- `data/image.go` - Image data utilities
- `examples/cnn_demo.go` - Working CNN example

## Next Steps

Possible enhancements:
- Batch training for CNNs
- More pooling types (Global pooling)
- Batch normalization
- Dropout for conv layers
- Data augmentation
- Pre-trained model loading
- More sophisticated image datasets
