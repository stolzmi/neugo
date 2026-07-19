# CNN Examples

## Overview

NeuGo now supports Convolutional Neural Networks for image classification tasks.

## Available Examples

### 1. cnn_demo.go - Basic CNN Demo
Simple pattern recognition task (vertical vs horizontal stripes).
- **Dataset**: 100 synthetic images (50 per class)
- **Architecture**: 2 conv layers, 2 pooling layers, 3 dense layers
- **Training**: 50 epochs
- **Expected Accuracy**: ~100%

```bash
go run examples/cnn_demo.go
```

### 2. cnn_showcase.go - Advanced CNN
Shape classification (circles vs squares) with noise.
- **Dataset**: 200 training + 40 test images
- **Architecture**: 3 conv layers (16→32→64 filters), 3 pooling, 3 dense
- **Features**: Learning rate decay, ~433K parameters
- **Training**: 100 epochs
- **Expected Accuracy**: 95-100%

```bash
go run examples/cnn_showcase.go
```

### 3. cifar10_demo.go - Real-World Images
Binary classification on CIFAR-10 (airplane vs automobile).
- **Dataset**: CIFAR-10 subset (32×32 RGB images)
- **Architecture**: 3 conv layers (32→64→128), 3 pooling, 3 dense
- **Training**: 20 epochs
- **Expected Accuracy**: 75-85%

**Setup Required:**
1. Download CIFAR-10 binary version from:
   https://www.cs.toronto.edu/~kriz/cifar.html
2. Extract to `dataset/cifar10/`
3. Files needed: `data_batch_1.bin`, `test_batch.bin`

```bash
go run examples/cifar10_demo.go
```

## CNN Architecture Components

### Convolutional Layers
- **Conv2D**: Extract spatial features
- **Parameters**: filters, kernel size, stride, padding, activation
- **Example**: `cnn.AddConv2D(32, 3, 1, 1, Network.ReLU)`

### Pooling Layers
- **MaxPool2D**: Spatial downsampling (keeps max values)
- **AvgPool2D**: Spatial downsampling (averages values)
- **Example**: `cnn.AddMaxPool2D(2, 2)`

### Other Layers
- **Flatten**: Convert 3D tensor to 1D for dense layers
- **Dense**: Fully-connected layers for classification

## Typical CNN Pipeline

```go
// 1. Create CNN
cnn := Network.NewCNN(height, width, channels, Network.BinaryCrossEntropy)

// 2. Add convolutional layers
cnn.AddConv2D(32, 3, 1, 1, Network.ReLU)
cnn.AddMaxPool2D(2, 2)
cnn.AddConv2D(64, 3, 1, 1, Network.ReLU)
cnn.AddMaxPool2D(2, 2)

// 3. Flatten and add dense layers
cnn.AddFlatten()
denseLayers := []Network.Layer{
    Network.NewLayerWithActivation(flatSize, Network.Linear),
    Network.NewLayerWithActivation(128, Network.ReLU),
    Network.NewLayerWithActivation(numClasses, Network.Sigmoid),
}
cnn.SetDenseNetwork(denseLayers)

// 4. Train
losses := cnn.Train(trainImages, trainLabels, epochs, learningRate)

// 5. Evaluate
metrics := cnn.Evaluate(testImages, testLabels, 0.5)
```

## Performance Tips

1. **Learning Rate**: Start with 0.01 for small datasets, 0.001 for CIFAR-10
2. **Epochs**: 50-100 for synthetic data, 20-50 for real images
3. **Architecture**: Start small, increase complexity if underfitting
4. **Filters**: Common progression: 16→32→64 or 32→64→128
5. **Pooling**: After 1-2 conv layers to reduce spatial dimensions

## Image Data Format

- **Grayscale**: 1 channel (28×28×1)
- **RGB**: 3 channels (32×32×3)
- **Values**: Normalized to [0, 1]
- **Storage**: [Channels][Height][Width]

## Next Steps

- Add data augmentation (rotation, flip, crop)
- Implement batch normalization
- Add dropout for conv layers
- Support for larger datasets (batch processing)
- Pre-trained model weights
- Multi-class classification (10+ classes)
