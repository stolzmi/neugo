// Package Network provides a simple neural network implementation in Go.
//
// This package includes:
//   - Flexible layer creation with configurable activation functions
//   - Multiple activation functions (Sigmoid, ReLU, Tanh, Linear, LeakyReLU)
//   - Multiple loss functions (MSE, Binary Cross-Entropy, Categorical Cross-Entropy, MAE)
//   - Xavier/Glorot weight initialization
//   - Backpropagation with gradient descent
//
// Example usage:
//
//	// Create layers with different activations
//	inputLayer := Network.NewLayer(2, Network.WithActivation(Network.Linear))
//	hiddenLayer := Network.NewLayer(4, Network.WithActivation(Network.ReLU))
//	outputLayer := Network.NewLayer(1, Network.WithActivation(Network.Sigmoid))
//
//	// Create network
//	layers := []Network.Layer{inputLayer, hiddenLayer, outputLayer}
//	network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)
//
//	// Train
//	network.ForwardPass(input)
//	network.BackPropagation(labels, learningRate)
package Network
