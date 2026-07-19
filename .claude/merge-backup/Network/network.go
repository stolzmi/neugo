package Network

import (
	"math"
	"math/rand"
)

type Network struct {
	layers  []Layer
	weights [][][]float32
	loss    LossFunction
}

func NewNetwork(layers []Layer) Network {
	return NewNetworkWithLoss(layers, MSE)
}

func NewNetworkWithLoss(layers []Layer, lossType LossType) Network {
	// Add bias nodes to all layers except input layer
	modifiedLayers := make([]Layer, len(layers))
	modifiedLayers[0] = layers[0] // Input layer without bias

	for i := 1; i < len(layers); i++ {
		// Add bias to all hidden and output layers
		size := layers[i].Size()
		if layers[i].HasBias() {
			size = layers[i].RegularSize()
		}
		modifiedLayers[i] = newLayerInternal(
			size,
			layers[i].activation.Type,
			true, // Add bias node
		)
	}

	// Initialize weights including connections from bias nodes
	weights := make([][][]float32, len(modifiedLayers)-1)
	for i := 0; i < len(modifiedLayers)-1; i++ {
		currentLayerSize := modifiedLayers[i].Size() // Includes input neurons (and bias if present)
		nextLayerRegularSize := modifiedLayers[i+1].RegularSize() // Only regular neurons (not bias)

		weights[i] = make([][]float32, currentLayerSize)

		// Initialize weights using Xavier/Glorot initialization
		// Scale based on the number of regular neurons (excluding bias in next layer)
		scale := float32(math.Sqrt(1.0 / float64(currentLayerSize)))

		for j := 0; j < currentLayerSize; j++ {
			weights[i][j] = make([]float32, nextLayerRegularSize)
			for k := 0; k < nextLayerRegularSize; k++ {
				weights[i][j][k] = (rand.Float32()*2 - 1) * scale
			}
		}
	}

	return Network{
		layers:  modifiedLayers,
		weights: weights,
		loss:    GetLossFunction(lossType),
	}
}

func (network *Network) Layers() []Layer {
	return network.layers
}

func (network *Network) Weights() [][][]float32 {
	return network.weights
}

func (network *Network) GetOutput() []Neuron {
	return network.layers[len(network.layers)-1].Neurons()
}

func (network *Network) CalculateLoss(labels []float32) float32 {
	outputLayer := network.layers[len(network.layers)-1]
	// Only use regular neurons (not bias) for predictions
	regularSize := outputLayer.RegularSize()
	predictions := make([]float32, regularSize)
	for i := 0; i < regularSize; i++ {
		predictions[i] = outputLayer.neurons[i].Activation()
	}
	return network.loss.Calculate(predictions, labels)
}

func (network *Network) ForwardPass(input []float32) {
	if len(input) != network.layers[0].Size() {
		return
	}

	// Set input layer activations
	for i, val := range input {
		network.layers[0].neurons[i].setActivation(val)
	}

	// Propagate through the network
	for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
		layer := network.layers[layerIndex]
		nextLayer := &network.layers[layerIndex+1]

		// Reset activations of regular neurons in next layer (not bias node)
		for neuronIndex := 0; neuronIndex < nextLayer.RegularSize(); neuronIndex++ {
			nextLayer.neurons[neuronIndex].setActivation(0)
		}

		// Accumulate weighted sums
		// This includes all neurons in current layer (regular neurons + bias if present)
		for neuronIndex := 0; neuronIndex < layer.Size(); neuronIndex++ {
			neuron := &layer.neurons[neuronIndex]

			// Only propagate to regular neurons in next layer (not to bias node)
			for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
				weight := network.weights[layerIndex][neuronIndex][nextNeuronIndex]
				nextLayer.neurons[nextNeuronIndex].setActivation(
					nextLayer.neurons[nextNeuronIndex].Activation() + neuron.Activation()*weight,
				)
			}
		}

		// Apply activation function to regular neurons in next layer
		activationFunc := nextLayer.activation.Apply
		for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
			rawActivation := nextLayer.neurons[nextNeuronIndex].Activation()
			nextLayer.neurons[nextNeuronIndex].setActivation(activationFunc(rawActivation))
		}

		// Keep bias node at 1.0 if next layer has bias
		if nextLayer.HasBias() {
			nextLayer.neurons[nextLayer.RegularSize()].setActivation(1.0)
		}
	}
}

func (network *Network) BackPropagation(labels []float32, learningRate float32) {
	outputLayer := &network.layers[len(network.layers)-1]
	if len(labels) != outputLayer.RegularSize() {
		return
	}

	// Reset gradients for all neurons
	for i := range network.layers {
		for j := range network.layers[i].neurons {
			network.layers[i].neurons[j].setGradient(0)
		}
	}

	// Calculate output layer gradients (only for regular neurons, not bias)
	activationDeriv := outputLayer.activation.Derivative
	lossGradient := network.loss.Gradient

	for i := 0; i < outputLayer.RegularSize(); i++ {
		activation := outputLayer.neurons[i].Activation()
		gradLoss := lossGradient(activation, labels[i])

		// For BCE + Sigmoid or CCE + Softmax, the gradient already includes activation derivative
		// For other combinations, we need to apply it
		// Check if we're using BCE with Sigmoid (simplified gradient)
		if network.loss.Type == BinaryCrossEntropy && outputLayer.activation.Type == Sigmoid {
			// BCE + Sigmoid gradient is already (p - y), don't multiply again
			outputLayer.neurons[i].setGradient(gradLoss)
		} else if network.loss.Type == CategoricalCrossEntropy && outputLayer.activation.Type == Softmax {
			// CCE + Softmax gradient is already (p - y), don't multiply again
			outputLayer.neurons[i].setGradient(gradLoss)
		} else {
			// For other combinations (e.g., MSE + Sigmoid), apply activation derivative
			outputLayer.neurons[i].setGradient(gradLoss * activationDeriv(activation))
		}
	}

	// Backpropagate through hidden layers
	for layerIndex := len(network.layers) - 2; layerIndex >= 1; layerIndex-- {
		currentLayer := &network.layers[layerIndex]
		nextLayer := &network.layers[layerIndex+1]
		activationDeriv := currentLayer.activation.Derivative

		// Backpropagate to regular neurons only (not bias node)
		for neuronIndex := 0; neuronIndex < currentLayer.RegularSize(); neuronIndex++ {
			gradient := float32(0.0)

			// Sum gradients from regular neurons in next layer (not from bias)
			for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
				weight := network.weights[layerIndex][neuronIndex][nextNeuronIndex]
				gradient += nextLayer.neurons[nextNeuronIndex].Gradient() * weight
			}

			// Apply activation derivative for hidden layers
			activation := currentLayer.neurons[neuronIndex].Activation()
			currentLayer.neurons[neuronIndex].setGradient(gradient * activationDeriv(activation))
		}

		// Bias node gradient computation
		// The bias node also receives gradient from next layer
		if currentLayer.HasBias() {
			biasIndex := currentLayer.RegularSize()
			gradient := float32(0.0)

			for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
				weight := network.weights[layerIndex][biasIndex][nextNeuronIndex]
				gradient += nextLayer.neurons[nextNeuronIndex].Gradient() * weight
			}

			// Bias activation is constant (1.0), so derivative is 1
			// No need to multiply by activation derivative
			currentLayer.neurons[biasIndex].setGradient(gradient)
		}
	}

	// Update weights (including bias weights)
	for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
		currentLayer := &network.layers[layerIndex]
		nextLayer := &network.layers[layerIndex+1]

		// Update all weights from current layer (including bias) to next layer regular neurons
		for neuronIndex := 0; neuronIndex < currentLayer.Size(); neuronIndex++ {
			for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
				// Update weight: w = w - learningRate * gradient * activation
				gradient := nextLayer.neurons[nextNeuronIndex].Gradient()
				activation := currentLayer.neurons[neuronIndex].Activation()
				network.weights[layerIndex][neuronIndex][nextNeuronIndex] -= learningRate * gradient * activation
			}
		}
	}
}
