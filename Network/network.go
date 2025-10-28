package Network

import (
	"math"
	"math/rand"
)

type Network struct {
	layers  []Layer
	weights [][][]float32
}

func NewNetwork(layers []Layer) Network {
	weights := make([][][]float32, len(layers)-1)
	for i := 0; i < len(layers)-1; i++ {
		weights[i] = make([][]float32, layers[i].Size())
		for j := 0; j < layers[i].Size(); j++ {
			weights[i][j] = make([]float32, layers[i+1].Size())
			// Initialize weights using Xavier/Glorot initialization
			// Random values scaled by sqrt(1/n) where n is input size
			scale := float32(math.Sqrt(1.0 / float64(layers[i].Size())))
			for k := 0; k < layers[i+1].Size(); k++ {
				weights[i][j][k] = (rand.Float32()*2 - 1) * scale
			}
		}
	}
	return Network{layers: layers, weights: weights}
}

func (network *Network) Layers() []Layer {
	return network.layers
}

func (network *Network) Weights() [][][]float32 {
	return network.weights
}

func (network *Network) GetOutput() []Neuron {
	return network.layers[len(network.layers)].Neurons()
}

func (network *Network) ForwardPass(input []float32) {
	if len(input) != len(network.layers[0].neurons) {
		return
	}

	// Reset all activations except input layer
	for layerIndex := 1; layerIndex < len(network.layers); layerIndex++ {
		for neuronIndex := range network.layers[layerIndex].neurons {
			network.layers[layerIndex].neurons[neuronIndex].setActivation(0)
		}
	}

	// Set input layer activations
	for i, val := range input {
		network.layers[0].neurons[i].setActivation(val)
	}

	// Propagate through the network
	for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
		layer := network.layers[layerIndex]
		neurons := layer.neurons
		nextLayer := &network.layers[layerIndex+1]

		// First accumulate weighted sums
		for neuronIndex, neuron := range neurons {
			for nextNeuronIndex := range nextLayer.neurons {
				weight := network.weights[layerIndex][neuronIndex][nextNeuronIndex]
				nextLayer.neurons[nextNeuronIndex].setActivation(
					nextLayer.neurons[nextNeuronIndex].Activation() + neuron.Activation()*weight,
				)
			}
		}

		// Apply activation function to next layer (except for input layer propagation)
		for nextNeuronIndex := range nextLayer.neurons {
			rawActivation := nextLayer.neurons[nextNeuronIndex].Activation() + nextLayer.neurons[nextNeuronIndex].Bias()
			nextLayer.neurons[nextNeuronIndex].setActivation(sigmoid(rawActivation))
		}
	}
}

func (network *Network) BackPropagation(labels []float32, learningRate float32) {
	if len(labels) != len(network.layers[len(network.layers)-1].neurons) {
		return
	}

	// Reset gradients
	for i := range network.layers {
		for j := range network.layers[i].neurons {
			network.layers[i].neurons[j].setGradient(0)
		}
	}

	// Calculate output layer gradients (using MSE loss derivative with sigmoid derivative)
	outputLayer := &network.layers[len(network.layers)-1]
	for i := range outputLayer.neurons {
		// Gradient = (activation - label) * sigmoid'(activation)
		activation := outputLayer.neurons[i].Activation()
		error := activation - labels[i]
		// Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
		sigmoidDeriv := activation * (1 - activation)
		outputLayer.neurons[i].setGradient(error * sigmoidDeriv)
	}

	// Backpropagate through hidden layers
	for layerIndex := len(network.layers) - 2; layerIndex >= 1; layerIndex-- {
		currentLayer := &network.layers[layerIndex]
		nextLayer := &network.layers[layerIndex+1]

		for neuronIndex := range currentLayer.neurons {
			gradient := float32(0.0)

			// Sum gradients from next layer weighted by connections
			for nextNeuronIndex := range nextLayer.neurons {
				weight := network.weights[layerIndex][neuronIndex][nextNeuronIndex]
				gradient += nextLayer.neurons[nextNeuronIndex].Gradient() * weight
			}

			// Apply sigmoid derivative for hidden layers
			activation := currentLayer.neurons[neuronIndex].Activation()
			sigmoidDeriv := activation * (1 - activation)
			currentLayer.neurons[neuronIndex].setGradient(gradient * sigmoidDeriv)
		}
	}

	// Update weights and biases
	for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
		currentLayer := &network.layers[layerIndex]
		nextLayer := &network.layers[layerIndex+1]

		for neuronIndex := range currentLayer.neurons {
			for nextNeuronIndex := range nextLayer.neurons {
				// Update weight: w = w - learningRate * gradient * activation
				gradient := nextLayer.neurons[nextNeuronIndex].Gradient()
				activation := currentLayer.neurons[neuronIndex].Activation()
				network.weights[layerIndex][neuronIndex][nextNeuronIndex] -= learningRate * gradient * activation
			}
		}

		// Update biases for next layer
		for nextNeuronIndex := range nextLayer.neurons {
			gradient := nextLayer.neurons[nextNeuronIndex].Gradient()
			currentBias := nextLayer.neurons[nextNeuronIndex].Bias()
			nextLayer.neurons[nextNeuronIndex].setBias(currentBias - learningRate*gradient)
		}
	}
}
