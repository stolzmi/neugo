package Network

import "math/rand"

// RegularizationType represents the type of regularization
type RegularizationType int

const (
	NoRegularization RegularizationType = iota
	L1Regularization
	L2Regularization
)

// RegularizationConfig holds regularization parameters
type RegularizationConfig struct {
	Type    RegularizationType
	Lambda  float32 // Regularization strength
	Dropout float32 // Dropout rate (0.0 to 1.0)
}

// ApplyL2Regularization applies L2 weight decay to gradients
func ApplyL2Regularization(weights [][][]float32, gradients [][][]float32, lambda float32) {
	for i := range weights {
		for j := range weights[i] {
			for k := range weights[i][j] {
				// L2 gradient: lambda * weight
				gradients[i][j][k] += lambda * weights[i][j][k]
			}
		}
	}
}

// ApplyL1Regularization applies L1 regularization to gradients
func ApplyL1Regularization(weights [][][]float32, gradients [][][]float32, lambda float32) {
	for i := range weights {
		for j := range weights[i] {
			for k := range weights[i][j] {
				// L1 gradient: lambda * sign(weight)
				if weights[i][j][k] > 0 {
					gradients[i][j][k] += lambda
				} else if weights[i][j][k] < 0 {
					gradients[i][j][k] -= lambda
				}
			}
		}
	}
}

// CalculateL2Loss calculates the L2 regularization loss
func CalculateL2Loss(weights [][][]float32, lambda float32) float32 {
	loss := float32(0.0)
	for i := range weights {
		for j := range weights[i] {
			for k := range weights[i][j] {
				loss += weights[i][j][k] * weights[i][j][k]
			}
		}
	}
	return 0.5 * lambda * loss
}

// CalculateL1Loss calculates the L1 regularization loss
func CalculateL1Loss(weights [][][]float32, lambda float32) float32 {
	loss := float32(0.0)
	for i := range weights {
		for j := range weights[i] {
			for k := range weights[i][j] {
				if weights[i][j][k] >= 0 {
					loss += weights[i][j][k]
				} else {
					loss -= weights[i][j][k]
				}
			}
		}
	}
	return lambda * loss
}

// ApplyDropout randomly sets activations to zero during training
// Returns a mask indicating which neurons were dropped
func ApplyDropout(layer *Layer, dropoutRate float32, training bool) []bool {
	if !training || dropoutRate <= 0 {
		return nil
	}

	mask := make([]bool, len(layer.neurons))
	scale := 1.0 / (1.0 - dropoutRate) // Inverted dropout

	for i := range layer.neurons {
		if rand.Float32() < dropoutRate {
			// Drop this neuron
			layer.neurons[i].setActivation(0)
			mask[i] = false
		} else {
			// Keep and scale
			currentActivation := layer.neurons[i].Activation()
			layer.neurons[i].setActivation(currentActivation * scale)
			mask[i] = true
		}
	}

	return mask
}

// UpdateLayerWithRegularization updates a layer configuration to include regularization
func (layer *Layer) SetRegularization(regType RegularizationType, lambda float32, dropout float32) {
	// Store in layer (would need to add fields to Layer struct)
	// For now, this is a placeholder for the API
}

// TrainBatchWithRegularization trains with regularization support
func (network *Network) TrainBatchWithRegularization(
	inputs [][]float32,
	labels [][]float32,
	learningRate float32,
	l2Lambda float32,
	dropoutRate float32,
) float32 {

	if len(inputs) == 0 || len(inputs) != len(labels) {
		return 0
	}

	batchSize := len(inputs)
	totalLoss := float32(0.0)

	// Accumulate gradients
	accumulatedGradients := make([][][]float32, len(network.layers)-1)

	for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
		currentLayerSize := network.layers[layerIndex].Size()
		nextLayerRegularSize := network.layers[layerIndex+1].RegularSize()

		accumulatedGradients[layerIndex] = make([][]float32, currentLayerSize)
		for i := 0; i < currentLayerSize; i++ {
			accumulatedGradients[layerIndex][i] = make([]float32, nextLayerRegularSize)
		}
	}

	// Process each sample
	for sampleIdx := 0; sampleIdx < batchSize; sampleIdx++ {
		// Forward pass with dropout
		network.forwardPassWithDropout(inputs[sampleIdx], dropoutRate, true)

		// Calculate loss
		totalLoss += network.CalculateLoss(labels[sampleIdx])

		// Backward pass
		network.calculateGradients(labels[sampleIdx])

		// Accumulate gradients (including bias weights)
		for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
			currentLayer := &network.layers[layerIndex]
			nextLayer := &network.layers[layerIndex+1]

			for neuronIndex := 0; neuronIndex < currentLayer.Size(); neuronIndex++ {
				activation := currentLayer.neurons[neuronIndex].Activation()
				for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
					gradient := nextLayer.neurons[nextNeuronIndex].Gradient()
					accumulatedGradients[layerIndex][neuronIndex][nextNeuronIndex] += gradient * activation
				}
			}
		}
	}

	// Apply L2 regularization to gradients
	if l2Lambda > 0 {
		ApplyL2Regularization(network.weights, accumulatedGradients, l2Lambda)
		// Add regularization loss
		totalLoss += CalculateL2Loss(network.weights, l2Lambda)
	}

	// Update weights (including bias weights) with averaged gradients
	for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
		currentLayer := &network.layers[layerIndex]
		nextLayer := &network.layers[layerIndex+1]

		for neuronIndex := 0; neuronIndex < currentLayer.Size(); neuronIndex++ {
			for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
				avgGradient := accumulatedGradients[layerIndex][neuronIndex][nextNeuronIndex] / float32(batchSize)
				network.weights[layerIndex][neuronIndex][nextNeuronIndex] -= learningRate * avgGradient
			}
		}
	}

	return totalLoss / float32(batchSize)
}

// forwardPassWithDropout performs forward pass with dropout
func (network *Network) forwardPassWithDropout(input []float32, dropoutRate float32, training bool) {
	if len(input) != network.layers[0].Size() {
		return
	}

	// Set input layer
	for i, val := range input {
		network.layers[0].neurons[i].setActivation(val)
	}

	// Propagate through network
	for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
		layer := network.layers[layerIndex]
		nextLayer := &network.layers[layerIndex+1]

		// Reset activations of regular neurons in next layer (not bias node)
		for neuronIndex := 0; neuronIndex < nextLayer.RegularSize(); neuronIndex++ {
			nextLayer.neurons[neuronIndex].setActivation(0)
		}

		// Accumulate weighted sums from all neurons (including bias) to regular neurons
		for neuronIndex := 0; neuronIndex < layer.Size(); neuronIndex++ {
			neuron := &layer.neurons[neuronIndex]

			for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
				weight := network.weights[layerIndex][neuronIndex][nextNeuronIndex]
				nextLayer.neurons[nextNeuronIndex].setActivation(
					nextLayer.neurons[nextNeuronIndex].Activation() + neuron.Activation()*weight,
				)
			}
		}

		// Apply activation function to regular neurons
		activationFunc := nextLayer.activation.Apply
		for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
			rawActivation := nextLayer.neurons[nextNeuronIndex].Activation()
			nextLayer.neurons[nextNeuronIndex].setActivation(activationFunc(rawActivation))
		}

		// Keep bias node at 1.0 if next layer has bias
		if nextLayer.HasBias() {
			nextLayer.neurons[nextLayer.RegularSize()].setActivation(1.0)
		}

		// Apply dropout (except output layer) - only to regular neurons, not bias
		if layerIndex < len(network.layers)-2 && training {
			ApplyDropout(nextLayer, dropoutRate, training)
		}
	}
}
