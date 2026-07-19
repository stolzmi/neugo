package Network

// TrainBatch trains the network on a batch of samples
// Accumulates gradients across the batch before updating weights
func (network *Network) TrainBatch(inputs [][]float32, labels [][]float32, learningRate float32) float32 {
	if len(inputs) == 0 || len(inputs) != len(labels) {
		return 0
	}

	batchSize := len(inputs)
	totalLoss := float32(0.0)

	// Accumulate gradients for each layer
	accumulatedGradients := make([][][]float32, len(network.layers)-1)

	// Initialize accumulated gradients
	for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
		currentLayerSize := network.layers[layerIndex].Size()
		nextLayerRegularSize := network.layers[layerIndex+1].RegularSize()

		accumulatedGradients[layerIndex] = make([][]float32, currentLayerSize)
		for i := 0; i < currentLayerSize; i++ {
			accumulatedGradients[layerIndex][i] = make([]float32, nextLayerRegularSize)
		}
	}

	// Process each sample in the batch
	for sampleIdx := 0; sampleIdx < batchSize; sampleIdx++ {
		// Forward pass
		network.ForwardPass(inputs[sampleIdx])

		// Calculate loss
		totalLoss += network.CalculateLoss(labels[sampleIdx])

		// Backward pass (calculate gradients but don't update weights yet)
		network.calculateGradients(labels[sampleIdx])

		// Accumulate gradients (including bias weights)
		for layerIndex := 0; layerIndex < len(network.layers)-1; layerIndex++ {
			currentLayer := &network.layers[layerIndex]
			nextLayer := &network.layers[layerIndex+1]

			// Accumulate weight gradients from all neurons (including bias) to regular neurons
			for neuronIndex := 0; neuronIndex < currentLayer.Size(); neuronIndex++ {
				activation := currentLayer.neurons[neuronIndex].Activation()
				for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
					gradient := nextLayer.neurons[nextNeuronIndex].Gradient()
					accumulatedGradients[layerIndex][neuronIndex][nextNeuronIndex] += gradient * activation
				}
			}
		}
	}

	// Update weights (including bias weights) using averaged gradients
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

	// Return average loss for the batch
	return totalLoss / float32(batchSize)
}

// calculateGradients computes gradients without updating weights (for batch training)
func (network *Network) calculateGradients(labels []float32) {
	outputLayer := &network.layers[len(network.layers)-1]
	if len(labels) != outputLayer.RegularSize() {
		return
	}

	// Reset gradients
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

		// For BCE + Sigmoid or CCE + Softmax, gradient is already (p - y)
		// For other combinations, apply activation derivative
		if network.loss.Type == BinaryCrossEntropy && outputLayer.activation.Type == Sigmoid {
			outputLayer.neurons[i].setGradient(gradLoss)
		} else if network.loss.Type == CategoricalCrossEntropy && outputLayer.activation.Type == Softmax {
			outputLayer.neurons[i].setGradient(gradLoss)
		} else {
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

			// Sum gradients from regular neurons in next layer
			for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
				weight := network.weights[layerIndex][neuronIndex][nextNeuronIndex]
				gradient += nextLayer.neurons[nextNeuronIndex].Gradient() * weight
			}

			activation := currentLayer.neurons[neuronIndex].Activation()
			currentLayer.neurons[neuronIndex].setGradient(gradient * activationDeriv(activation))
		}

		// Bias node gradient (if present)
		if currentLayer.HasBias() {
			biasIndex := currentLayer.RegularSize()
			gradient := float32(0.0)

			for nextNeuronIndex := 0; nextNeuronIndex < nextLayer.RegularSize(); nextNeuronIndex++ {
				weight := network.weights[layerIndex][biasIndex][nextNeuronIndex]
				gradient += nextLayer.neurons[nextNeuronIndex].Gradient() * weight
			}

			currentLayer.neurons[biasIndex].setGradient(gradient)
		}
	}
}

// Fit trains the network for multiple epochs with batch support
func (network *Network) Fit(inputs [][]float32, labels [][]float32, epochs int, batchSize int, learningRate float32, verbose bool) []float32 {
	losses := make([]float32, 0, epochs)
	numSamples := len(inputs)

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0.0)
		numBatches := 0

		// Process data in batches
		for i := 0; i < numSamples; i += batchSize {
			end := i + batchSize
			if end > numSamples {
				end = numSamples
			}

			batchInputs := inputs[i:end]
			batchLabels := labels[i:end]

			batchLoss := network.TrainBatch(batchInputs, batchLabels, learningRate)
			epochLoss += batchLoss
			numBatches++
		}

		avgLoss := epochLoss / float32(numBatches)
		losses = append(losses, avgLoss)

		if verbose && ((epoch+1)%100 == 0 || epoch == 0) {
			println("Epoch", epoch+1, "- Loss:", avgLoss)
		}
	}

	return losses
}
