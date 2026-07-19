package main

import (
	"fmt"
	"neugo/Network"
)

func main() {
	// Create a simple network for XOR problem
	// Input: 2 neurons, Hidden: 4 neurons, Output: 1 neuron
	layer1 := Network.NewLayer(2) // Input layer
	layer2 := Network.NewLayer(4) // Hidden layer
	layer3 := Network.NewLayer(1) // Output layer

	layers := []Network.Layer{layer1, layer2, layer3}
	network := Network.NewNetwork(layers)

	// XOR dataset
	inputs := [][]float32{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}
	labels := [][]float32{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}

	learningRate := float32(0.1)
	epochs := 10000

	fmt.Println("Training XOR network...")
	fmt.Printf("Architecture: %d -> %d -> %d\n", 2, 4, 1)
	fmt.Printf("Learning rate: %.2f\n", learningRate)
	fmt.Printf("Epochs: %d\n\n", epochs)

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)

		// Train on each sample
		for i := 0; i < len(inputs); i++ {
			// Forward pass
			network.ForwardPass(inputs[i])

			// Calculate loss
			outputLayer := network.Layers()[len(network.Layers())-1]
			for j, neuron := range outputLayer.Neurons() {
				diff := neuron.Activation() - labels[i][j]
				totalLoss += diff * diff
			}

			// Backpropagation
			network.BackPropagation(labels[i], learningRate)
		}

		// Average loss
		totalLoss /= float32(len(inputs))

		// Print progress
		if (epoch+1)%1000 == 0 || epoch == 0 {
			fmt.Printf("Epoch %d - Loss: %.6f\n", epoch+1, totalLoss)
		}
	}

	// Test the trained network
	fmt.Println("\n=== Testing Trained Network ===")
	for i := 0; i < len(inputs); i++ {
		network.ForwardPass(inputs[i])
		outputLayer := network.Layers()[len(network.Layers())-1]
		output := outputLayer.Neurons()[0].Activation()
		fmt.Printf("Input: [%.1f, %.1f] -> Output: %.4f (Expected: %.1f)\n",
			inputs[i][0], inputs[i][1], output, labels[i][0])
	}
}
