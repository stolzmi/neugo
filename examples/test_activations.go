package main

import (
	"fmt"
	"neugo/Network"
)

func main() {
	// Test with ReLU activation
	fmt.Println("=== Testing with ReLU Activation ===")
	layer1 := Network.NewLayerWithActivation(2, Network.Linear)  // Input layer (no activation)
	layer2 := Network.NewLayerWithActivation(4, Network.ReLU)    // Hidden layer with ReLU
	layer3 := Network.NewLayerWithActivation(1, Network.Sigmoid) // Output layer with Sigmoid

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

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(inputs); i++ {
			network.ForwardPass(inputs[i])
			network.BackPropagation(labels[i], learningRate)
		}

		if (epoch+1)%2000 == 0 || epoch == 0 {
			totalLoss := float32(0.0)
			for i := 0; i < len(inputs); i++ {
				network.ForwardPass(inputs[i])
				outputLayer := network.Layers()[len(network.Layers())-1]
				for j, neuron := range outputLayer.Neurons() {
					diff := neuron.Activation() - labels[i][j]
					totalLoss += diff * diff
				}
			}
			totalLoss /= float32(len(inputs))
			fmt.Printf("Epoch %d - Loss: %.6f\n", epoch+1, totalLoss)
		}
	}

	// Test the trained network
	fmt.Println("\nResults:")
	for i := 0; i < len(inputs); i++ {
		network.ForwardPass(inputs[i])
		outputLayer := network.Layers()[len(network.Layers())-1]
		output := outputLayer.Neurons()[0].Activation()
		fmt.Printf("Input: [%.1f, %.1f] -> Output: %.4f (Expected: %.1f)\n",
			inputs[i][0], inputs[i][1], output, labels[i][0])
	}

	// Test with Tanh activation
	fmt.Println("\n=== Testing with Tanh Activation ===")
	layer1Tanh := Network.NewLayerWithActivation(2, Network.Linear)
	layer2Tanh := Network.NewLayerWithActivation(4, Network.Tanh)
	layer3Tanh := Network.NewLayerWithActivation(1, Network.Sigmoid)

	layersTanh := []Network.Layer{layer1Tanh, layer2Tanh, layer3Tanh}
	networkTanh := Network.NewNetwork(layersTanh)

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(inputs); i++ {
			networkTanh.ForwardPass(inputs[i])
			networkTanh.BackPropagation(labels[i], learningRate)
		}

		if (epoch+1)%2000 == 0 || epoch == 0 {
			totalLoss := float32(0.0)
			for i := 0; i < len(inputs); i++ {
				networkTanh.ForwardPass(inputs[i])
				outputLayer := networkTanh.Layers()[len(networkTanh.Layers())-1]
				for j, neuron := range outputLayer.Neurons() {
					diff := neuron.Activation() - labels[i][j]
					totalLoss += diff * diff
				}
			}
			totalLoss /= float32(len(inputs))
			fmt.Printf("Epoch %d - Loss: %.6f\n", epoch+1, totalLoss)
		}
	}

	// Test the trained network
	fmt.Println("\nResults:")
	for i := 0; i < len(inputs); i++ {
		networkTanh.ForwardPass(inputs[i])
		outputLayer := networkTanh.Layers()[len(networkTanh.Layers())-1]
		output := outputLayer.Neurons()[0].Activation()
		fmt.Printf("Input: [%.1f, %.1f] -> Output: %.4f (Expected: %.1f)\n",
			inputs[i][0], inputs[i][1], output, labels[i][0])
	}
}
