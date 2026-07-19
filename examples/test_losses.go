package main

import (
	"fmt"
	"neugo/Network"
)

func trainAndTest(name string, network Network.Network, inputs [][]float32, labels [][]float32, epochs int, learningRate float32) {
	fmt.Printf("\n=== Testing %s ===\n", name)

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)

		for i := 0; i < len(inputs); i++ {
			network.ForwardPass(inputs[i])
			totalLoss += network.CalculateLoss(labels[i])
			network.BackPropagation(labels[i], learningRate)
		}

		totalLoss /= float32(len(inputs))

		if (epoch+1)%2000 == 0 || epoch == 0 {
			fmt.Printf("Epoch %d - Loss: %.6f\n", epoch+1, totalLoss)
		}
	}

	// Test the trained network
	fmt.Println("\nResults:")
	for i := 0; i < len(inputs); i++ {
		network.ForwardPass(inputs[i])
		output := network.GetOutput()[0].Activation()
		fmt.Printf("Input: [%.1f, %.1f] -> Output: %.4f (Expected: %.1f)\n",
			inputs[i][0], inputs[i][1], output, labels[i][0])
	}
}

func main() {
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

	epochs := 10000
	learningRate := float32(0.1)

	// Test with MSE loss
	layer1MSE := Network.NewLayerWithActivation(2, Network.Sigmoid)
	layer2MSE := Network.NewLayerWithActivation(4, Network.Sigmoid)
	layer3MSE := Network.NewLayerWithActivation(1, Network.Sigmoid)
	layersMSE := []Network.Layer{layer1MSE, layer2MSE, layer3MSE}
	networkMSE := Network.NewNetworkWithLoss(layersMSE, Network.MSE)
	trainAndTest("MSE Loss", networkMSE, inputs, labels, epochs, learningRate)

	// Test with Binary Cross-Entropy loss
	layer1BCE := Network.NewLayerWithActivation(2, Network.Sigmoid)
	layer2BCE := Network.NewLayerWithActivation(4, Network.Sigmoid)
	layer3BCE := Network.NewLayerWithActivation(1, Network.Sigmoid)
	layersBCE := []Network.Layer{layer1BCE, layer2BCE, layer3BCE}
	networkBCE := Network.NewNetworkWithLoss(layersBCE, Network.BinaryCrossEntropy)
	trainAndTest("Binary Cross-Entropy Loss", networkBCE, inputs, labels, epochs, learningRate)

	// Test with MAE loss
	layer1MAE := Network.NewLayerWithActivation(2, Network.Sigmoid)
	layer2MAE := Network.NewLayerWithActivation(4, Network.Sigmoid)
	layer3MAE := Network.NewLayerWithActivation(1, Network.Sigmoid)
	layersMAE := []Network.Layer{layer1MAE, layer2MAE, layer3MAE}
	networkMAE := Network.NewNetworkWithLoss(layersMAE, Network.MAE)
	trainAndTest("MAE Loss", networkMAE, inputs, labels, epochs, learningRate)
}
