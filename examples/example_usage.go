package main

import (
	"fmt"
	"neugo/Network"
)

func main() {
	fmt.Println("=== Example: Different ways to create layers ===\n")

	// 1. Default layer (uses Sigmoid activation by default)
	layer1 := Network.NewLayer(3)
	fmt.Println("1. Default layer (3 neurons, Sigmoid activation)")
	fmt.Printf("   Size: %d, Activation: %v\n\n", layer1.Size(), layer1.Activation().Type)

	// 2. Layer with custom activation
	layer2 := Network.NewLayerWithActivation(5, Network.ReLU)
	fmt.Println("2. Layer with ReLU activation (5 neurons)")
	fmt.Printf("   Size: %d, Activation: %v\n\n", layer2.Size(), layer2.Activation().Type)

	// 3. Multiple layers with different activations
	inputLayer := Network.NewLayerWithActivation(2, Network.Linear)
	hiddenLayer := Network.NewLayerWithActivation(4, Network.Tanh)
	outputLayer := Network.NewLayerWithActivation(1, Network.Sigmoid)

	fmt.Println("3. Creating a complete network:")
	fmt.Printf("   Input layer:  %d neurons, Activation: %v\n", inputLayer.Size(), inputLayer.Activation().Type)
	fmt.Printf("   Hidden layer: %d neurons, Activation: %v\n", hiddenLayer.Size(), hiddenLayer.Activation().Type)
	fmt.Printf("   Output layer: %d neurons, Activation: %v\n\n", outputLayer.Size(), outputLayer.Activation().Type)

	// 4. Create network with default loss
	layers := []Network.Layer{inputLayer, hiddenLayer, outputLayer}
	network := Network.NewNetwork(layers)
	fmt.Println("4. Network created with default loss (MSE)")
	fmt.Printf("   Layers: %d\n\n", len(network.Layers()))

	// 5. Create network with custom loss
	layer1BCE := Network.NewLayer(2)
	layer2BCE := Network.NewLayer(4)
	layer3BCE := Network.NewLayer(1)
	layersBCE := []Network.Layer{layer1BCE, layer2BCE, layer3BCE}
	networkBCE := Network.NewNetworkWithLoss(layersBCE, Network.BinaryCrossEntropy)
	fmt.Println("5. Network created with Binary Cross-Entropy loss")
	fmt.Printf("   Layers: %d\n\n", len(networkBCE.Layers()))

	fmt.Println("=== Summary ===")
	fmt.Println("✓ Backward compatible: NewLayer(size) still works with defaults")
	fmt.Println("✓ Flexible: NewLayer(size, WithActivation(...)) allows customization")
	fmt.Println("✓ Extensible: Can add more options in the future (e.g., WithBias, WithDropout)")
}
