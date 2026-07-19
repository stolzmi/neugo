package main

import (
	"fmt"
	"math/rand"
	"neugo/Network"
)

func main() {
	// Create 4 layers with different sizes
	layer1 := Network.NewLayer(3) // Input layer with 3 neurons
	layer2 := Network.NewLayer(5) // Hidden layer with 5 neurons
	layer3 := Network.NewLayer(4) // Hidden layer with 4 neurons
	layer4 := Network.NewLayer(2) // Output layer with 2 neurons

	// Create network with these layers
	layers := []Network.Layer{layer1, layer2, layer3, layer4}
	network := Network.NewNetwork(layers)

	// Print network information
	fmt.Println("Network created with 4 layers:")
	for i, layer := range network.Layers() {
		fmt.Printf("Layer %d: %d neurons\n", i+1, layer.Size())
	}
	fmt.Println()

	// Initialize weights with random values
	weights := network.Weights()
	for i := range weights {
		for j := range weights[i] {
			for k := range weights[i][j] {
				weights[i][j][k] = rand.Float32()*2 - 1 // Random value between -1 and 1
			}
		}
	}

	// Print initial weights
	fmt.Println("Weights between layers:")
	for i := range weights {
		fmt.Printf("\nLayer %d -> Layer %d:\n", i+1, i+2)
		for j := range weights[i] {
			fmt.Printf("  Neuron %d: %v\n", j, weights[i][j])
		}
	}
	fmt.Println()

	// Test data
	input := []float32{1.0, 0.5, 0.25}
	labels := []float32{0.8, 0.2} // Target output
	learningRate := float32(1)

	fmt.Printf("Input: %v\n", input)
	fmt.Printf("Target labels: %v\n\n", labels)

	// Step 1: Forward pass
	fmt.Println("=== FORWARD PASS ===")
	network.ForwardPass(input)

	// Print all layer activations
	fmt.Println("Activations after forward pass:")
	for i, layer := range network.Layers() {
		fmt.Printf("Layer %d:\n", i+1)
		for j, neuron := range layer.Neurons() {
			fmt.Printf("  Neuron %d - Activation: %.4f\n", j, neuron.Activation())
		}
	}
	fmt.Println()

	// Calculate loss before backprop
	outputLayer := network.Layers()[len(network.Layers())-1]
	lossBefore := float32(0.0)
	for i := 0; i < len(labels); i++ {
		neuron := outputLayer.Neurons()[i]
		diff := neuron.Activation() - labels[i]
		lossBefore += diff * diff
	}
	lossBefore /= float32(len(labels))
	fmt.Printf("Loss before backprop (MSE): %.6f\n\n", lossBefore)

	// Step 2: Backpropagation
	fmt.Println("=== BACKPROPAGATION ===")
	fmt.Printf("Learning rate: %.4f\n", learningRate)
	network.BackPropagation(labels, learningRate)

	// Print gradients after backprop
	fmt.Println("\nGradients after backpropagation:")
	for i, layer := range network.Layers() {
		fmt.Printf("Layer %d:\n", i+1)
		for j, neuron := range layer.Neurons() {
			fmt.Printf("  Neuron %d - Gradient: %.4f\n", j, neuron.Gradient())
		}
	}
	fmt.Println()

	// Step 3: Forward pass again to see updated output
	fmt.Println("=== AFTER ONE ITERATION ===")
	network.ForwardPass(input)

	fmt.Println("Output layer activations after one iteration:")
	for i := 0; i < len(labels); i++ {
		neuron := outputLayer.Neurons()[i]
		fmt.Printf("  Neuron %d - Activation: %.4f (target: %.1f)\n", i, neuron.Activation(), labels[i])
	}
	fmt.Println()

	// Calculate loss after backprop
	lossAfter := float32(0.0)
	for i := 0; i < len(labels); i++ {
		neuron := outputLayer.Neurons()[i]
		diff := neuron.Activation() - labels[i]
		lossAfter += diff * diff
	}
	lossAfter /= float32(len(labels))
	fmt.Printf("Loss after one iteration (MSE): %.6f\n", lossAfter)
	fmt.Printf("Loss change: %.6f\n", lossBefore-lossAfter)
}
