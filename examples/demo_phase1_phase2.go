package main

import (
	"fmt"
	"neugo/Network"
)

func main() {
	fmt.Println("=== NeuGo Phase 1 & 2 Features Demo ===\n")

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

	// ============ PHASE 1 FEATURES ============

	fmt.Println("--- Phase 1: Core Functionality ---\n")

	// Feature 1: Batch Training
	fmt.Println("1. Batch Training")
	layer1 := Network.NewLayerWithActivation(2, Network.Sigmoid)
	layer2 := Network.NewLayerWithActivation(4, Network.Sigmoid)
	layer3 := Network.NewLayerWithActivation(1, Network.Sigmoid)
	layers := []Network.Layer{layer1, layer2, layer3}
	network := Network.NewNetwork(layers)

	losses := network.Fit(inputs, labels, 1000, 2, 0.5, false)
	fmt.Printf("   Trained for 1000 epochs with batch size 2\n")
	fmt.Printf("   Initial loss: %.6f, Final loss: %.6f\n\n", losses[0], losses[len(losses)-1])

	// Feature 2: Model Serialization
	fmt.Println("2. Model Serialization")
	err := network.SaveToFile("trained_model.json")
	if err != nil {
		fmt.Println("   Error saving model:", err)
	} else {
		fmt.Println("   ✓ Model saved to trained_model.json")
	}

	loadedNetwork, err := Network.LoadFromFile("trained_model.json")
	if err != nil {
		fmt.Println("   Error loading model:", err)
	} else {
		fmt.Println("   ✓ Model loaded from trained_model.json")
		loadedNetwork.ForwardPass(inputs[0])
		output := loadedNetwork.GetOutput()[0].Activation()
		fmt.Printf("   Loaded model prediction for [0,0]: %.4f\n\n", output)
	}

	// Feature 3: Validation & Metrics
	fmt.Println("3. Validation & Metrics")
	metrics := network.Evaluate(inputs, labels, 0.5)
	fmt.Printf("   Accuracy: %.2f%%\n", metrics.Accuracy)
	fmt.Printf("   Precision: %.4f\n", metrics.Precision)
	fmt.Printf("   Recall: %.4f\n", metrics.Recall)
	fmt.Printf("   F1 Score: %.4f\n", metrics.F1Score)
	fmt.Println("   Confusion Matrix:")
	for i, row := range metrics.ConfusionMatrix {
		fmt.Printf("     %v\n", row)
		_ = i
	}
	fmt.Println()

	// ============ PHASE 2 FEATURES ============

	fmt.Println("--- Phase 2: Advanced Training ---\n")

	// Feature 4: Adam Optimizer
	fmt.Println("4. Adam Optimizer")
	layer1Adam := Network.NewLayerWithActivation(2, Network.ReLU)
	layer2Adam := Network.NewLayerWithActivation(4, Network.ReLU)
	layer3Adam := Network.NewLayerWithActivation(1, Network.Sigmoid)
	layersAdam := []Network.Layer{layer1Adam, layer2Adam, layer3Adam}
	networkAdam := Network.NewNetworkWithLoss(layersAdam, Network.BinaryCrossEntropy)

	// Note: Full Adam integration would require modifying TrainBatch
	// For now, demonstrating the creation
	adam := Network.NewAdam(0.001, 0.9, 0.999, 1e-8)
	networkAdam.SetOptimizer(adam)
	fmt.Println("   ✓ Adam optimizer created and attached")
	fmt.Printf("   Learning rate: %.4f, Beta1: %.2f, Beta2: %.3f\n\n",
		adam.GetLearningRate(), adam.Beta1, adam.Beta2)

	// Feature 5: Regularization
	fmt.Println("5. Regularization (L2 + Dropout)")
	layer1Reg := Network.NewLayerWithActivation(2, Network.ReLU)
	layer2Reg := Network.NewLayerWithActivation(8, Network.ReLU)
	layer3Reg := Network.NewLayerWithActivation(1, Network.Sigmoid)
	layersReg := []Network.Layer{layer1Reg, layer2Reg, layer3Reg}
	networkReg := Network.NewNetwork(layersReg)

	// Train with L2 regularization and dropout
	l2Lambda := float32(0.01)
	dropoutRate := float32(0.2)

	epochLoss := float32(0.0)
	for i := 0; i < 100; i++ {
		loss := networkReg.TrainBatchWithRegularization(inputs, labels, 0.1, l2Lambda, dropoutRate)
		epochLoss = loss
	}
	fmt.Printf("   Trained with L2 (lambda=%.3f) and Dropout (rate=%.2f)\n", l2Lambda, dropoutRate)
	fmt.Printf("   Final loss: %.6f\n\n", epochLoss)

	// Feature 6: Learning Rate Scheduling
	fmt.Println("6. Learning Rate Scheduling")

	// Step Decay
	fmt.Println("   a) Step Decay Scheduler")
	layer1Sched := Network.NewLayerWithActivation(2, Network.Sigmoid)
	layer2Sched := Network.NewLayerWithActivation(4, Network.Sigmoid)
	layer3Sched := Network.NewLayerWithActivation(1, Network.Sigmoid)
	layersSched := []Network.Layer{layer1Sched, layer2Sched, layer3Sched}
	networkSched := Network.NewNetwork(layersSched)

	stepScheduler := Network.NewStepDecay(0.5, 0.5, 200)
	lossesStep := networkSched.FitWithScheduler(inputs, labels, 500, 2, stepScheduler, false)
	fmt.Printf("      LR starts at 0.5, decays by 0.5 every 200 epochs\n")
	fmt.Printf("      Initial loss: %.6f, Final loss: %.6f\n", lossesStep[0], lossesStep[len(lossesStep)-1])

	// Exponential Decay
	fmt.Println("   b) Exponential Decay Scheduler")
	networkSched2 := Network.NewNetwork(layersSched)
	expScheduler := Network.NewExponentialDecay(0.5, 0.99)
	lossesExp := networkSched2.FitWithScheduler(inputs, labels, 500, 2, expScheduler, false)
	fmt.Printf("      LR decays exponentially from 0.5 by 0.99 each epoch\n")
	fmt.Printf("      Initial loss: %.6f, Final loss: %.6f\n", lossesExp[0], lossesExp[len(lossesExp)-1])

	// Cosine Annealing
	fmt.Println("   c) Cosine Annealing Scheduler")
	networkSched3 := Network.NewNetwork(layersSched)
	cosineScheduler := Network.NewCosineAnnealing(0.5, 0.001, 500)
	lossesCosine := networkSched3.FitWithScheduler(inputs, labels, 500, 2, cosineScheduler, false)
	fmt.Printf("      LR anneals from 0.5 to 0.001 using cosine schedule\n")
	fmt.Printf("      Initial loss: %.6f, Final loss: %.6f\n\n", lossesCosine[0], lossesCosine[len(lossesCosine)-1])

	// ============ COMBINED EXAMPLE ============

	fmt.Println("--- Combined Example: All Features Together ---\n")

	// Create network with all best practices
	inputLayer := Network.NewLayerWithActivation(2, Network.Linear)
	hiddenLayer1 := Network.NewLayerWithActivation(6, Network.ReLU)
	hiddenLayer2 := Network.NewLayerWithActivation(4, Network.ReLU)
	outputLayer := Network.NewLayerWithActivation(1, Network.Sigmoid)

	finalLayers := []Network.Layer{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer}
	finalNetwork := Network.NewNetworkWithLoss(finalLayers, Network.BinaryCrossEntropy)

	// Train with all features
	fmt.Println("Training XOR with:")
	fmt.Println("  - Batch training (size: 2)")
	fmt.Println("  - L2 regularization (0.001)")
	fmt.Println("  - Dropout (0.1)")
	fmt.Println("  - Cosine annealing LR schedule")
	fmt.Println()

	scheduler := Network.NewCosineAnnealing(0.5, 0.01, 2000)

	for epoch := 0; epoch < 2000; epoch++ {
		lr := scheduler.GetLearningRate(epoch)
		loss := finalNetwork.TrainBatchWithRegularization(inputs, labels, lr, 0.001, 0.1)
		scheduler.Step()

		if (epoch+1)%500 == 0 {
			fmt.Printf("Epoch %d - Loss: %.6f, LR: %.4f\n", epoch+1, loss, lr)
		}
	}

	// Final evaluation
	fmt.Println("\nFinal Evaluation:")
	finalMetrics := finalNetwork.Evaluate(inputs, labels, 0.5)
	fmt.Printf("  Accuracy: %.2f%%\n", finalMetrics.Accuracy)
	fmt.Printf("  Loss: %.6f\n\n", finalMetrics.Loss)

	// Save final model
	finalNetwork.SaveToFile("xor_final_model.json")
	fmt.Println("✓ Final model saved to xor_final_model.json")

	fmt.Println("\n=== Demo Complete ===")
}
