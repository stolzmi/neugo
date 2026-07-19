package main

import (
	"fmt"
	"neugo/Network"
	"neugo/data"
)

func main() {
	fmt.Println("🔍 Debugging Training Performance")
	fmt.Println("==================================")

	// Load and prepare data
	dataset, err := data.QuickLoadBinaryCSV("dataset/wine_quality/winequality-red.csv", ';', 6.0)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Preprocess
	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	// Split
	split := data.SplitData(normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7,
		ValRatio:   0.15,
		TestRatio:  0.15,
		Shuffle:    true,
		Seed:       42,
	})

	// Balance
	trainX, trainY := data.OversampleMinorityClass(
		split.TrainX, split.TrainY,
		data.OversampleConfig{TargetRatio: 0.4, Strategy: "duplicate", Seed: 42},
	)

	fmt.Printf("Training samples: %d\n", len(trainX))
	fmt.Printf("Validation samples: %d\n", len(split.ValX))

	// Simple network
	layers := []Network.Layer{
		Network.NewLayerWithActivation(dataset.NumFeatures, Network.Linear),
		Network.NewLayerWithActivation(16, Network.ReLU),
		Network.NewLayerWithActivation(8, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)

	fmt.Println("\n📊 Training for 100 epochs with detailed logging...")
	fmt.Println("Epoch | Train Loss | Val Loss | Val Acc | Val F1 | LR")
	fmt.Println("------|------------|----------|---------|--------|--------")

	scheduler := Network.NewCosineAnnealing(0.1, 0.001, 100)
	batchSize := 32

	for epoch := 0; epoch < 100; epoch++ {
		lr := scheduler.GetLearningRate(epoch)

		// Train
		epochLoss := float32(0.0)
		numBatches := 0

		for i := 0; i < len(trainX); i += batchSize {
			end := i + batchSize
			if end > len(trainX) {
				end = len(trainX)
			}

			loss := network.TrainBatchWithRegularization(trainX[i:end], trainY[i:end], lr, 0.001, 0.2)
			epochLoss += loss
			numBatches++
		}

		avgTrainLoss := epochLoss / float32(numBatches)

		// Validate
		valMetrics := network.Evaluate(split.ValX, split.ValY, 0.5)

		// Print every 5 epochs
		if epoch%5 == 0 || epoch == 99 {
			fmt.Printf("%5d | %10.6f | %8.6f | %6.2f%% | %6.4f | %.6f\n",
				epoch+1, avgTrainLoss, valMetrics.Loss, valMetrics.Accuracy, valMetrics.F1Score, lr)
		}

		scheduler.Step()
	}

	// Final test
	testMetrics := network.Evaluate(split.TestX, split.TestY, 0.5)
	fmt.Printf("\n🎯 Final Test Results:\n")
	fmt.Printf("   Accuracy: %.2f%%\n", testMetrics.Accuracy)
	fmt.Printf("   F1 Score: %.4f\n", testMetrics.F1Score)
	fmt.Printf("   Precision: %.4f\n", testMetrics.Precision)
	fmt.Printf("   Recall: %.4f\n", testMetrics.Recall)

	// Detailed prediction analysis
	fmt.Println("\n🔍 Prediction Analysis:")

	// Show samples from each class
	class0Count := 0
	class1Count := 0

	fmt.Println("\nClass 0 samples (bad wine):")
	for i := 0; i < len(split.TestX) && class0Count < 5; i++ {
		if split.TestY[i][0] == 0 {
			network.ForwardPass(split.TestX[i])
			pred := network.GetOutput()[0].Activation()
			predicted := float32(0)
			if pred >= 0.5 {
				predicted = 1
			}
			correct := ""
			if predicted == split.TestY[i][0] {
				correct = "✓"
			} else {
				correct = "✗"
			}
			fmt.Printf("   %s pred=%.4f, actual=%.0f\n", correct, pred, split.TestY[i][0])
			class0Count++
		}
	}

	fmt.Println("\nClass 1 samples (good wine):")
	for i := 0; i < len(split.TestX) && class1Count < 5; i++ {
		if split.TestY[i][0] == 1 {
			network.ForwardPass(split.TestX[i])
			pred := network.GetOutput()[0].Activation()
			predicted := float32(0)
			if pred >= 0.5 {
				predicted = 1
			}
			correct := ""
			if predicted == split.TestY[i][0] {
				correct = "✓"
			} else {
				correct = "✗"
			}
			fmt.Printf("   %s pred=%.4f, actual=%.0f\n", correct, pred, split.TestY[i][0])
			class1Count++
		}
	}

	// Class distribution in test set
	class0Total := 0
	class1Total := 0
	for i := 0; i < len(split.TestY); i++ {
		if split.TestY[i][0] == 0 {
			class0Total++
		} else {
			class1Total++
		}
	}
	fmt.Printf("\nTest set distribution: Class 0 = %d, Class 1 = %d\n", class0Total, class1Total)
}
