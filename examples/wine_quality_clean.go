package main

import (
	"fmt"
	"neugo/Network"
	"neugo/data"
	"strings"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                                                                ║")
	fmt.Println("║                    🍷  WINE QUALITY CLASSIFIER 🍷                              ║")
	fmt.Println("║                  Using NeuGo with Data Utilities                              ║")
	fmt.Println("║                                                                                ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════╝")

	// Load data using data utilities
	fmt.Println("\n📂 Loading dataset...")
	dataset, err := data.QuickLoadBinaryCSV("dataset/wine_quality/winequality-red.csv", ';', 6.0)
	if err != nil {
		fmt.Println("❌ Error loading data:", err)
		return
	}

	fmt.Printf("   ✓ Total samples: %d\n", dataset.NumSamples)
	fmt.Printf("   ✓ Features: %d\n", dataset.NumFeatures)

	// Analyze class distribution
	fmt.Println("\n📊 Class Distribution:")
	dist := data.AnalyzeClassDistribution(dataset.Labels, 0.5)
	fmt.Printf("   Bad/Average Wine (≤6):  %d (%.2f%%)\n", dist.Counts[0], dist.Percentages[0])
	fmt.Printf("   Good Wine (>6):         %d (%.2f%%)\n", dist.Counts[1], dist.Percentages[1])
	if !dist.IsBalanced {
		fmt.Println("   ⚠️  Dataset is imbalanced")
	}

	// Normalize data
	fmt.Println("\n🔧 Preprocessing data...")
	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)
	fmt.Println("   ✓ Data normalized (z-score normalization)")

	// Split data
	fmt.Println("\n✂️  Splitting data...")
	split := data.SplitData(normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7,
		ValRatio:   0.15,
		TestRatio:  0.15,
		Shuffle:    true,
		Seed:       42,
	})

	fmt.Printf("   Training set: %d samples\n", len(split.TrainX))
	fmt.Printf("   Validation set: %d samples\n", len(split.ValX))
	fmt.Printf("   Test set: %d samples\n", len(split.TestX))

	// Balance training data
	fmt.Println("\n⚖️  Balancing training data...")
	trainX, trainY := data.OversampleMinorityClass(
		split.TrainX, split.TrainY,
		data.OversampleConfig{
			TargetRatio: 0.4,
			Strategy:    "duplicate",
			Seed:        42,
		},
	)

	distAfter := data.AnalyzeClassDistribution(trainY, 0.5)
	fmt.Printf("   After oversampling: %d samples (%.1f%% good wine)\n",
		len(trainY), distAfter.Percentages[1])

	// Build network
	fmt.Println("\n🏗️  Building neural network...")
	layers := []Network.Layer{
		Network.NewLayerWithActivation(dataset.NumFeatures, Network.Linear),
		Network.NewLayerWithActivation(16, Network.ReLU),
		Network.NewLayerWithActivation(8, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)

	fmt.Printf("   Architecture: %d → 16 → 8 → 1\n", dataset.NumFeatures)
	fmt.Println("   Activation: ReLU (hidden), Sigmoid (output)")
	fmt.Println("   Loss: Binary Cross-Entropy")

	// Training configuration
	fmt.Println("\n⚙️  Training Configuration:")
	fmt.Println("   ├─ Optimizer: SGD")
	fmt.Println("   ├─ Initial LR: 0.1")
	fmt.Println("   ├─ Scheduler: Cosine Annealing (0.1 → 0.001)")
	fmt.Println("   ├─ L2 Regularization: 0.001")
	fmt.Println("   ├─ Dropout: 0.2 (20%)")
	fmt.Println("   ├─ Batch Size: 32")
	fmt.Println("   ├─ Epochs: 150")
	fmt.Println("   └─ Early Stopping: Patience 20")

	// Setup training components
	scheduler := Network.NewCosineAnnealing(0.1, 0.001, 150)
	earlyStopping := Network.NewEarlyStopping(20, 0.0001)

	// Training loop
	fmt.Println("\n🏋️  Training...")
	fmt.Println("\n   Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Val F1   |    LR")
	fmt.Println("   -------|--------------|------------|-----------|-----------|----------")

	batchSize := 32
	bestValF1 := float32(0.0)

	for epoch := 0; epoch < 150; epoch++ {
		lr := scheduler.GetLearningRate(epoch)

		// Train on batches
		epochLoss := float32(0.0)
		numBatches := 0

		for i := 0; i < len(trainX); i += batchSize {
			end := i + batchSize
			if end > len(trainX) {
				end = len(trainX)
			}

			batchInputs := trainX[i:end]
			batchLabels := trainY[i:end]

			loss := network.TrainBatchWithRegularization(batchInputs, batchLabels, lr, 0.001, 0.2)
			epochLoss += loss
			numBatches++
		}

		avgTrainLoss := epochLoss / float32(numBatches)

		// Validation
		valMetrics := network.Evaluate(split.ValX, split.ValY, 0.5)

		// Early stopping
		earlyStopping.Update(valMetrics.Loss, &network)

		// Track best F1
		if valMetrics.F1Score > bestValF1 {
			bestValF1 = valMetrics.F1Score
		}

		// Print progress
		if epoch%10 == 0 || epoch == 149 || earlyStopping.ShouldStop {
			fmt.Printf("   %5d  |   %.6f   |  %.6f  |  %6.2f%%  |  %.4f    | %.6f\n",
				epoch+1, avgTrainLoss, valMetrics.Loss, valMetrics.Accuracy, valMetrics.F1Score, lr)
		}

		if earlyStopping.ShouldStop {
			fmt.Println("\n   🛑 Early stopping triggered!")
			earlyStopping.RestoreBestWeights(&network)
			break
		}

		scheduler.Step()
	}

	// Final evaluation on test set
	fmt.Println("\n" + strings.Repeat("─", 80))
	fmt.Println("\n📊 FINAL TEST RESULTS")
	fmt.Println(strings.Repeat("─", 80))

	testMetrics := network.Evaluate(split.TestX, split.TestY, 0.5)

	fmt.Println("\n📈 Classification Metrics:")
	fmt.Println("   ┌─────────────────────────────────────┐")
	fmt.Printf("   │ Accuracy:    %6.2f%%              │\n", testMetrics.Accuracy)
	fmt.Printf("   │ Precision:   %6.4f                │\n", testMetrics.Precision)
	fmt.Printf("   │ Recall:      %6.4f                │\n", testMetrics.Recall)
	fmt.Printf("   │ F1 Score:    %6.4f                │\n", testMetrics.F1Score)
	fmt.Printf("   │ Loss:        %6.4f                │\n", testMetrics.Loss)
	fmt.Println("   └─────────────────────────────────────┘")

	fmt.Println("\n📋 Confusion Matrix:")
	fmt.Println("   ┌───────────────────────────────┐")
	fmt.Println("   │         Predicted             │")
	fmt.Println("   │   Bad/Avg    Good             │")
	fmt.Println("   ├───────────────────────────────┤")
	for i, row := range testMetrics.ConfusionMatrix {
		if i == 0 {
			fmt.Printf("   │ B/A │  %5d    %5d        │\n", row[0], row[1])
		} else {
			fmt.Printf("   │  G  │  %5d    %5d        │\n", row[0], row[1])
		}
	}
	fmt.Println("   └───────────────────────────────┘")

	// Calculate additional metrics
	tn := testMetrics.ConfusionMatrix[0][0]
	fp := testMetrics.ConfusionMatrix[0][1]
	fn := testMetrics.ConfusionMatrix[1][0]
	tp := testMetrics.ConfusionMatrix[1][1]

	fmt.Println("\n📌 Detailed Metrics:")
	fmt.Printf("   True Negatives:  %d\n", tn)
	fmt.Printf("   False Positives: %d\n", fp)
	fmt.Printf("   False Negatives: %d\n", fn)
	fmt.Printf("   True Positives:  %d\n", tp)

	// Save model
	fmt.Println("\n💾 Saving trained model...")
	err = network.SaveToFile("wine_quality_model.json")
	if err != nil {
		fmt.Println("   ❌ Error saving model:", err)
	} else {
		fmt.Println("   ✓ Model saved to: wine_quality_model.json")
	}

	// Demonstrate predictions on a few test samples
	fmt.Println("\n🔮 Sample Predictions:")
	fmt.Println("   ┌────────────────────────────────────────────┐")
	for i := 0; i < 5 && i < len(split.TestX); i++ {
		network.ForwardPass(split.TestX[i])
		prediction := network.GetOutput()[0].Activation()
		actual := split.TestY[i][0]

		predictedClass := "Bad/Avg"
		actualClass := "Bad/Avg"
		if prediction > 0.5 {
			predictedClass = "Good"
		}
		if actual > 0.5 {
			actualClass = "Good"
		}

		correct := "✓"
		if (prediction > 0.5) != (actual > 0.5) {
			correct = "✗"
		}

		fmt.Printf("   │ Sample %d: %.2f%% → %-7s (Actual: %-7s) %s │\n",
			i+1, prediction*100, predictedClass, actualClass, correct)
	}
	fmt.Println("   └────────────────────────────────────────────┘")

	fmt.Println("\n" + strings.Repeat("═", 80))
	fmt.Println("\n✅ Wine Quality Classification Complete!")
	fmt.Println("\n   This example demonstrates the data utilities package:")
	fmt.Println("   • QuickLoadBinaryCSV for easy CSV loading")
	fmt.Println("   • AnalyzeClassDistribution for balance analysis")
	fmt.Println("   • NormalizeZScore for feature scaling")
	fmt.Println("   • SplitData for train/val/test splitting")
	fmt.Println("   • OversampleMinorityClass for handling imbalance")
	fmt.Println("\n" + strings.Repeat("═", 80))
}
