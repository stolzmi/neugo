package main

import (
	"fmt"
	"neugo/Network"
	"neugo/data"
	"time"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          🎯  NEUGO COMPLETE FEATURE SHOWCASE  🎯              ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	dataset, err := data.QuickLoadBinaryCSV("dataset/wine_quality/winequality-red.csv", ';', 6.0)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Printf("\n📂 Dataset: %d samples, %d features\n", dataset.NumSamples, dataset.NumFeatures)

	dist := data.AnalyzeClassDistribution(dataset.Labels, 0.5)
	fmt.Printf("Class 0: %d (%.1f%%), Class 1: %d (%.1f%%)\n",
		dist.Counts[0], dist.Percentages[0], dist.Counts[1], dist.Percentages[1])

	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	split := data.SplitData(normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7,
		ValRatio:   0.15,
		TestRatio:  0.15,
		Shuffle:    true,
		Seed:       42,
	})

	trainX, trainY := data.OversampleMinorityClass(
		split.TrainX, split.TrainY,
		data.OversampleConfig{TargetRatio: 0.4, Strategy: "duplicate", Seed: 42},
	)
	fmt.Printf("Train: %d, Val: %d, Test: %d\n", len(trainX), len(split.ValX), len(split.TestX))

	layers := []Network.Layer{
		Network.NewLayerWithActivation(dataset.NumFeatures, Network.Linear),
		Network.NewLayerWithActivation(32, Network.ReLU),
		Network.NewLayerWithActivation(16, Network.ReLU),
		Network.NewLayerWithActivation(8, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)
	fmt.Printf("\n🏗️  Architecture: %d → 32 → 16 → 8 → 1 (BCE Loss)\n", dataset.NumFeatures)

	history := Network.NewHistory()
	checkpoint := Network.NewModelCheckpoint("showcase_best_model.json", "f1", "max", true, true)
	callbacks := Network.NewCallbackList(history, checkpoint)

	fmt.Println("\n🏋️  Training 100 epochs...")
	startTime := time.Now()

	config := &Network.TrainingConfig{
		Epochs:           100,
		BatchSize:        32,
		LearningRate:     0.1,
		L2Lambda:         0.001,
		DropoutRate:      0.2,
		Scheduler:        Network.NewCosineAnnealing(0.1, 0.001, 100),
		Callbacks:        callbacks,
		ValidationData:   split.ValX,
		ValidationLabels: split.ValY,
		Threshold:        0.5,
		Verbose:          false,
	}

	trainedHistory := network.Train(trainX, trainY, config)
	duration := time.Since(startTime)

	fmt.Printf("\n📈 Training Results:\n")
	fmt.Printf("   Duration: %v\n", duration)
	fmt.Printf("   Initial loss: %.4f → Final loss: %.4f\n",
		trainedHistory.TrainLoss[0], trainedHistory.TrainLoss[len(trainedHistory.TrainLoss)-1])
	fmt.Printf("   Best val accuracy: %.2f%%\n", findMax(trainedHistory.ValAcc))
	fmt.Printf("   Best val F1: %.4f\n", findMax(trainedHistory.ValF1))

	cvSize := 400
	cvX := normalized[:cvSize]
	cvY := dataset.Labels[:cvSize]

	fmt.Printf("\n🔄 Running 5-Fold Cross-Validation (%d samples)...\n", cvSize)

	createNetwork := func() Network.Network {
		return Network.NewNetworkWithLoss(
			[]Network.Layer{
				Network.NewLayerWithActivation(dataset.NumFeatures, Network.Linear),
				Network.NewLayerWithActivation(16, Network.ReLU),
				Network.NewLayerWithActivation(8, Network.ReLU),
				Network.NewLayerWithActivation(1, Network.Sigmoid),
			},
			Network.BinaryCrossEntropy,
		)
	}

	cvConfig := &Network.TrainingConfig{
		Epochs:       20,
		BatchSize:    32,
		LearningRate: 0.1,
		L2Lambda:     0.001,
		DropoutRate:  0.2,
		Threshold:    0.5,
		Verbose:      false,
	}

	cvResult := Network.CrossValidate(createNetwork, cvX, cvY, 5, cvConfig, false)
	fmt.Printf("   Mean Accuracy: %.2f%% ± %.2f%%\n", cvResult.MeanAccuracy, cvResult.StdAccuracy)
	fmt.Printf("   Mean F1: %.4f ± %.4f\n", cvResult.MeanF1, cvResult.StdF1)

	stratResult := Network.CrossValidateStratified(createNetwork, cvX, cvY, 5, cvConfig, false)
	fmt.Printf("   Stratified Accuracy: %.2f%% ± %.2f%%\n", stratResult.MeanAccuracy, stratResult.StdAccuracy)

	err = network.SaveToFile("showcase_final_model.json")
	if err != nil {
		fmt.Println("Save error:", err)
	} else {
		fmt.Println("\n💾 Model saved to showcase_final_model.json")
	}

	loadedNetwork, err := Network.LoadFromFile("showcase_final_model.json")
	if err == nil {
		loadedMetrics := loadedNetwork.Evaluate(split.TestX, split.TestY, 0.5)
		fmt.Printf("📂 Loaded model test accuracy: %.2f%%\n", loadedMetrics.Accuracy)
	}

	testMetrics := network.Evaluate(split.TestX, split.TestY, 0.5)

	fmt.Println("\n🎯 Final Test Performance:")
	fmt.Println("   ┌──────────────────────────────────────┐")
	fmt.Printf("   │ Accuracy:      %6.2f%%              │\n", testMetrics.Accuracy)
	fmt.Printf("   │ Precision:     %6.4f                │\n", testMetrics.Precision)
	fmt.Printf("   │ Recall:        %6.4f                │\n", testMetrics.Recall)
	fmt.Printf("   │ F1 Score:      %6.4f                │\n", testMetrics.F1Score)
	fmt.Printf("   │ Loss:          %6.4f                │\n", testMetrics.Loss)
	fmt.Println("   └──────────────────────────────────────┘")

	fmt.Println("\n🔮 Sample Predictions:")
	for i := 0; i < 5 && i < len(split.TestX); i++ {
		network.ForwardPass(split.TestX[i])
		pred := network.GetOutput()[0].Activation()
		actual := split.TestY[i][0]

		predClass := "Bad"
		actualClass := "Bad"
		if pred > 0.5 {
			predClass = "Good"
		}
		if actual > 0.5 {
			actualClass = "Good"
		}

		correct := "✓"
		if (pred > 0.5) != (actual > 0.5) {
			correct = "✗"
		}

		fmt.Printf("   %s Sample %d: %.1f%% → %s (Actual: %s)\n",
			correct, i+1, pred*100, predClass, actualClass)
	}

	fmt.Println("\n✅ Features Demonstrated:")
	fmt.Println("   • Data loading & preprocessing (Phase 3)")
	fmt.Println("   • Class balancing & normalization (Phase 3)")
	fmt.Println("   • Network architecture & training (Core)")
	fmt.Println("   • Optimizers & regularization (Phase 2)")
	fmt.Println("   • Callbacks & checkpointing (Phase 4)")
	fmt.Println("   • Cross-validation (Phase 4)")
	fmt.Println("   • Model serialization (Phase 1)")
	fmt.Println("   • Comprehensive metrics (Phase 1)")

	fmt.Println("\n═════════════════════════════════════════════════════════════════")
	fmt.Println("✅ NEUGO SHOWCASE COMPLETE!")
	fmt.Println("═════════════════════════════════════════════════════════════════")
}

func findMax(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}
