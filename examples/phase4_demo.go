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
	fmt.Println("║                      🚀  PHASE 4 FEATURES DEMO  🚀                            ║")
	fmt.Println("║          Callbacks, History, Checkpointing & Cross-Validation                ║")
	fmt.Println("║                                                                                ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════╝")

	// Load and prepare data
	fmt.Println("\n📂 Loading Wine Quality Dataset...")
	dataset, err := data.QuickLoadBinaryCSV("dataset/wine_quality/winequality-red.csv", ';', 6.0)
	if err != nil {
		fmt.Println("❌ Error loading data:", err)
		return
	}

	fmt.Printf("   ✓ Loaded %d samples with %d features\n", dataset.NumSamples, dataset.NumFeatures)

	// Preprocess
	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	// Split data
	split := data.SplitData(normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7,
		ValRatio:   0.15,
		TestRatio:  0.15,
		Shuffle:    true,
		Seed:       42,
	})

	// Balance training data
	trainX, trainY := data.OversampleMinorityClass(
		split.TrainX, split.TrainY,
		data.OversampleConfig{TargetRatio: 0.4, Strategy: "duplicate", Seed: 42},
	)

	fmt.Printf("   ✓ Training: %d, Validation: %d, Test: %d\n",
		len(trainX), len(split.ValX), len(split.TestX))

	// ═══════════════════════════════════════════════════════════════════
	// DEMO 1: Training with Callbacks
	// ═══════════════════════════════════════════════════════════════════

	fmt.Println("\n" + strings.Repeat("═", 80))
	fmt.Println("DEMO 1: Training with Callbacks")
	fmt.Println(strings.Repeat("═", 80))

	// Build network
	createNetwork := func() Network.Network {
		layers := []Network.Layer{
			Network.NewLayerWithActivation(dataset.NumFeatures, Network.Linear),
			Network.NewLayerWithActivation(16, Network.ReLU),
			Network.NewLayerWithActivation(8, Network.ReLU),
			Network.NewLayerWithActivation(1, Network.Sigmoid),
		}
		return Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)
	}

	network := createNetwork()

	// Setup callbacks
	history := Network.NewHistory()
	checkpoint := Network.NewModelCheckpoint(
		"best_wine_model.json",
		"f1",      // Monitor F1 score
		"max",     // Maximize F1
		true,      // Save best only
		true,      // Verbose
	)
	progress := Network.NewProgressBar(50, 10, true)

	// Custom callback example
	customCallback := Network.NewCustomCallback()
	customCallback.OnEpochEndFunc = func(epoch int, net *Network.Network, metrics *Network.Metrics) {
		if epoch == 25 {
			fmt.Println("   🎯 Halfway through training!")
		}
	}

	callbacks := Network.NewCallbackList(history, checkpoint, progress, customCallback)

	// Training configuration
	fmt.Println("\n⚙️  Training Configuration:")
	fmt.Println("   ├─ Epochs: 50")
	fmt.Println("   ├─ Batch Size: 32")
	fmt.Println("   ├─ Learning Rate: 0.1 → 0.001 (Cosine)")
	fmt.Println("   ├─ Callbacks: History, Checkpoint, Progress, Custom")
	fmt.Println("   └─ Early Stopping: Patience 15")

	config := &Network.TrainingConfig{
		Epochs:           50,
		BatchSize:        32,
		LearningRate:     0.1,
		L2Lambda:         0.001,
		DropoutRate:      0.2,
		Scheduler:        Network.NewCosineAnnealing(0.1, 0.001, 50),
		EarlyStopping:    Network.NewEarlyStopping(15, 0.0001),
		Callbacks:        callbacks,
		ValidationData:   split.ValX,
		ValidationLabels: split.ValY,
		Threshold:        0.5,
		Verbose:          false, // Progress callback handles output
	}

	fmt.Println("\n🏋️  Training with Callbacks...")
	trainedHistory := network.Train(trainX, trainY, config)

	// Display training history
	fmt.Println("\n📊 Training History Summary:")
	fmt.Printf("   Total Duration: %v\n", trainedHistory.Duration())
	fmt.Printf("   Total Epochs: %d\n", len(trainedHistory.Epochs))
	fmt.Printf("   Initial Train Loss: %.4f\n", trainedHistory.TrainLoss[0])
	fmt.Printf("   Final Train Loss: %.4f\n", trainedHistory.TrainLoss[len(trainedHistory.TrainLoss)-1])
	if len(trainedHistory.ValAcc) > 0 {
		fmt.Printf("   Best Val Accuracy: %.2f%%\n", findMax(trainedHistory.ValAcc))
		fmt.Printf("   Best Val F1 Score: %.4f\n", findMax(trainedHistory.ValF1))
	}

	// ═══════════════════════════════════════════════════════════════════
	// DEMO 2: Cross-Validation
	// ═══════════════════════════════════════════════════════════════════

	fmt.Println("\n" + strings.Repeat("═", 80))
	fmt.Println("DEMO 2: K-Fold Cross-Validation")
	fmt.Println(strings.Repeat("═", 80))

	// Use a subset for faster cross-validation demo
	subsetSize := 500
	cvX := normalized[:subsetSize]
	cvY := dataset.Labels[:subsetSize]

	fmt.Printf("\nUsing subset of %d samples for faster demo\n", subsetSize)

	// Setup CV configuration
	cvConfig := &Network.TrainingConfig{
		Epochs:       30,
		BatchSize:    32,
		LearningRate: 0.1,
		L2Lambda:     0.001,
		DropoutRate:  0.2,
		Scheduler:    Network.NewCosineAnnealing(0.1, 0.001, 30),
		Threshold:    0.5,
		Verbose:      false,
	}

	// Standard K-Fold
	fmt.Println("\n📊 Standard 5-Fold Cross-Validation:")
	cvResult := Network.CrossValidate(
		createNetwork,
		cvX, cvY,
		5, // K=5 folds
		cvConfig,
		true, // Verbose
	)

	// Stratified K-Fold
	fmt.Println("\n📊 Stratified 5-Fold Cross-Validation:")
	stratifiedResult := Network.CrossValidateStratified(
		createNetwork,
		cvX, cvY,
		5,
		cvConfig,
		true,
	)

	// ═══════════════════════════════════════════════════════════════════
	// DEMO 3: Final Evaluation on Test Set
	// ═══════════════════════════════════════════════════════════════════

	fmt.Println("\n" + strings.Repeat("═", 80))
	fmt.Println("DEMO 3: Final Test Set Evaluation")
	fmt.Println(strings.Repeat("═", 80))

	// Load best model from checkpoint
	bestNetwork, err := Network.LoadFromFile("best_wine_model.json")
	if err != nil {
		fmt.Println("\n⚠️  Using trained model (checkpoint not available)")
		bestNetwork = network
	} else {
		fmt.Println("\n✓ Loaded best model from checkpoint")
	}

	testMetrics := bestNetwork.Evaluate(split.TestX, split.TestY, 0.5)

	fmt.Println("\n📈 Test Set Performance:")
	fmt.Println("   ┌─────────────────────────────────────┐")
	fmt.Printf("   │ Accuracy:    %6.2f%%              │\n", testMetrics.Accuracy)
	fmt.Printf("   │ Precision:   %6.4f                │\n", testMetrics.Precision)
	fmt.Printf("   │ Recall:      %6.4f                │\n", testMetrics.Recall)
	fmt.Printf("   │ F1 Score:    %6.4f                │\n", testMetrics.F1Score)
	fmt.Printf("   │ Loss:        %6.4f                │\n", testMetrics.Loss)
	fmt.Println("   └─────────────────────────────────────┘")

	// ═══════════════════════════════════════════════════════════════════
	// Summary
	// ═══════════════════════════════════════════════════════════════════

	fmt.Println("\n" + strings.Repeat("═", 80))
	fmt.Println("📋 PHASE 4 FEATURES SUMMARY")
	fmt.Println(strings.Repeat("═", 80))

	fmt.Println("\n✅ Demonstrated Features:")
	fmt.Println("   1. Callbacks System:")
	fmt.Println("      • History tracking (loss, accuracy, F1 over time)")
	fmt.Println("      • Model checkpointing (auto-save best model)")
	fmt.Println("      • Progress bar (training feedback)")
	fmt.Println("      • Custom callbacks (user-defined behavior)")
	fmt.Println()
	fmt.Println("   2. Training Configuration:")
	fmt.Println("      • Single config object for all parameters")
	fmt.Println("      • Easy integration with schedulers & early stopping")
	fmt.Println("      • Built-in validation during training")
	fmt.Println()
	fmt.Println("   3. Cross-Validation:")
	fmt.Println("      • Standard K-Fold cross-validation")
	fmt.Println("      • Stratified K-Fold (preserves class distribution)")
	fmt.Println("      • Automatic statistics (mean, std dev)")
	fmt.Println("      • Fold-by-fold performance tracking")

	fmt.Println("\n📊 Performance Comparison:")
	fmt.Println("   ┌──────────────────────────────────────────────────────┐")
	fmt.Printf("   │ Single Training:         Acc: %.2f%%, F1: %.4f      │\n",
		testMetrics.Accuracy, testMetrics.F1Score)
	fmt.Printf("   │ Standard CV (5-fold):    Acc: %.2f%% ± %.2f%%        │\n",
		cvResult.MeanAccuracy, cvResult.StdAccuracy)
	fmt.Printf("   │ Stratified CV (5-fold):  Acc: %.2f%% ± %.2f%%        │\n",
		stratifiedResult.MeanAccuracy, stratifiedResult.StdAccuracy)
	fmt.Println("   └──────────────────────────────────────────────────────┘")

	fmt.Println("\n" + strings.Repeat("═", 80))
	fmt.Println("✅ Phase 4 Demo Complete!")
	fmt.Println(strings.Repeat("═", 80))
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
