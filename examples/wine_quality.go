package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"neugo/Network"
	"os"
	"strconv"
	"strings"
)

// DataStats holds statistics for normalization
type DataStats struct {
	Mean   []float32
	StdDev []float32
}

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                                                                ║")
	fmt.Println("║                    🍷  WINE QUALITY CLASSIFIER 🍷                              ║")
	fmt.Println("║                     Using NeuGo Neural Network                                ║")
	fmt.Println("║                                                                                ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	fmt.Println("\n📂 Loading dataset...")
	data, labels, err := loadWineData("dataset/wine_quality/winequality-red.csv")
	if err != nil {
		fmt.Println("❌ Error loading data:", err)
		return
	}

	fmt.Printf("   ✓ Total samples: %d\n", len(labels))
	fmt.Printf("   ✓ Features: %d\n", len(data[0]))

	// Check class distribution
	goodCount := 0
	for _, label := range labels {
		if label[0] > 0.5 {
			goodCount++
		}
	}
	fmt.Printf("\n📊 Class Distribution:\n")
	fmt.Printf("   Bad/Average Wine (≤6):  %d (%.2f%%)\n", len(labels)-goodCount, float64(len(labels)-goodCount)/float64(len(labels))*100)
	fmt.Printf("   Good Wine (>6):         %d (%.2f%%)\n", goodCount, float64(goodCount)/float64(len(labels))*100)

	// Normalize data
	fmt.Println("\n🔧 Preprocessing data...")
	stats := calculateStats(data)
	data = normalizeData(data, stats)
	fmt.Println("   ✓ Data normalized (z-score normalization)")

	// Split data: 70% train, 15% validation, 15% test
	fmt.Println("\n✂️  Splitting data...")
	trainSize := int(float64(len(data)) * 0.7)
	valSize := int(float64(len(data)) * 0.15)

	trainX := data[:trainSize]
	trainY := labels[:trainSize]
	valX := data[trainSize : trainSize+valSize]
	valY := labels[trainSize : trainSize+valSize]
	testX := data[trainSize+valSize:]
	testY := labels[trainSize+valSize:]

	fmt.Printf("   Training set: %d samples\n", len(trainX))
	fmt.Printf("   Validation set: %d samples\n", len(valX))
	fmt.Printf("   Test set: %d samples\n", len(testX))

	// Balance training data with oversampling
	fmt.Println("\n⚖️  Balancing training data...")
	trainX, trainY = oversampleMinorityClass(trainX, trainY)

	// Count good wines after balancing
	goodAfterBalance := 0
	for i := range trainY {
		if trainY[i][0] > 0.5 {
			goodAfterBalance++
		}
	}
	goodRatio := float32(goodAfterBalance) / float32(len(trainY)) * 100
	fmt.Printf("   After oversampling: %d samples (%.1f%% good wine)\n", len(trainY), goodRatio)

	// Build network
	fmt.Println("\n🏗️  Building neural network...")
	inputSize := len(trainX[0])

	layers := []Network.Layer{
		Network.NewLayerWithActivation(inputSize, Network.Linear),
		Network.NewLayerWithActivation(16, Network.ReLU),
		Network.NewLayerWithActivation(8, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)

	fmt.Printf("   Architecture: %d → 16 → 8 → 1\n", inputSize)
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
		valMetrics := network.Evaluate(valX, valY, 0.5)

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

	testMetrics := network.Evaluate(testX, testY, 0.5)

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
	for i := 0; i < 5 && i < len(testX); i++ {
		network.ForwardPass(testX[i])
		prediction := network.GetOutput()[0].Activation()
		actual := testY[i][0]

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
	fmt.Println("\n" + strings.Repeat("═", 80))
}

// loadWineData loads wine quality data from CSV
func loadWineData(path string) ([][]float32, [][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';' // Wine dataset uses semicolon delimiter

	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	// Skip header
	records = records[1:]

	features := make([][]float32, len(records))
	labels := make([][]float32, len(records))

	for i, record := range records {
		// Parse features (all columns except last)
		features[i] = make([]float32, len(record)-1)
		for j := 0; j < len(record)-1; j++ {
			val, _ := strconv.ParseFloat(record[j], 32)
			features[i][j] = float32(val)
		}

		// Parse label (last column)
		// Convert to binary: quality > 6 is "good" (1), otherwise "bad/average" (0)
		quality, _ := strconv.ParseFloat(record[len(record)-1], 32)
		if quality > 6 {
			labels[i] = []float32{1.0}
		} else {
			labels[i] = []float32{0.0}
		}
	}

	return features, labels, nil
}

// calculateStats calculates mean and standard deviation for each feature
func calculateStats(data [][]float32) DataStats {
	if len(data) == 0 {
		return DataStats{}
	}

	numFeatures := len(data[0])
	mean := make([]float32, numFeatures)
	stdDev := make([]float32, numFeatures)

	// Calculate mean
	for _, sample := range data {
		for j, val := range sample {
			mean[j] += val
		}
	}
	for j := range mean {
		mean[j] /= float32(len(data))
	}

	// Calculate standard deviation
	for _, sample := range data {
		for j, val := range sample {
			diff := val - mean[j]
			stdDev[j] += diff * diff
		}
	}
	for j := range stdDev {
		stdDev[j] = float32(math.Sqrt(float64(stdDev[j] / float32(len(data)))))
		if stdDev[j] < 1e-8 {
			stdDev[j] = 1.0 // Avoid division by zero
		}
	}

	return DataStats{Mean: mean, StdDev: stdDev}
}

// normalizeData applies z-score normalization
func normalizeData(data [][]float32, stats DataStats) [][]float32 {
	normalized := make([][]float32, len(data))
	for i, sample := range data {
		normalized[i] = make([]float32, len(sample))
		for j, val := range sample {
			normalized[i][j] = (val - stats.Mean[j]) / stats.StdDev[j]
		}
	}
	return normalized
}

// oversampleMinorityClass balances dataset by oversampling minority class to 40%
func oversampleMinorityClass(features [][]float32, labels [][]float32) ([][]float32, [][]float32) {
	balancedX := make([][]float32, 0)
	balancedY := make([][]float32, 0)

	// Add all original samples
	balancedX = append(balancedX, features...)
	balancedY = append(balancedY, labels...)

	// Count minority class (good wine = 1)
	minorityCount := 0
	minorityIndices := make([]int, 0)
	for i := range labels {
		if labels[i][0] > 0.5 {
			minorityCount++
			minorityIndices = append(minorityIndices, i)
		}
	}

	// Target 40% minority class (good wine)
	targetMinorityRatio := float32(0.4)
	targetMinorityCount := int(float32(len(labels)) / (1 - targetMinorityRatio) * targetMinorityRatio)
	samplesToAdd := targetMinorityCount - minorityCount

	if samplesToAdd > 0 && len(minorityIndices) > 0 {
		// Add samples in round-robin fashion
		for added := 0; added < samplesToAdd; added++ {
			idx := minorityIndices[added%len(minorityIndices)]
			balancedX = append(balancedX, features[idx])
			balancedY = append(balancedY, labels[idx])
		}
	}

	return balancedX, balancedY
}
