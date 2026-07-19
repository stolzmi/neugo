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
	fmt.Println("║                    🕵️  FRAUD DETECTION CLASSIFIER 🕵️                          ║")
	fmt.Println("║                     Using NeuGo Neural Network                                ║")
	fmt.Println("║                                                                                ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════╝")

	// Load data
	fmt.Println("\n📂 Loading dataset...")
	trainX, trainY, err := loadData("dataset/train_set/x_train_aggregated.csv", "dataset/train_set/y_train.csv")
	if err != nil {
		fmt.Println("❌ Error loading training data:", err)
		return
	}

	testX, testY, err := loadData("dataset/test_set/x_test_aggregated.csv", "")
	if err != nil {
		fmt.Println("❌ Error loading test data:", err)
		return
	}

	fmt.Printf("   ✓ Training samples: %d\n", len(trainY))
	fmt.Printf("   ✓ Test samples: %d\n", len(testX))
	fmt.Printf("   ✓ Features: %d\n", len(trainX[0]))

	// Check class distribution
	fraudCount := 0
	for _, label := range trainY {
		if label[0] > 0.5 {
			fraudCount++
		}
	}
	fmt.Printf("\n📊 Class Distribution:\n")
	fmt.Printf("   Non-Fraud: %d (%.2f%%)\n", len(trainY)-fraudCount, float64(len(trainY)-fraudCount)/float64(len(trainY))*100)
	fmt.Printf("   Fraud:     %d (%.2f%%)\n", fraudCount, float64(fraudCount)/float64(len(trainY))*100)

	// Normalize data
	fmt.Println("\n🔧 Preprocessing data...")
	stats := calculateStats(trainX)
	trainX = normalizeData(trainX, stats)
	testX = normalizeData(testX, stats)
	fmt.Println("   ✓ Data normalized (z-score normalization)")

	// Split training data for validation
	fmt.Println("\n✂️  Splitting data...")
	valSize := len(trainX) / 5 // 20% for validation
	valX := trainX[len(trainX)-valSize:]
	valY := trainY[len(trainY)-valSize:]
	trainX = trainX[:len(trainX)-valSize]
	trainY = trainY[:len(trainY)-valSize]

	fmt.Printf("   Training set: %d samples\n", len(trainX))
	fmt.Printf("   Validation set: %d samples\n", len(valX))
	fmt.Printf("   Test set: %d samples\n", len(testX))

	// Build network (simpler architecture to prevent collapse)
	fmt.Println("\n🏗️  Building neural network...")
	inputSize := len(trainX[0])

	layers := []Network.Layer{
		Network.NewLayerWithActivation(inputSize, Network.Linear),
		Network.NewLayerWithActivation(32, Network.ReLU),
		Network.NewLayerWithActivation(16, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)

	fmt.Printf("   Architecture: %d → 32 → 16 → 1\n", inputSize)
	fmt.Println("   Activation: ReLU (hidden), Sigmoid (output)")
	fmt.Println("   Loss: Binary Cross-Entropy with class weighting")

	// Training configuration
	fmt.Println("\n⚙️  Training Configuration:")
	fmt.Println("   ├─ Optimizer: SGD")
	fmt.Println("   ├─ Initial LR: 0.05")
	fmt.Println("   ├─ Scheduler: Cosine Annealing (0.05 → 0.001)")
	fmt.Println("   ├─ L2 Regularization: 0.0005")
	fmt.Println("   ├─ Dropout: 0.2 (20%)")
	fmt.Println("   ├─ Batch Size: 32")
	fmt.Println("   ├─ Epochs: 200")
	fmt.Println("   ├─ Decision Threshold: 0.5 (standard)")
	fmt.Println("   ├─ Class Balancing: Partial oversample to 30% fraud")
	fmt.Println("   └─ Early Stopping: Patience 25")

	// Setup training components
	scheduler := Network.NewCosineAnnealing(0.05, 0.001, 200)
	earlyStopping := Network.NewEarlyStopping(25, 0.0001)

	// Oversample fraud cases to balance training
	balancedTrainX := make([][]float32, 0)
	balancedTrainY := make([][]float32, 0)

	// Add all samples
	balancedTrainX = append(balancedTrainX, trainX...)
	balancedTrainY = append(balancedTrainY, trainY...)

	// Count fraud in training set
	fraudTrainCount := 0
	for i := range trainY {
		if trainY[i][0] > 0.5 {
			fraudTrainCount++
		}
	}

	// Partially oversample fraud cases (not full balance, just improve ratio)
	// Target: 30% fraud instead of 16%
	targetFraudRatio := float32(0.3)
	targetFraudCount := int(float32(len(trainY)) / (1 - targetFraudRatio) * targetFraudRatio)
	fraudSamplesToAdd := targetFraudCount - fraudTrainCount

	if fraudSamplesToAdd > 0 {
		// Collect all fraud indices
		fraudIndices := make([]int, 0)
		for i := range trainY {
			if trainY[i][0] > 0.5 {
				fraudIndices = append(fraudIndices, i)
			}
		}

		// Repeat fraud samples in round-robin fashion
		for added := 0; added < fraudSamplesToAdd; added++ {
			idx := fraudIndices[added%len(fraudIndices)]
			balancedTrainX = append(balancedTrainX, trainX[idx])
			balancedTrainY = append(balancedTrainY, trainY[idx])
		}
	}

	// Replace trainX and trainY with balanced versions
	trainX = balancedTrainX
	trainY = balancedTrainY

	// Count fraud after balancing
	fraudAfterBalance := 0
	for i := range trainY {
		if trainY[i][0] > 0.5 {
			fraudAfterBalance++
		}
	}
	fraudRatio := float32(fraudAfterBalance) / float32(len(trainY)) * 100
	fmt.Printf("\n   📊 After oversampling: %d samples (%.1f%% fraud)\n", len(trainY), fraudRatio)

	// Training loop
	fmt.Println("\n🏋️  Training...")
	fmt.Println("\n   Epoch  |  Train Loss  |  Val Loss  |  Val F1   |  Val Acc  |    LR")
	fmt.Println("   -------|--------------|------------|-----------|-----------|----------")

	batchSize := 32
	bestValF1 := float32(0.0)
	threshold := float32(0.5) // Standard threshold

	for epoch := 0; epoch < 200; epoch++ {
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

			loss := network.TrainBatchWithRegularization(batchInputs, batchLabels, lr, 0.0005, 0.2)
			epochLoss += loss
			numBatches++
		}

		avgTrainLoss := epochLoss / float32(numBatches)

		// Validation
		valMetrics := network.Evaluate(valX, valY, threshold)

		// Early stopping based on F1 score
		if valMetrics.F1Score > bestValF1 {
			bestValF1 = valMetrics.F1Score
			earlyStopping.BestLoss = valMetrics.Loss
			earlyStopping.Counter = 0
		} else {
			earlyStopping.Counter++
		}

		// Print progress
		if epoch%20 == 0 || epoch == 199 || earlyStopping.Counter >= earlyStopping.Patience {
			fmt.Printf("   %5d  |   %.6f   |  %.6f  |  %.4f    |  %6.2f%%  | %.6f\n",
				epoch+1, avgTrainLoss, valMetrics.Loss, valMetrics.F1Score, valMetrics.Accuracy, lr)
		}

		if earlyStopping.Counter >= earlyStopping.Patience {
			fmt.Println("\n   🛑 Early stopping triggered!")
			break
		}

		scheduler.Step()
	}

	// Final evaluation on validation set
	fmt.Println("\n" + strings.Repeat("─", 80))
	fmt.Println("\n📊 FINAL VALIDATION RESULTS")
	fmt.Println(strings.Repeat("─", 80))

	finalMetrics := network.Evaluate(valX, valY, threshold)

	fmt.Println("\n📈 Classification Metrics:")
	fmt.Println("   ┌─────────────────────────────────────┐")
	fmt.Printf("   │ Accuracy:    %6.2f%%              │\n", finalMetrics.Accuracy)
	fmt.Printf("   │ Precision:   %6.4f                │\n", finalMetrics.Precision)
	fmt.Printf("   │ Recall:      %6.4f                │\n", finalMetrics.Recall)
	fmt.Printf("   │ F1 Score:    %6.4f                │\n", finalMetrics.F1Score)
	fmt.Printf("   │ Loss:        %6.4f                │\n", finalMetrics.Loss)
	fmt.Println("   └─────────────────────────────────────┘")

	fmt.Println("\n📋 Confusion Matrix:")
	fmt.Println("   ┌───────────────────────────┐")
	fmt.Println("   │       Predicted           │")
	fmt.Println("   │   Non-Fraud   Fraud       │")
	fmt.Println("   ├───────────────────────────┤")
	for i, row := range finalMetrics.ConfusionMatrix {
		if i == 0 {
			fmt.Printf("   │ NF │  %5d    %5d      │\n", row[0], row[1])
		} else {
			fmt.Printf("   │ F  │  %5d    %5d      │\n", row[0], row[1])
		}
	}
	fmt.Println("   └───────────────────────────┘")

	// Calculate additional metrics
	tn := finalMetrics.ConfusionMatrix[0][0]
	fp := finalMetrics.ConfusionMatrix[0][1]
	fn := finalMetrics.ConfusionMatrix[1][0]
	tp := finalMetrics.ConfusionMatrix[1][1]

	fmt.Println("\n📌 Detailed Metrics:")
	fmt.Printf("   True Negatives:  %d\n", tn)
	fmt.Printf("   False Positives: %d (%.2f%%)\n", fp, float64(fp)/float64(tn+fp)*100)
	fmt.Printf("   False Negatives: %d (%.2f%%)\n", fn, float64(fn)/float64(fn+tp)*100)
	fmt.Printf("   True Positives:  %d\n", tp)

	// Save model
	fmt.Println("\n💾 Saving trained model...")
	err = network.SaveToFile("fraud_detection_model.json")
	if err != nil {
		fmt.Println("   ❌ Error saving model:", err)
	} else {
		fmt.Println("   ✓ Model saved to: fraud_detection_model.json")
	}

	// Save predictions for test set (if we have labels)
	if len(testY) > 0 {
		fmt.Println("\n🧪 Evaluating on test set...")
		testMetrics := network.Evaluate(testX, testY, threshold)
		fmt.Printf("   Test Accuracy: %.2f%%\n", testMetrics.Accuracy)
		fmt.Printf("   Test F1 Score: %.4f\n", testMetrics.F1Score)
	} else {
		fmt.Println("\n🔮 Generating predictions for test set...")
		predictions := make([]float32, len(testX))
		for i, input := range testX {
			network.ForwardPass(input)
			predictions[i] = network.GetOutput()[0].Activation()
		}

		// Save predictions with adjusted threshold
		predFile, err := os.Create("test_predictions.csv")
		if err == nil {
			defer predFile.Close()
			writer := csv.NewWriter(predFile)
			writer.Write([]string{"SampleID", "FraudProbability", "PredictedClass"})
			for i, pred := range predictions {
				class := "0"
				if pred >= threshold {
					class = "1"
				}
				writer.Write([]string{fmt.Sprintf("%d", i), fmt.Sprintf("%.6f", pred), class})
			}
			writer.Flush()
			fmt.Println("   ✓ Predictions saved to: test_predictions.csv")
			fmt.Printf("   ✓ Using decision threshold: %.2f\n", threshold)
		}
	}

	fmt.Println("\n" + strings.Repeat("═", 80))
	fmt.Println("\n✅ Fraud Detection Training Complete!")
	fmt.Println("\n" + strings.Repeat("═", 80))
}

// loadData loads features and labels from CSV files
func loadData(featuresPath, labelsPath string) ([][]float32, [][]float32, error) {
	// Load features
	featuresFile, err := os.Open(featuresPath)
	if err != nil {
		return nil, nil, err
	}
	defer featuresFile.Close()

	reader := csv.NewReader(featuresFile)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	// Skip header
	records = records[1:]

	features := make([][]float32, len(records))
	for i, record := range records {
		// Skip AccountID (first column)
		features[i] = make([]float32, len(record)-1)
		for j := 1; j < len(record); j++ {
			val, _ := strconv.ParseFloat(record[j], 32)
			features[i][j-1] = float32(val)
		}
	}

	// Load labels if path provided
	var labels [][]float32
	if labelsPath != "" {
		labelsFile, err := os.Open(labelsPath)
		if err != nil {
			return nil, nil, err
		}
		defer labelsFile.Close()

		reader = csv.NewReader(labelsFile)
		records, err = reader.ReadAll()
		if err != nil {
			return nil, nil, err
		}

		// Skip header
		records = records[1:]

		labels = make([][]float32, len(records))
		for i, record := range records {
			val, _ := strconv.ParseFloat(record[1], 32)
			labels[i] = []float32{float32(val)}
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
