package Network

import (
	"fmt"
	"math"
)

// CrossValResult holds results from cross-validation
type CrossValResult struct {
	FoldMetrics  []Metrics
	MeanAccuracy float32
	StdAccuracy  float32
	MeanF1       float32
	StdF1        float32
	MeanLoss     float32
	StdLoss      float32
	BestFold     int
	WorstFold    int
}

// KFoldSplit holds train/test splits for K-fold cross-validation
type KFoldSplit struct {
	TrainX [][]float32
	TrainY [][]float32
	TestX  [][]float32
	TestY  [][]float32
}

// CreateKFoldSplits creates K-fold splits from data
func CreateKFoldSplits(features [][]float32, labels [][]float32, k int) []KFoldSplit {
	n := len(features)
	foldSize := n / k
	splits := make([]KFoldSplit, k)

	for fold := 0; fold < k; fold++ {
		testStart := fold * foldSize
		testEnd := testStart + foldSize
		if fold == k-1 {
			testEnd = n // Last fold gets remaining samples
		}

		split := KFoldSplit{
			TrainX: make([][]float32, 0),
			TrainY: make([][]float32, 0),
			TestX:  make([][]float32, 0),
			TestY:  make([][]float32, 0),
		}

		// Split data
		for i := 0; i < n; i++ {
			if i >= testStart && i < testEnd {
				split.TestX = append(split.TestX, features[i])
				split.TestY = append(split.TestY, labels[i])
			} else {
				split.TrainX = append(split.TrainX, features[i])
				split.TrainY = append(split.TrainY, labels[i])
			}
		}

		splits[fold] = split
	}

	return splits
}

// CrossValidate performs K-fold cross-validation
func CrossValidate(
	createNetworkFunc func() Network,
	features [][]float32,
	labels [][]float32,
	k int,
	config *TrainingConfig,
	verbose bool,
) CrossValResult {
	if verbose {
		fmt.Printf("\n🔄 Starting %d-Fold Cross-Validation\n", k)
		fmt.Println("═══════════════════════════════════════════")
	}

	splits := CreateKFoldSplits(features, labels, k)
	foldMetrics := make([]Metrics, k)

	for fold := 0; fold < k; fold++ {
		if verbose {
			fmt.Printf("\n📁 Fold %d/%d\n", fold+1, k)
			fmt.Printf("   Training samples: %d\n", len(splits[fold].TrainX))
			fmt.Printf("   Test samples: %d\n", len(splits[fold].TestX))
		}

		// Create fresh network for this fold
		network := createNetworkFunc()

		// Train on fold
		foldConfig := *config // Copy config
		foldConfig.Verbose = false // Suppress per-epoch output
		foldConfig.ValidationData = nil // No validation during training
		foldConfig.Callbacks = NewCallbackList() // Fresh callbacks

		network.Train(splits[fold].TrainX, splits[fold].TrainY, &foldConfig)

		// Evaluate on test fold
		metrics := network.Evaluate(splits[fold].TestX, splits[fold].TestY, config.Threshold)
		foldMetrics[fold] = metrics

		if verbose {
			fmt.Printf("   Accuracy: %.2f%%\n", metrics.Accuracy)
			fmt.Printf("   F1 Score: %.4f\n", metrics.F1Score)
			fmt.Printf("   Loss: %.4f\n", metrics.Loss)
		}
	}

	// Calculate statistics
	result := calculateCrossValStats(foldMetrics, verbose)

	if verbose {
		fmt.Println("\n═══════════════════════════════════════════")
		fmt.Println("📊 Cross-Validation Results:")
		fmt.Printf("   Mean Accuracy: %.2f%% (± %.2f%%)\n", result.MeanAccuracy, result.StdAccuracy)
		fmt.Printf("   Mean F1 Score: %.4f (± %.4f)\n", result.MeanF1, result.StdF1)
		fmt.Printf("   Mean Loss: %.4f (± %.4f)\n", result.MeanLoss, result.StdLoss)
		fmt.Printf("   Best Fold: #%d (Acc: %.2f%%)\n", result.BestFold+1, foldMetrics[result.BestFold].Accuracy)
		fmt.Printf("   Worst Fold: #%d (Acc: %.2f%%)\n", result.WorstFold+1, foldMetrics[result.WorstFold].Accuracy)
		fmt.Println("═══════════════════════════════════════════")
	}

	return result
}

// calculateCrossValStats computes statistics from fold results
func calculateCrossValStats(metrics []Metrics, verbose bool) CrossValResult {
	k := len(metrics)
	result := CrossValResult{
		FoldMetrics: metrics,
	}

	// Calculate means
	sumAcc := float32(0.0)
	sumF1 := float32(0.0)
	sumLoss := float32(0.0)

	bestAcc := float32(-1.0)
	worstAcc := float32(1000.0)
	bestFold := 0
	worstFold := 0

	for i, m := range metrics {
		sumAcc += m.Accuracy
		sumF1 += m.F1Score
		sumLoss += m.Loss

		if m.Accuracy > bestAcc {
			bestAcc = m.Accuracy
			bestFold = i
		}
		if m.Accuracy < worstAcc {
			worstAcc = m.Accuracy
			worstFold = i
		}
	}

	result.MeanAccuracy = sumAcc / float32(k)
	result.MeanF1 = sumF1 / float32(k)
	result.MeanLoss = sumLoss / float32(k)
	result.BestFold = bestFold
	result.WorstFold = worstFold

	// Calculate standard deviations
	sumAccVar := float32(0.0)
	sumF1Var := float32(0.0)
	sumLossVar := float32(0.0)

	for _, m := range metrics {
		sumAccVar += (m.Accuracy - result.MeanAccuracy) * (m.Accuracy - result.MeanAccuracy)
		sumF1Var += (m.F1Score - result.MeanF1) * (m.F1Score - result.MeanF1)
		sumLossVar += (m.Loss - result.MeanLoss) * (m.Loss - result.MeanLoss)
	}

	result.StdAccuracy = float32(math.Sqrt(float64(sumAccVar / float32(k))))
	result.StdF1 = float32(math.Sqrt(float64(sumF1Var / float32(k))))
	result.StdLoss = float32(math.Sqrt(float64(sumLossVar / float32(k))))

	return result
}

// StratifiedKFold creates stratified K-fold splits (preserves class distribution)
func StratifiedKFold(features [][]float32, labels [][]float32, k int, threshold float32) []KFoldSplit {
	// Separate by class
	class0X := make([][]float32, 0)
	class0Y := make([][]float32, 0)
	class1X := make([][]float32, 0)
	class1Y := make([][]float32, 0)

	for i := range labels {
		if labels[i][0] > threshold {
			class1X = append(class1X, features[i])
			class1Y = append(class1Y, labels[i])
		} else {
			class0X = append(class0X, features[i])
			class0Y = append(class0Y, labels[i])
		}
	}

	// Create splits for each class
	splits0 := CreateKFoldSplits(class0X, class0Y, k)
	splits1 := CreateKFoldSplits(class1X, class1Y, k)

	// Combine splits
	combinedSplits := make([]KFoldSplit, k)
	for fold := 0; fold < k; fold++ {
		combinedSplits[fold] = KFoldSplit{
			TrainX: append(splits0[fold].TrainX, splits1[fold].TrainX...),
			TrainY: append(splits0[fold].TrainY, splits1[fold].TrainY...),
			TestX:  append(splits0[fold].TestX, splits1[fold].TestX...),
			TestY:  append(splits0[fold].TestY, splits1[fold].TestY...),
		}
	}

	return combinedSplits
}

// CrossValidateStratified performs stratified K-fold cross-validation
func CrossValidateStratified(
	createNetworkFunc func() Network,
	features [][]float32,
	labels [][]float32,
	k int,
	config *TrainingConfig,
	verbose bool,
) CrossValResult {
	if verbose {
		fmt.Printf("\n🔄 Starting Stratified %d-Fold Cross-Validation\n", k)
		fmt.Println("═══════════════════════════════════════════")
	}

	splits := StratifiedKFold(features, labels, k, config.Threshold)
	foldMetrics := make([]Metrics, k)

	for fold := 0; fold < k; fold++ {
		if verbose {
			fmt.Printf("\n📁 Fold %d/%d\n", fold+1, k)
		}

		// Create fresh network
		network := createNetworkFunc()

		// Train
		foldConfig := *config
		foldConfig.Verbose = false
		foldConfig.ValidationData = nil
		foldConfig.Callbacks = NewCallbackList()

		network.Train(splits[fold].TrainX, splits[fold].TrainY, &foldConfig)

		// Evaluate
		metrics := network.Evaluate(splits[fold].TestX, splits[fold].TestY, config.Threshold)
		foldMetrics[fold] = metrics

		if verbose {
			fmt.Printf("   Accuracy: %.2f%%, F1: %.4f\n", metrics.Accuracy, metrics.F1Score)
		}
	}

	return calculateCrossValStats(foldMetrics, verbose)
}
