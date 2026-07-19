/*
Package data provides utilities for data loading, preprocessing, and balancing.

This package includes tools commonly needed when preparing datasets for neural network training:

# Data Loading

Load CSV files with flexible configuration:

	config := data.CSVConfig{
		Delimiter:       ';',
		HasHeader:       true,
		LabelColumn:     -1, // -1 = last column
		LabelType:       "binary",
		BinaryThreshold: 6.0,
	}
	dataset, err := data.LoadCSV("data.csv", config)

Quick load functions for common cases:

	// Binary classification
	dataset, err := data.QuickLoadBinaryCSV("data.csv", ',', 0.5)

	// Regression
	dataset, err := data.QuickLoadRegressionCSV("data.csv", ',')

# Preprocessing

Calculate statistics and normalize:

	stats := data.CalculateStats(dataset.Features)

	// Z-score normalization (mean=0, std=1)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	// Min-max normalization (scaled to [0, 1])
	normalized := data.NormalizeMinMax(dataset.Features, stats)

Shuffle data:

	shuffledX, shuffledY := data.ShuffleData(features, labels, 42)

Split data:

	split := data.SplitData(features, labels, data.SplitConfig{
		TrainRatio: 0.7,
		ValRatio:   0.15,
		TestRatio:  0.15,
		Shuffle:    true,
		Seed:       42,
	})

	// Access splits
	trainX := split.TrainX
	trainY := split.TrainY

# Class Balancing

Analyze class distribution:

	dist := data.AnalyzeClassDistribution(labels, 0.5)
	fmt.Printf("Class 0: %.2f%%\n", dist.Percentages[0])
	fmt.Printf("Class 1: %.2f%%\n", dist.Percentages[1])
	fmt.Printf("Balanced: %v\n", dist.IsBalanced)

Oversample minority class:

	balancedX, balancedY := data.OversampleMinorityClass(
		features, labels,
		data.OversampleConfig{
			TargetRatio: 0.4,  // 40% minority class
			Strategy:    "duplicate",
			Seed:        42,
		},
	)

Undersample majority class:

	balancedX, balancedY := data.UndersampleMajorityClass(
		features, labels,
		data.UndersampleConfig{
			TargetRatio: 0.4,
			Strategy:    "random",
			Seed:        42,
		},
	)

Automatic balancing:

	balancedX, balancedY := data.BalanceDataset(
		features, labels,
		0.4,   // target ratio
		true,  // prefer oversample
	)

# Complete Example

	package main

	import (
		"fmt"
		"neugo/Network"
		"neugo/data"
	)

	func main() {
		// Load data
		dataset, _ := data.QuickLoadBinaryCSV("wine.csv", ';', 6.0)

		// Analyze distribution
		dist := data.AnalyzeClassDistribution(dataset.Labels, 0.5)
		fmt.Printf("Classes: %v\n", dist.Counts)

		// Normalize
		stats := data.CalculateStats(dataset.Features)
		normalized := data.NormalizeZScore(dataset.Features, stats)

		// Split data
		split := data.SplitData(normalized, dataset.Labels, data.SplitConfig{
			TrainRatio: 0.7,
			ValRatio:   0.15,
			TestRatio:  0.15,
			Shuffle:    true,
		})

		// Balance training data
		trainX, trainY := data.BalanceDataset(
			split.TrainX, split.TrainY,
			0.4, true,
		)

		// Build and train network
		layers := []Network.Layer{
			Network.NewLayerWithActivation(dataset.NumFeatures, Network.Linear),
			Network.NewLayerWithActivation(16, Network.ReLU),
			Network.NewLayerWithActivation(1, Network.Sigmoid),
		}
		network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)

		// Train
		losses := network.Fit(trainX, trainY, 100, 32, 0.1, true)

		// Evaluate
		metrics := network.Evaluate(split.TestX, split.TestY, 0.5)
		fmt.Printf("Accuracy: %.2f%%\n", metrics.Accuracy)
	}
*/
package data
