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

	rng := rand.New(rand.NewSource(42))
	shuffledX, shuffledY := data.ShuffleData(rng, features, labels)

Split data:

	split := data.SplitData(rng, features, labels, data.SplitConfig{
		TrainRatio: 0.7,
		ValRatio:   0.15,
		TestRatio:  0.15,
		Shuffle:    true,
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
		rng, features, labels,
		data.OversampleConfig{
			TargetRatio: 0.4,  // 40% minority class
			Strategy:    "duplicate",
		},
	)

Undersample majority class:

	balancedX, balancedY := data.UndersampleMajorityClass(
		rng, features, labels,
		data.UndersampleConfig{
			TargetRatio: 0.4,
			Strategy:    "random",
		},
	)

Automatic balancing:

	balancedX, balancedY := data.BalanceDataset(
		rng, features, labels,
		0.4,   // target ratio
		true,  // prefer oversample
	)
*/
package data
