package main

import (
	"context"
	"fmt"
	"neugo/data"
	"neugo/nn"
	"neugo/train"
	"neugo/tune"
	"runtime"
	"sync/atomic"
	"time"
)

func toTensor(rows [][]float32) *nn.Tensor {
	cols := len(rows[0])
	flat := make([]float32, len(rows)*cols)
	for i, row := range rows {
		copy(flat[i*cols:(i+1)*cols], row)
	}
	t, _ := nn.NewTensorFromData(flat, []int{len(rows), cols})
	return t
}

func main() {
	// Load and prepare data
	dataset, err := data.LoadCSV("dataset/wine_quality/winequality-red.csv", data.CSVConfig{
		Delimiter:       ';',
		HasHeader:       true,
		LabelColumn:     -1,
		LabelType:       "binary",
		BinaryThreshold: 6.0,
	})
	if err != nil {
		fmt.Println("load csv:", err)
		return
	}

	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	dataRNG := nn.NewRNG(2)
	split := data.SplitData(dataRNG, normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7, ValRatio: 0.15, TestRatio: 0.15, Shuffle: true,
	})

	trainX := toTensor(split.TrainX)
	trainY := toTensor(split.TrainY)
	valX := toTensor(split.ValX)
	valY := toTensor(split.ValY)

	// Create search space
	space := tune.NewSpace().
		LogFloat("lr", 1e-4, 0.5).
		Int("hidden", 4, 64).
		Choice("act", "relu", "tanh")

	// Counter for total epochs executed
	var totalEpochs int64

	// Define objective function
	objective := func(trial *tune.Trial) (float64, error) {
		// Extract hyperparameters
		lr := trial.Params.Float("lr")
		hidden := trial.Params.Int("hidden")
		activation := trial.Params.Choice("act")

		// Build model with trial seed for reproducibility
		rng := nn.NewRNG(trial.Seed)

		var model *nn.SequentialModel
		if activation == "relu" {
			m, err := nn.Sequential([]int{1, dataset.NumFeatures},
				nn.Linear(rng, dataset.NumFeatures, hidden, nn.HeInit()),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(rng, hidden, 1, nn.XavierInit()),
				nn.Sigmoid(),
			)
			if err != nil {
				return 1.0, err
			}
			model = m
		} else { // tanh
			m, err := nn.Sequential([]int{1, dataset.NumFeatures},
				nn.Linear(rng, dataset.NumFeatures, hidden, nn.HeInit()),
				nn.Tanh(),
				nn.Dropout(0.2),
				nn.Linear(rng, hidden, 1, nn.XavierInit()),
				nn.Sigmoid(),
			)
			if err != nil {
				return 1.0, err
			}
			model = m
		}

		// Create trainer
		trainer := train.New(model, train.SGD(float32(lr)), train.BCELoss())

		// Train epoch by epoch, reporting to ASHA
		maxEpochs := 32
		bestValLoss := float64(1.0)

		for epoch := 1; epoch <= maxEpochs; epoch++ {
			// Train for one epoch
			_, err := trainer.Fit(
				trainX, trainY,
				train.Epochs(1), train.BatchSize(32), train.Shuffle(true), train.Seed(trial.Seed+int64(epoch)),
				train.Validation(valX, valY),
			)
			if err != nil {
				return 1.0, err
			}

			// Evaluate on validation set
			metrics, err := trainer.Evaluate(valX, valY)
			if err != nil {
				return 1.0, err
			}

			valLoss := float64(metrics.Loss)
			if valLoss < bestValLoss {
				bestValLoss = valLoss
			}

			// Report to ASHA for pruning decision
			trial.Report(epoch, valLoss)

			// Check if trial should be pruned
			if trial.ShouldPrune() {
				return bestValLoss, nil
			}

			// Increment total epochs counter
			atomic.AddInt64(&totalEpochs, 1)
		}

		return bestValLoss, nil
	}

	// Run hyperparameter tuning
	fmt.Println("Starting hyperparameter tuning...")
	wallStart := time.Now()

	cfg := tune.Config{
		Trials:   60,
		Workers:  runtime.NumCPU(),
		Seed:     42,
		Maximize: false, // Minimize validation loss
		ASHA: &tune.ASHAConfig{
			MinResource:     2,
			MaxResource:     32,
			ReductionFactor: 4,
		},
	}

	results, err := tune.Run(context.Background(), space, objective, cfg)
	if err != nil {
		fmt.Println("tuning error:", err)
		return
	}

	wallTime := time.Since(wallStart)

	// Display results
	fmt.Println("\n=== Top 10 Results ===")
	fmt.Println(results.String())

	best := results.Best()
	if best.Err == nil {
		fmt.Printf("\nBest validation loss: %.4f\n", best.Value)
		fmt.Printf("  Learning rate: %.6f\n", best.Params.Float("lr"))
		fmt.Printf("  Hidden size: %d\n", best.Params.Int("hidden"))
		fmt.Printf("  Activation: %s\n", best.Params.Choice("act"))
	}

	// Print ASHA efficiency stats
	totalPossibleEpochs := cfg.Trials * cfg.ASHA.MaxResource
	actualEpochs := atomic.LoadInt64(&totalEpochs)
	efficiency := float64(actualEpochs) / float64(totalPossibleEpochs) * 100

	fmt.Printf("\n=== ASHA Efficiency ===\n")
	fmt.Printf("Wall time: %v\n", wallTime)
	fmt.Printf("Total epochs executed: %d (via atomic counter)\n", actualEpochs)
	fmt.Printf("Possible epochs (60 trials × 32 max): %d\n", totalPossibleEpochs)
	fmt.Printf("ASHA pruning efficiency: %.1f%% (lower = more aggressive pruning)\n", efficiency)
}
