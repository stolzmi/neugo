package data

import (
	"math"
	"math/rand"
	"time"
)

// Stats holds statistical information for normalization
type Stats struct {
	Mean   []float32
	StdDev []float32
	Min    []float32
	Max    []float32
}

// CalculateStats computes mean, std dev, min, max for each feature
func CalculateStats(data [][]float32) Stats {
	if len(data) == 0 {
		return Stats{}
	}

	numFeatures := len(data[0])
	mean := make([]float32, numFeatures)
	stdDev := make([]float32, numFeatures)
	min := make([]float32, numFeatures)
	max := make([]float32, numFeatures)

	// Initialize min/max
	for j := range min {
		min[j] = float32(math.Inf(1))
		max[j] = float32(math.Inf(-1))
	}

	// Calculate mean, min, max
	for _, sample := range data {
		for j, val := range sample {
			mean[j] += val
			if val < min[j] {
				min[j] = val
			}
			if val > max[j] {
				max[j] = val
			}
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

	return Stats{Mean: mean, StdDev: stdDev, Min: min, Max: max}
}

// NormalizeZScore applies z-score normalization (standardization)
// Formula: (x - mean) / std_dev
func NormalizeZScore(data [][]float32, stats Stats) [][]float32 {
	normalized := make([][]float32, len(data))
	for i, sample := range data {
		normalized[i] = make([]float32, len(sample))
		for j, val := range sample {
			normalized[i][j] = (val - stats.Mean[j]) / stats.StdDev[j]
		}
	}
	return normalized
}

// NormalizeMinMax applies min-max normalization (scales to [0, 1])
// Formula: (x - min) / (max - min)
func NormalizeMinMax(data [][]float32, stats Stats) [][]float32 {
	normalized := make([][]float32, len(data))
	for i, sample := range data {
		normalized[i] = make([]float32, len(sample))
		for j, val := range sample {
			range_ := stats.Max[j] - stats.Min[j]
			if range_ < 1e-8 {
				normalized[i][j] = 0.0
			} else {
				normalized[i][j] = (val - stats.Min[j]) / range_
			}
		}
	}
	return normalized
}

// Denormalize converts normalized data back to original scale
func Denormalize(data [][]float32, stats Stats, method string) [][]float32 {
	denormalized := make([][]float32, len(data))
	for i, sample := range data {
		denormalized[i] = make([]float32, len(sample))
		for j, val := range sample {
			if method == "zscore" {
				denormalized[i][j] = val*stats.StdDev[j] + stats.Mean[j]
			} else { // minmax
				denormalized[i][j] = val*(stats.Max[j]-stats.Min[j]) + stats.Min[j]
			}
		}
	}
	return denormalized
}

// ShuffleData shuffles features and labels in unison
func ShuffleData(features [][]float32, labels [][]float32, seed int64) ([][]float32, [][]float32) {
	if seed > 0 {
		rand.Seed(seed)
	} else {
		rand.Seed(time.Now().UnixNano())
	}

	n := len(features)
	indices := rand.Perm(n)

	shuffledX := make([][]float32, n)
	shuffledY := make([][]float32, n)

	for i, idx := range indices {
		shuffledX[i] = features[idx]
		shuffledY[i] = labels[idx]
	}

	return shuffledX, shuffledY
}

// SplitConfig holds configuration for data splitting
type SplitConfig struct {
	TrainRatio float64 // e.g., 0.7 for 70%
	ValRatio   float64 // e.g., 0.15 for 15%
	TestRatio  float64 // e.g., 0.15 for 15%
	Shuffle    bool    // Whether to shuffle before splitting
	Seed       int64   // Random seed for reproducibility
}

// Split holds the split datasets
type Split struct {
	TrainX [][]float32
	TrainY [][]float32
	ValX   [][]float32
	ValY   [][]float32
	TestX  [][]float32
	TestY  [][]float32
}

// SplitData splits data into train/validation/test sets
func SplitData(features [][]float32, labels [][]float32, config SplitConfig) Split {
	// Validate ratios
	total := config.TrainRatio + config.ValRatio + config.TestRatio
	if math.Abs(total-1.0) > 0.01 {
		// Auto-adjust if test ratio not specified
		if config.TestRatio == 0 {
			config.TestRatio = 1.0 - config.TrainRatio - config.ValRatio
		}
	}

	// Shuffle if requested
	if config.Shuffle {
		features, labels = ShuffleData(features, labels, config.Seed)
	}

	n := len(features)
	trainSize := int(float64(n) * config.TrainRatio)
	valSize := int(float64(n) * config.ValRatio)

	split := Split{}
	split.TrainX = features[:trainSize]
	split.TrainY = labels[:trainSize]
	split.ValX = features[trainSize : trainSize+valSize]
	split.ValY = labels[trainSize : trainSize+valSize]
	split.TestX = features[trainSize+valSize:]
	split.TestY = labels[trainSize+valSize:]

	return split
}
