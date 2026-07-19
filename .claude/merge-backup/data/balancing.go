package data

import (
	"math/rand"
	"time"
)

// ClassDistribution holds information about class balance
type ClassDistribution struct {
	Counts      map[int]int     // Class label -> count
	Percentages map[int]float64 // Class label -> percentage
	Total       int
	NumClasses  int
	IsBalanced  bool // True if all classes within 10% of each other
}

// AnalyzeClassDistribution analyzes the distribution of classes in labels
func AnalyzeClassDistribution(labels [][]float32, threshold float32) ClassDistribution {
	counts := make(map[int]int)
	total := len(labels)

	// Count classes (for binary: 0 or 1)
	for _, label := range labels {
		class := 0
		if label[0] > threshold {
			class = 1
		}
		counts[class]++
	}

	// Calculate percentages
	percentages := make(map[int]float64)
	for class, count := range counts {
		percentages[class] = float64(count) / float64(total) * 100
	}

	// Check if balanced (within 10% of 50/50 for binary)
	isBalanced := true
	if len(counts) == 2 {
		ratio := float64(counts[1]) / float64(total)
		isBalanced = ratio >= 0.4 && ratio <= 0.6
	}

	return ClassDistribution{
		Counts:      counts,
		Percentages: percentages,
		Total:       total,
		NumClasses:  len(counts),
		IsBalanced:  isBalanced,
	}
}

// OversampleConfig holds configuration for oversampling
type OversampleConfig struct {
	TargetRatio float64 // Target ratio for minority class (e.g., 0.4 for 40%)
	Strategy    string  // "duplicate" or "random" (random sampling with replacement)
	Seed        int64   // Random seed for reproducibility
}

// OversampleMinorityClass increases minority class samples to improve balance
func OversampleMinorityClass(features [][]float32, labels [][]float32, config OversampleConfig) ([][]float32, [][]float32) {
	if config.Seed > 0 {
		rand.Seed(config.Seed)
	} else {
		rand.Seed(time.Now().UnixNano())
	}

	balancedX := make([][]float32, 0)
	balancedY := make([][]float32, 0)

	// Add all original samples
	balancedX = append(balancedX, features...)
	balancedY = append(balancedY, labels...)

	// Identify minority class (assuming binary classification)
	minorityCount := 0
	minorityIndices := make([]int, 0)
	for i := range labels {
		if labels[i][0] > 0.5 {
			minorityCount++
			minorityIndices = append(minorityIndices, i)
		}
	}

	// Calculate target minority count
	targetMinorityCount := int(float64(len(labels)) / (1 - config.TargetRatio) * config.TargetRatio)
	samplesToAdd := targetMinorityCount - minorityCount

	if samplesToAdd > 0 && len(minorityIndices) > 0 {
		if config.Strategy == "random" {
			// Random sampling with replacement
			for added := 0; added < samplesToAdd; added++ {
				idx := minorityIndices[rand.Intn(len(minorityIndices))]
				balancedX = append(balancedX, features[idx])
				balancedY = append(balancedY, labels[idx])
			}
		} else {
			// Duplicate strategy (round-robin)
			for added := 0; added < samplesToAdd; added++ {
				idx := minorityIndices[added%len(minorityIndices)]
				balancedX = append(balancedX, features[idx])
				balancedY = append(balancedY, labels[idx])
			}
		}
	}

	return balancedX, balancedY
}

// UndersampleConfig holds configuration for undersampling
type UndersampleConfig struct {
	TargetRatio float64 // Target ratio for minority class (e.g., 0.4)
	Strategy    string  // "random" or "systematic" (every nth sample)
	Seed        int64   // Random seed
}

// UndersampleMajorityClass reduces majority class samples
func UndersampleMajorityClass(features [][]float32, labels [][]float32, config UndersampleConfig) ([][]float32, [][]float32) {
	if config.Seed > 0 {
		rand.Seed(config.Seed)
	} else {
		rand.Seed(time.Now().UnixNano())
	}

	minorityX := make([][]float32, 0)
	minorityY := make([][]float32, 0)
	majorityX := make([][]float32, 0)
	majorityY := make([][]float32, 0)

	// Separate minority and majority
	for i := range labels {
		if labels[i][0] > 0.5 {
			minorityX = append(minorityX, features[i])
			minorityY = append(minorityY, labels[i])
		} else {
			majorityX = append(majorityX, features[i])
			majorityY = append(majorityY, labels[i])
		}
	}

	// Calculate how many majority samples to keep
	minorityCount := len(minorityY)
	targetMajorityCount := int(float64(minorityCount) * (1-config.TargetRatio) / config.TargetRatio)

	// Undersample majority class
	if targetMajorityCount < len(majorityY) {
		if config.Strategy == "random" {
			// Random sampling without replacement
			indices := rand.Perm(len(majorityY))[:targetMajorityCount]
			sampledMajorityX := make([][]float32, targetMajorityCount)
			sampledMajorityY := make([][]float32, targetMajorityCount)
			for i, idx := range indices {
				sampledMajorityX[i] = majorityX[idx]
				sampledMajorityY[i] = majorityY[idx]
			}
			majorityX = sampledMajorityX
			majorityY = sampledMajorityY
		} else {
			// Systematic sampling (every nth)
			step := len(majorityY) / targetMajorityCount
			sampledMajorityX := make([][]float32, 0, targetMajorityCount)
			sampledMajorityY := make([][]float32, 0, targetMajorityCount)
			for i := 0; i < len(majorityY) && len(sampledMajorityY) < targetMajorityCount; i += step {
				sampledMajorityX = append(sampledMajorityX, majorityX[i])
				sampledMajorityY = append(sampledMajorityY, majorityY[i])
			}
			majorityX = sampledMajorityX
			majorityY = sampledMajorityY
		}
	}

	// Combine minority and undersampled majority
	balancedX := append(minorityX, majorityX...)
	balancedY := append(minorityY, majorityY...)

	return balancedX, balancedY
}

// BalanceDataset automatically balances dataset using best strategy
func BalanceDataset(features [][]float32, labels [][]float32, targetRatio float64, preferOversample bool) ([][]float32, [][]float32) {
	dist := AnalyzeClassDistribution(labels, 0.5)

	// If already balanced, return as-is
	if dist.IsBalanced {
		return features, labels
	}

	// Choose strategy based on preference and data size
	if preferOversample {
		config := OversampleConfig{
			TargetRatio: targetRatio,
			Strategy:    "duplicate",
			Seed:        42,
		}
		return OversampleMinorityClass(features, labels, config)
	} else {
		config := UndersampleConfig{
			TargetRatio: targetRatio,
			Strategy:    "random",
			Seed:        42,
		}
		return UndersampleMajorityClass(features, labels, config)
	}
}
