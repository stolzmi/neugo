package Network

import "math"

// Metrics holds evaluation metrics
type Metrics struct {
	Loss      float32
	Accuracy  float32
	Precision float32
	Recall    float32
	F1Score   float32
	ConfusionMatrix [][]int
}

// Evaluate evaluates the network on a dataset and returns metrics
func (network *Network) Evaluate(inputs [][]float32, labels [][]float32, threshold float32) Metrics {
	if len(inputs) == 0 || len(inputs) != len(labels) {
		return Metrics{}
	}

	totalLoss := float32(0.0)
	correct := 0
	total := len(inputs)

	// For binary classification metrics
	truePositives := 0
	falsePositives := 0
	trueNegatives := 0
	falseNegatives := 0

	outputSize := len(labels[0])

	// Initialize confusion matrix for multi-class
	confusionMatrix := make([][]int, outputSize)
	for i := range confusionMatrix {
		confusionMatrix[i] = make([]int, outputSize)
	}

	for i := 0; i < len(inputs); i++ {
		network.ForwardPass(inputs[i])

		// Calculate loss
		totalLoss += network.CalculateLoss(labels[i])

		// Get predictions
		output := network.GetOutput()

		if outputSize == 1 {
			// Binary classification
			prediction := output[0].Activation()
			label := labels[i][0]

			predictedClass := float32(0)
			if prediction >= threshold {
				predictedClass = 1
			}

			actualClass := float32(0)
			if label >= threshold {
				actualClass = 1
			}

			if predictedClass == actualClass {
				correct++
			}

			// Update confusion matrix values
			if actualClass == 1 && predictedClass == 1 {
				truePositives++
			} else if actualClass == 0 && predictedClass == 1 {
				falsePositives++
			} else if actualClass == 0 && predictedClass == 0 {
				trueNegatives++
			} else if actualClass == 1 && predictedClass == 0 {
				falseNegatives++
			}
		} else {
			// Multi-class classification
			predictedClass := argmax(output)
			actualClass := argmaxSlice(labels[i])

			if predictedClass == actualClass {
				correct++
			}

			confusionMatrix[actualClass][predictedClass]++
		}
	}

	accuracy := float32(correct) / float32(total) * 100

	// Calculate precision, recall, F1
	precision := float32(0.0)
	recall := float32(0.0)
	f1Score := float32(0.0)

	if outputSize == 1 {
		// Binary classification metrics
		if truePositives+falsePositives > 0 {
			precision = float32(truePositives) / float32(truePositives+falsePositives)
		}
		if truePositives+falseNegatives > 0 {
			recall = float32(truePositives) / float32(truePositives+falseNegatives)
		}
		if precision+recall > 0 {
			f1Score = 2 * (precision * recall) / (precision + recall)
		}

		// Build 2x2 confusion matrix
		confusionMatrix = [][]int{
			{trueNegatives, falsePositives},
			{falseNegatives, truePositives},
		}
	} else {
		// Macro-averaged metrics for multi-class
		totalPrecision := float32(0.0)
		totalRecall := float32(0.0)
		numClasses := 0

		for class := 0; class < outputSize; class++ {
			tp := confusionMatrix[class][class]
			fp := 0
			fn := 0

			for i := 0; i < outputSize; i++ {
				if i != class {
					fp += confusionMatrix[i][class]
					fn += confusionMatrix[class][i]
				}
			}

			if tp+fp > 0 {
				totalPrecision += float32(tp) / float32(tp+fp)
				numClasses++
			}
			if tp+fn > 0 {
				totalRecall += float32(tp) / float32(tp+fn)
			}
		}

		if numClasses > 0 {
			precision = totalPrecision / float32(numClasses)
			recall = totalRecall / float32(numClasses)
		}

		if precision+recall > 0 {
			f1Score = 2 * (precision * recall) / (precision + recall)
		}
	}

	return Metrics{
		Loss:            totalLoss / float32(total),
		Accuracy:        accuracy,
		Precision:       precision,
		Recall:          recall,
		F1Score:         f1Score,
		ConfusionMatrix: confusionMatrix,
	}
}

// argmax returns the index of the maximum value in neuron slice
func argmax(neurons []Neuron) int {
	maxIdx := 0
	maxVal := neurons[0].Activation()

	for i := 1; i < len(neurons); i++ {
		if neurons[i].Activation() > maxVal {
			maxVal = neurons[i].Activation()
			maxIdx = i
		}
	}

	return maxIdx
}

// argmaxSlice returns the index of the maximum value in float32 slice
func argmaxSlice(values []float32) int {
	maxIdx := 0
	maxVal := values[0]

	for i := 1; i < len(values); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
			maxIdx = i
		}
	}

	return maxIdx
}

// FitWithValidation trains with validation set and returns training and validation losses
func (network *Network) FitWithValidation(
	trainInputs [][]float32,
	trainLabels [][]float32,
	valInputs [][]float32,
	valLabels [][]float32,
	epochs int,
	batchSize int,
	learningRate float32,
	verbose bool,
) ([]float32, []float32) {

	trainLosses := make([]float32, 0, epochs)
	valLosses := make([]float32, 0, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		// Training
		epochLoss := float32(0.0)
		numBatches := 0
		numSamples := len(trainInputs)

		for i := 0; i < numSamples; i += batchSize {
			end := i + batchSize
			if end > numSamples {
				end = numSamples
			}

			batchInputs := trainInputs[i:end]
			batchLabels := trainLabels[i:end]

			batchLoss := network.TrainBatch(batchInputs, batchLabels, learningRate)
			epochLoss += batchLoss
			numBatches++
		}

		avgTrainLoss := epochLoss / float32(numBatches)
		trainLosses = append(trainLosses, avgTrainLoss)

		// Validation
		valLoss := float32(0.0)
		for i := 0; i < len(valInputs); i++ {
			network.ForwardPass(valInputs[i])
			valLoss += network.CalculateLoss(valLabels[i])
		}
		avgValLoss := valLoss / float32(len(valInputs))
		valLosses = append(valLosses, avgValLoss)

		if verbose && ((epoch+1)%100 == 0 || epoch == 0) {
			println("Epoch", epoch+1, "- Train Loss:", avgTrainLoss, "- Val Loss:", avgValLoss)
		}
	}

	return trainLosses, valLosses
}

// EarlyStopping implements early stopping based on validation loss
type EarlyStopping struct {
	Patience    int
	MinDelta    float32
	BestLoss    float32
	Counter     int
	ShouldStop  bool
	BestWeights [][][]float32
}

// NewEarlyStopping creates a new early stopping callback
func NewEarlyStopping(patience int, minDelta float32) *EarlyStopping {
	return &EarlyStopping{
		Patience:   patience,
		MinDelta:   minDelta,
		BestLoss:   float32(math.Inf(1)),
		Counter:    0,
		ShouldStop: false,
	}
}

// Update updates the early stopping state
func (es *EarlyStopping) Update(valLoss float32, network *Network) {
	if valLoss < es.BestLoss-es.MinDelta {
		es.BestLoss = valLoss
		es.Counter = 0
		// Save best weights
		es.BestWeights = copyWeights(network.weights)
	} else {
		es.Counter++
		if es.Counter >= es.Patience {
			es.ShouldStop = true
		}
	}
}

// RestoreBestWeights restores the best weights to the network
func (es *EarlyStopping) RestoreBestWeights(network *Network) {
	if es.BestWeights != nil {
		network.weights = copyWeights(es.BestWeights)
	}
}

// copyWeights creates a deep copy of weights
func copyWeights(weights [][][]float32) [][][]float32 {
	copy := make([][][]float32, len(weights))
	for i := range weights {
		copy[i] = make([][]float32, len(weights[i]))
		for j := range weights[i] {
			copy[i][j] = make([]float32, len(weights[i][j]))
			for k := range weights[i][j] {
				copy[i][j][k] = weights[i][j][k]
			}
		}
	}
	return copy
}
