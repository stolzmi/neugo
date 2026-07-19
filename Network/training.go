package Network

// TrainingConfig holds all training parameters
type TrainingConfig struct {
	Epochs          int
	BatchSize       int
	LearningRate    float32
	L2Lambda        float32
	DropoutRate     float32
	Scheduler       LRScheduler
	EarlyStopping   *EarlyStopping
	Callbacks       *CallbackList
	ValidationData  [][]float32
	ValidationLabels [][]float32
	Threshold       float32 // For metrics evaluation
	Verbose         bool
}

// NewTrainingConfig creates a default training configuration
func NewTrainingConfig(epochs int, batchSize int, learningRate float32) *TrainingConfig {
	return &TrainingConfig{
		Epochs:       epochs,
		BatchSize:    batchSize,
		LearningRate: learningRate,
		L2Lambda:     0.0,
		DropoutRate:  0.0,
		Threshold:    0.5,
		Verbose:      true,
		Callbacks:    NewCallbackList(),
	}
}

// Train trains the network with callbacks and full configuration
func (network *Network) Train(
	trainX [][]float32,
	trainY [][]float32,
	config *TrainingConfig,
) *History {
	// Create history tracker
	history := NewHistory()
	config.Callbacks.Add(history)

	// Call train begin callbacks
	config.Callbacks.OnTrainBegin(network)

	numSamples := len(trainX)
	bestValLoss := float32(1e9)

	for epoch := 0; epoch < config.Epochs; epoch++ {
		// Get learning rate from scheduler if available
		lr := config.LearningRate
		if config.Scheduler != nil {
			lr = config.Scheduler.GetLearningRate(epoch)
		}

		// Epoch begin callbacks
		config.Callbacks.OnEpochBegin(epoch, network)

		// Train on batches
		epochLoss := float32(0.0)
		numBatches := 0

		for i := 0; i < numSamples; i += config.BatchSize {
			end := i + config.BatchSize
			if end > numSamples {
				end = numSamples
			}

			batchInputs := trainX[i:end]
			batchLabels := trainY[i:end]

			// Batch begin callback
			config.Callbacks.OnBatchBegin(numBatches, network)

			// Train batch
			loss := network.TrainBatchWithRegularization(
				batchInputs, batchLabels,
				lr, config.L2Lambda, config.DropoutRate,
			)

			epochLoss += loss
			numBatches++

			// Batch end callback
			config.Callbacks.OnBatchEnd(numBatches, network, loss)
		}

		avgTrainLoss := epochLoss / float32(numBatches)
		history.RecordTrainLoss(avgTrainLoss)

		// Validation if data provided
		var valMetrics *Metrics
		if len(config.ValidationData) > 0 {
			metrics := network.Evaluate(
				config.ValidationData,
				config.ValidationLabels,
				config.Threshold,
			)
			valMetrics = &metrics
		}

		// Epoch end callbacks
		config.Callbacks.OnEpochEnd(epoch, network, valMetrics)

		// Early stopping check
		if config.EarlyStopping != nil && valMetrics != nil {
			config.EarlyStopping.Update(valMetrics.Loss, network)
			if valMetrics.Loss < bestValLoss {
				bestValLoss = valMetrics.Loss
			}

			if config.EarlyStopping.ShouldStop {
				if config.Verbose {
					println("Early stopping triggered at epoch", epoch+1)
				}
				config.EarlyStopping.RestoreBestWeights(network)
				break
			}
		}

		// Step scheduler
		if config.Scheduler != nil {
			config.Scheduler.Step()
		}
	}

	// Call train end callbacks
	config.Callbacks.OnTrainEnd(network)

	return history
}

// FitWithCallbacks provides a simpler interface with callbacks
func (network *Network) FitWithCallbacks(
	trainX [][]float32,
	trainY [][]float32,
	valX [][]float32,
	valY [][]float32,
	epochs int,
	batchSize int,
	learningRate float32,
	callbacks *CallbackList,
) *History {
	config := &TrainingConfig{
		Epochs:           epochs,
		BatchSize:        batchSize,
		LearningRate:     learningRate,
		ValidationData:   valX,
		ValidationLabels: valY,
		Threshold:        0.5,
		Verbose:          true,
		Callbacks:        callbacks,
	}

	return network.Train(trainX, trainY, config)
}
