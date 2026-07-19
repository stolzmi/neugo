package main

import (
	"fmt"
	"neugo/Network"
	"neugo/data"
)

func main() {
	fmt.Println("🔍 Checking Prediction Distribution")
	fmt.Println("====================================")

	dataset, err := data.QuickLoadBinaryCSV("dataset/wine_quality/winequality-red.csv", ';', 6.0)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	split := data.SplitData(normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7,
		ValRatio:   0.15,
		TestRatio:  0.15,
		Shuffle:    true,
		Seed:       42,
	})

	trainX, trainY := data.OversampleMinorityClass(
		split.TrainX, split.TrainY,
		data.OversampleConfig{TargetRatio: 0.4, Strategy: "duplicate", Seed: 42},
	)

	layers := []Network.Layer{
		Network.NewLayerWithActivation(dataset.NumFeatures, Network.Linear),
		Network.NewLayerWithActivation(32, Network.ReLU),
		Network.NewLayerWithActivation(16, Network.ReLU),
		Network.NewLayerWithActivation(8, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	network := Network.NewNetworkWithLoss(layers, Network.BinaryCrossEntropy)

	fmt.Println("\n🏋️  Training 100 epochs...")
	scheduler := Network.NewCosineAnnealing(0.1, 0.001, 100)

	for epoch := 0; epoch < 100; epoch++ {
		lr := scheduler.GetLearningRate(epoch)
		epochLoss := float32(0.0)
		numBatches := 0

		for i := 0; i < len(trainX); i += 32 {
			end := i + 32
			if end > len(trainX) {
				end = len(trainX)
			}
			loss := network.TrainBatchWithRegularization(trainX[i:end], trainY[i:end], lr, 0.001, 0.2)
			epochLoss += loss
			numBatches++
		}

		if epoch%20 == 0 || epoch == 99 {
			avgLoss := epochLoss / float32(numBatches)
			valMetrics := network.Evaluate(split.ValX, split.ValY, 0.5)
			fmt.Printf("Epoch %3d: Loss=%.4f, Val Acc=%.2f%%, Val F1=%.4f\n",
				epoch+1, avgLoss, valMetrics.Accuracy, valMetrics.F1Score)
		}

		scheduler.Step()
	}

	fmt.Println("\n📊 Analyzing Predictions on Test Set:")
	fmt.Println("======================================")

	predictedClass0 := 0
	predictedClass1 := 0
	actualClass0 := 0
	actualClass1 := 0

	truePositive := 0
	trueNegative := 0
	falsePositive := 0
	falseNegative := 0

	predictions := make([]float32, len(split.TestX))

	for i := 0; i < len(split.TestX); i++ {
		network.ForwardPass(split.TestX[i])
		pred := network.GetOutput()[0].Activation()
		predictions[i] = pred
		actual := split.TestY[i][0]

		if actual == 0 {
			actualClass0++
		} else {
			actualClass1++
		}

		if pred >= 0.5 {
			predictedClass1++
			if actual == 1 {
				truePositive++
			} else {
				falsePositive++
			}
		} else {
			predictedClass0++
			if actual == 0 {
				trueNegative++
			} else {
				falseNegative++
			}
		}
	}

	fmt.Printf("\nActual Distribution:\n")
	fmt.Printf("  Class 0 (Bad):  %d samples (%.1f%%)\n", actualClass0, float32(actualClass0)/float32(len(split.TestX))*100)
	fmt.Printf("  Class 1 (Good): %d samples (%.1f%%)\n", actualClass1, float32(actualClass1)/float32(len(split.TestX))*100)

	fmt.Printf("\nPredicted Distribution:\n")
	fmt.Printf("  Class 0 (Bad):  %d predictions (%.1f%%)\n", predictedClass0, float32(predictedClass0)/float32(len(split.TestX))*100)
	fmt.Printf("  Class 1 (Good): %d predictions (%.1f%%)\n", predictedClass1, float32(predictedClass1)/float32(len(split.TestX))*100)

	fmt.Printf("\nConfusion Matrix:\n")
	fmt.Printf("                 Predicted Bad  Predicted Good\n")
	fmt.Printf("  Actual Bad     %6d         %6d\n", trueNegative, falsePositive)
	fmt.Printf("  Actual Good    %6d         %6d\n", falseNegative, truePositive)

	fmt.Printf("\nMetrics:\n")
	precision := float32(0)
	if predictedClass1 > 0 {
		precision = float32(truePositive) / float32(predictedClass1)
	}
	recall := float32(0)
	if actualClass1 > 0 {
		recall = float32(truePositive) / float32(actualClass1)
	}
	accuracy := float32(truePositive+trueNegative) / float32(len(split.TestX))

	fmt.Printf("  Accuracy:  %.2f%%\n", accuracy*100)
	fmt.Printf("  Precision: %.4f\n", precision)
	fmt.Printf("  Recall:    %.4f\n", recall)

	fmt.Println("\n🔍 Sample Predictions (showing variety):")
	fmt.Println("   Low predictions (<0.3):")
	count := 0
	for i := 0; i < len(predictions) && count < 3; i++ {
		if predictions[i] < 0.3 {
			fmt.Printf("      Sample %d: pred=%.4f, actual=%.0f\n", i+1, predictions[i], split.TestY[i][0])
			count++
		}
	}

	fmt.Println("   Medium predictions (0.3-0.7):")
	count = 0
	for i := 0; i < len(predictions) && count < 3; i++ {
		if predictions[i] >= 0.3 && predictions[i] <= 0.7 {
			fmt.Printf("      Sample %d: pred=%.4f, actual=%.0f\n", i+1, predictions[i], split.TestY[i][0])
			count++
		}
	}

	fmt.Println("   High predictions (>0.7):")
	count = 0
	for i := 0; i < len(predictions) && count < 3; i++ {
		if predictions[i] > 0.7 {
			fmt.Printf("      Sample %d: pred=%.4f, actual=%.0f\n", i+1, predictions[i], split.TestY[i][0])
			count++
		}
	}
}
