package main

import (
	"fmt"
	"neugo/Network"
	"neugo/data"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          🎯  NEUGO FASHION-MNIST CNN DEMO  🎯                 ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	fmt.Println("\n📂 Loading Fashion-MNIST Dataset...")
	fmt.Println("   Binary Classification: T-shirt/Top (0) vs Trouser (1)")
	fmt.Println("\n   Download from:")
	fmt.Println("   https://www.kaggle.com/datasets/zalando-research/fashionmnist")

	trainDataset, err := data.LoadBinaryImageFromCSV(
		"dataset/fashion_mnist/fashion-mnist_train.csv",
		1.0,
	)

	if err != nil {
		fmt.Println("\n❌ Error loading training data:", err)
		fmt.Println("\n   To use this demo:")
		fmt.Println("   1. Download Fashion-MNIST CSV from Kaggle")
		fmt.Println("   2. Place files in dataset/fashion_mnist/")
		fmt.Println("   3. Files: fashion-mnist_train.csv, fashion-mnist_test.csv")
		return
	}

	testDataset, err := data.LoadBinaryImageFromCSV(
		"dataset/fashion_mnist/fashion-mnist_test.csv",
		1.0,
	)

	if err != nil {
		fmt.Println("\n❌ Error loading test data:", err)
		return
	}

	trainSubset := trainDataset.Images[:2000]
	trainLabelsSubset := trainDataset.Labels[:2000]
	testSubset := testDataset.Images[:400]
	testLabelsSubset := testDataset.Labels[:400]

	class0Count := 0
	class1Count := 0
	for _, label := range trainLabelsSubset {
		if label[0] == 0 {
			class0Count++
		} else {
			class1Count++
		}
	}

	fmt.Printf("   ✓ Loaded %d training images (%d T-shirts, %d Trousers)\n",
		len(trainSubset), class0Count, class1Count)
	fmt.Printf("   ✓ Loaded %d test images\n", len(testSubset))
	fmt.Printf("   Image dimensions: 28×28×1 (grayscale)\n")

	fmt.Println("\n🏗️  Building CNN Architecture...")
	cnn := Network.NewCNN(28, 28, 1, Network.BinaryCrossEntropy)

	cnn.AddConv2D(32, 5, 1, 2, Network.ReLU)
	fmt.Println("   Conv2D: 1→32 filters, 5×5 kernel, padding=2, ReLU (28×28×32)")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   MaxPool2D: 2×2, stride=2 (14×14×32)")

	cnn.AddConv2D(64, 5, 1, 2, Network.ReLU)
	fmt.Println("   Conv2D: 32→64 filters, 5×5 kernel, padding=2, ReLU (14×14×64)")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   MaxPool2D: 2×2, stride=2 (7×7×64)")

	cnn.AddConv2D(128, 3, 1, 1, Network.ReLU)
	fmt.Println("   Conv2D: 64→128 filters, 3×3 kernel, ReLU (7×7×128)")

	cnn.AddFlatten()
	fmt.Println("   Flatten (6272)")

	flattenedSize := 7 * 7 * 128
	denseLayers := []Network.Layer{
		Network.NewLayerWithActivation(flattenedSize, Network.Linear),
		Network.NewLayerWithActivation(256, Network.ReLU),
		Network.NewLayerWithActivation(128, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	cnn.SetDenseNetwork(denseLayers)
	fmt.Println("   Dense: 6272→256→128→1 (Sigmoid)")

	totalParams := estimateParams()
	fmt.Printf("\n   📊 Total parameters: ~%d\n", totalParams)

	fmt.Println("\n🏋️  Training CNN on Fashion-MNIST...")
	fmt.Println("   Using mini-batch approach for efficiency")
	fmt.Println("\nEpoch |   Loss   |  LR")
	fmt.Println("------|----------|--------")

	epochs := 30
	initialLR := float32(0.005)

	for epoch := 0; epoch < epochs; epoch++ {
		lr := initialLR / float32(1.0+0.1*float64(epoch))

		epochLoss := float32(0.0)
		batchSize := 100

		for i := 0; i < len(trainSubset); i += batchSize {
			end := i + batchSize
			if end > len(trainSubset) {
				end = len(trainSubset)
			}

			batchLoss := float32(0.0)
			for j := i; j < end; j++ {
				cnn.ForwardPass(trainSubset[j])
				output := []float32{cnn.DenseNetwork.GetOutput()[0].Activation()}
				loss := cnn.Loss.Calculate(output, trainLabelsSubset[j])
				batchLoss += loss
				cnn.BackPropagation(trainSubset[j], trainLabelsSubset[j], lr)
			}
			epochLoss += batchLoss
		}

		avgLoss := epochLoss / float32(len(trainSubset))

		if epoch%5 == 0 || epoch == epochs-1 {
			testMetrics := cnn.Evaluate(testSubset[:100], testLabelsSubset[:100], 0.5)
			fmt.Printf("%5d | %.6f | %.5f | Val Acc: %.2f%%\n",
				epoch+1, avgLoss, lr, testMetrics.Accuracy)
		}
	}

	fmt.Println("\n📊 Final Evaluation on Full Test Set...")
	testMetrics := cnn.Evaluate(testSubset, testLabelsSubset, 0.5)

	fmt.Println("\n┌─────────────────────────────────────────────────────────┐")
	fmt.Println("│               FASHION-MNIST TEST RESULTS                │")
	fmt.Println("├─────────────────────────────────────────────────────────┤")
	fmt.Printf("│ Accuracy:       %6.2f%%                               │\n", testMetrics.Accuracy)
	fmt.Printf("│ Precision:      %6.4f                                 │\n", testMetrics.Precision)
	fmt.Printf("│ Recall:         %6.4f                                 │\n", testMetrics.Recall)
	fmt.Printf("│ F1 Score:       %6.4f                                 │\n", testMetrics.F1Score)
	fmt.Printf("│ Loss:           %6.4f                                 │\n", testMetrics.Loss)
	fmt.Println("└─────────────────────────────────────────────────────────┘")

	fmt.Println("\n📈 Confusion Matrix:")
	fmt.Printf("   [[TN=%d, FP=%d],\n", testMetrics.ConfusionMatrix[0][0], testMetrics.ConfusionMatrix[0][1])
	fmt.Printf("    [FN=%d, TP=%d]]\n", testMetrics.ConfusionMatrix[1][0], testMetrics.ConfusionMatrix[1][1])

	fmt.Println("\n🔮 Sample Predictions (First 10 test images):")
	classNames := []string{"T-shirt/Top", "Trouser"}

	for i := 0; i < 10 && i < len(testSubset); i++ {
		pred := cnn.Predict(testSubset[i])
		actual := testLabelsSubset[i][0]

		predClass := classNames[0]
		actualClass := classNames[0]
		if pred > 0.5 {
			predClass = classNames[1]
		}
		if actual > 0.5 {
			actualClass = classNames[1]
		}

		correct := "✓"
		if (pred > 0.5) != (actual > 0.5) {
			correct = "✗"
		}

		confidence := pred
		if pred < 0.5 {
			confidence = 1.0 - pred
		}

		fmt.Printf("   %s Sample %2d: %s (%.1f%% conf) | Actual: %s\n",
			correct, i+1, predClass, confidence*100, actualClass)
	}

	fmt.Println("\n✅ Fashion-MNIST CNN Highlights:")
	fmt.Println("   • Real-world clothing classification")
	fmt.Println("   • 28×28 grayscale images (same as MNIST)")
	fmt.Println("   • More challenging than digit recognition")
	fmt.Println("   • Multi-scale feature extraction (32→64→128 filters)")
	fmt.Println("   • 5×5 kernels for larger receptive fields")
	fmt.Println("   • Deep architecture with 3 conv + 3 dense layers")
	fmt.Println("   • Mini-batch training for efficiency")

	fmt.Println("\n💡 Fashion-MNIST Classes:")
	fmt.Println("   0: T-shirt/Top    5: Sandal")
	fmt.Println("   1: Trouser        6: Shirt")
	fmt.Println("   2: Pullover       7: Sneaker")
	fmt.Println("   3: Dress          8: Bag")
	fmt.Println("   4: Coat           9: Ankle boot")
	fmt.Println("\n   This demo uses binary classification (0 vs 1)")
	fmt.Println("   But the framework supports multi-class with modifications!")

	fmt.Println("\n═════════════════════════════════════════════════════════════════")
	fmt.Println("✅ FASHION-MNIST DEMO COMPLETE!")
	fmt.Println("═════════════════════════════════════════════════════════════════")
}

func estimateParams() int {
	params := 0

	params += 32 * 1 * 5 * 5
	params += 64 * 32 * 5 * 5
	params += 128 * 64 * 3 * 3

	params += 6272 * 256
	params += 256 * 128
	params += 128 * 1

	params += 32 + 64 + 128
	params += 256 + 128 + 1

	return params
}
