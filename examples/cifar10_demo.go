package main

import (
	"fmt"
	"neugo/Network"
	"neugo/data"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          🎯  NEUGO CIFAR-10 CNN DEMO  🎯                      ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	fmt.Println("\n📂 Loading CIFAR-10 Dataset...")
	fmt.Println("   Note: This demo uses a binary classification task")
	fmt.Println("   Classes: Airplane (0) vs Automobile (1)")
	fmt.Println("\n   Please ensure you have CIFAR-10 binary files:")
	fmt.Println("   - data_batch_1.bin")
	fmt.Println("   - test_batch.bin")

	trainDataset, err := data.LoadCIFAR10BinaryClassSubset(
		"dataset/cifar10/data_batch_1.bin",
		[]int{0, 1},
	)

	if err != nil {
		fmt.Println("\n❌ Error loading training data:", err)
		fmt.Println("\n   To use this demo:")
		fmt.Println("   1. Download CIFAR-10 binary version from:")
		fmt.Println("      https://www.cs.toronto.edu/~kriz/cifar.html")
		fmt.Println("   2. Extract to dataset/cifar10/")
		return
	}

	testDataset, err := data.LoadCIFAR10BinaryClassSubset(
		"dataset/cifar10/test_batch.bin",
		[]int{0, 1},
	)

	if err != nil {
		fmt.Println("\n❌ Error loading test data:", err)
		return
	}

	fmt.Printf("   ✓ Loaded %d training images\n", len(trainDataset.Images))
	fmt.Printf("   ✓ Loaded %d test images\n", len(testDataset.Images))
	fmt.Printf("   Image dimensions: 32×32×3 (RGB)\n")
	fmt.Printf("   Classes: %s vs %s\n", trainDataset.ClassNames[0], trainDataset.ClassNames[1])

	fmt.Println("\n🏗️  Building CNN for CIFAR-10...")
	cnn := Network.NewCNN(32, 32, 3, Network.BinaryCrossEntropy)

	cnn.AddConv2D(32, 3, 1, 1, Network.ReLU)
	fmt.Println("   Conv2D: 3→32 filters, 3×3, ReLU (32×32×32)")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   MaxPool2D: 2×2 (16×16×32)")

	cnn.AddConv2D(64, 3, 1, 1, Network.ReLU)
	fmt.Println("   Conv2D: 32→64 filters, 3×3, ReLU (16×16×64)")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   MaxPool2D: 2×2 (8×8×64)")

	cnn.AddConv2D(128, 3, 1, 1, Network.ReLU)
	fmt.Println("   Conv2D: 64→128 filters, 3×3, ReLU (8×8×128)")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   MaxPool2D: 2×2 (4×4×128)")

	cnn.AddFlatten()
	fmt.Println("   Flatten (2048)")

	flattenedSize := 4 * 4 * 128
	denseLayers := []Network.Layer{
		Network.NewLayerWithActivation(flattenedSize, Network.Linear),
		Network.NewLayerWithActivation(256, Network.ReLU),
		Network.NewLayerWithActivation(128, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	cnn.SetDenseNetwork(denseLayers)
	fmt.Println("   Dense: 2048→256→128→1 (Sigmoid)")

	fmt.Println("\n🏋️  Training CNN on CIFAR-10...")
	fmt.Println("Epoch | Avg Loss | LR")
	fmt.Println("------|----------|--------")

	epochs := 20
	initialLR := float32(0.001)

	for epoch := 0; epoch < epochs; epoch++ {
		lr := initialLR * float32(1.0/(1.0+0.05*float64(epoch)))

		epochLoss := float32(0.0)
		batchSize := 50

		for i := 0; i < len(trainDataset.Images); i += batchSize {
			end := i + batchSize
			if end > len(trainDataset.Images) {
				end = len(trainDataset.Images)
			}

			for j := i; j < end; j++ {
				cnn.ForwardPass(trainDataset.Images[j])
				output := []float32{cnn.DenseNetwork.GetOutput()[0].Activation()}
				loss := cnn.Loss.Calculate(output, trainDataset.Labels[j])
				epochLoss += loss
				cnn.BackPropagation(trainDataset.Images[j], trainDataset.Labels[j], lr)
			}
		}

		avgLoss := epochLoss / float32(len(trainDataset.Images))

		if epoch%5 == 0 || epoch == epochs-1 {
			fmt.Printf("%5d | %.6f | %.6f\n", epoch+1, avgLoss, lr)
		}
	}

	fmt.Println("\n📊 Evaluating on Test Set...")
	testMetrics := cnn.Evaluate(testDataset.Images, testDataset.Labels, 0.5)

	fmt.Println("\n┌─────────────────────────────────────────────────────────┐")
	fmt.Println("│                    TEST RESULTS                         │")
	fmt.Println("├─────────────────────────────────────────────────────────┤")
	fmt.Printf("│ Accuracy:       %6.2f%%                               │\n", testMetrics.Accuracy)
	fmt.Printf("│ Precision:      %6.4f                                 │\n", testMetrics.Precision)
	fmt.Printf("│ Recall:         %6.4f                                 │\n", testMetrics.Recall)
	fmt.Printf("│ F1 Score:       %6.4f                                 │\n", testMetrics.F1Score)
	fmt.Printf("│ Loss:           %6.4f                                 │\n", testMetrics.Loss)
	fmt.Println("└─────────────────────────────────────────────────────────┘")

	fmt.Println("\n🔮 Sample Predictions:")
	for i := 0; i < 10 && i < len(testDataset.Images); i++ {
		pred := cnn.Predict(testDataset.Images[i])
		actual := testDataset.Labels[i][0]

		predClass := testDataset.ClassNames[0]
		actualClass := testDataset.ClassNames[0]
		if pred > 0.5 {
			predClass = testDataset.ClassNames[1]
		}
		if actual > 0.5 {
			actualClass = testDataset.ClassNames[1]
		}

		correct := "✓"
		if (pred > 0.5) != (actual > 0.5) {
			correct = "✗"
		}

		fmt.Printf("   %s Sample %2d: %s (%.1f%% confidence) | Actual: %s\n",
			correct, i+1, predClass, pred*100, actualClass)
	}

	fmt.Println("\n✅ CIFAR-10 CNN Features:")
	fmt.Println("   • RGB image support (3 channels)")
	fmt.Println("   • Multi-layer deep CNN architecture")
	fmt.Println("   • Progressive filter increase (32→64→128)")
	fmt.Println("   • Multiple pooling layers for spatial reduction")
	fmt.Println("   • Real-world image classification")
	fmt.Println("   • Binary classification on natural images")

	fmt.Println("\n═════════════════════════════════════════════════════════════════")
	fmt.Println("✅ CIFAR-10 DEMO COMPLETE!")
	fmt.Println("═════════════════════════════════════════════════════════════════")
}
