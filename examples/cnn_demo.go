package main

import (
	"fmt"
	"neugo/Network"
	"neugo/tensor"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          🎯  NEUGO CNN DEMONSTRATION  🎯                      ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	fmt.Println("\n📊 Creating Synthetic Image Dataset...")

	trainImages := make([]*tensor.Tensor3D, 100)
	trainLabels := make([][]float32, 100)

	for i := 0; i < 50; i++ {
		img := createPatternImage(28, 28, "vertical")
		trainImages[i] = img
		trainLabels[i] = []float32{0}
	}

	for i := 50; i < 100; i++ {
		img := createPatternImage(28, 28, "horizontal")
		trainImages[i] = img
		trainLabels[i] = []float32{1}
	}

	testImages := make([]*tensor.Tensor3D, 20)
	testLabels := make([][]float32, 20)

	for i := 0; i < 10; i++ {
		img := createPatternImage(28, 28, "vertical")
		testImages[i] = img
		testLabels[i] = []float32{0}
	}

	for i := 10; i < 20; i++ {
		img := createPatternImage(28, 28, "horizontal")
		testImages[i] = img
		testLabels[i] = []float32{1}
	}

	fmt.Printf("   Train: %d images\n", len(trainImages))
	fmt.Printf("   Test: %d images\n", len(testImages))
	fmt.Println("   Classes: 0=Vertical pattern, 1=Horizontal pattern")

	fmt.Println("\n🏗️  Building CNN Architecture...")
	cnn := Network.NewCNN(28, 28, 1, Network.BinaryCrossEntropy)

	cnn.AddConv2D(8, 3, 1, 1, Network.ReLU)
	fmt.Println("   Conv2D: 1 → 8 filters, 3x3 kernel, ReLU")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   MaxPool2D: 2x2, stride 2")

	cnn.AddConv2D(16, 3, 1, 1, Network.ReLU)
	fmt.Println("   Conv2D: 8 → 16 filters, 3x3 kernel, ReLU")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   MaxPool2D: 2x2, stride 2")

	cnn.AddFlatten()
	fmt.Println("   Flatten")

	flattenedSize := 7 * 7 * 16
	denseLayers := []Network.Layer{
		Network.NewLayerWithActivation(flattenedSize, Network.Linear),
		Network.NewLayerWithActivation(32, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	cnn.SetDenseNetwork(denseLayers)
	fmt.Printf("   Dense: %d → 32 → 1 (Sigmoid)\n", flattenedSize)

	fmt.Println("\n🏋️  Training CNN...")
	fmt.Println("Epoch | Avg Loss")
	fmt.Println("------|----------")

	epochs := 50
	learningRate := float32(0.01)

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0.0)

		for i := 0; i < len(trainImages); i++ {
			cnn.ForwardPass(trainImages[i])

			output := []float32{cnn.DenseNetwork.GetOutput()[0].Activation()}
			loss := cnn.Loss.Calculate(output, trainLabels[i])
			epochLoss += loss

			cnn.BackPropagation(trainImages[i], trainLabels[i], learningRate)
		}

		avgLoss := epochLoss / float32(len(trainImages))

		if epoch%10 == 0 || epoch == epochs-1 {
			fmt.Printf("%5d | %.6f\n", epoch+1, avgLoss)
		}
	}

	fmt.Println("\n📊 Evaluating on Test Set...")
	metrics := cnn.Evaluate(testImages, testLabels, 0.5)

	fmt.Println("\n🎯 Test Results:")
	fmt.Println("   ┌──────────────────────────────────────┐")
	fmt.Printf("   │ Accuracy:      %6.2f%%              │\n", metrics.Accuracy)
	fmt.Printf("   │ Precision:     %6.4f                │\n", metrics.Precision)
	fmt.Printf("   │ Recall:        %6.4f                │\n", metrics.Recall)
	fmt.Printf("   │ F1 Score:      %6.4f                │\n", metrics.F1Score)
	fmt.Printf("   │ Loss:          %6.4f                │\n", metrics.Loss)
	fmt.Println("   └──────────────────────────────────────┘")

	fmt.Println("\n🔮 Sample Predictions:")
	for i := 0; i < 5 && i < len(testImages); i++ {
		pred := cnn.Predict(testImages[i])
		actual := testLabels[i][0]

		predClass := "Vertical"
		actualClass := "Vertical"
		if pred > 0.5 {
			predClass = "Horizontal"
		}
		if actual > 0.5 {
			actualClass = "Horizontal"
		}

		correct := "✓"
		if (pred > 0.5) != (actual > 0.5) {
			correct = "✗"
		}

		fmt.Printf("   %s Sample %d: %.1f%% → %s (Actual: %s)\n",
			correct, i+1, pred*100, predClass, actualClass)
	}

	fmt.Println("\n✅ CNN Features Demonstrated:")
	fmt.Println("   • Conv2D layers with multiple filters")
	fmt.Println("   • MaxPooling for spatial downsampling")
	fmt.Println("   • ReLU activation in convolutional layers")
	fmt.Println("   • Flatten layer to connect CNN to dense layers")
	fmt.Println("   • Dense layers for classification")
	fmt.Println("   • End-to-end backpropagation through CNN")

	fmt.Println("\n═════════════════════════════════════════════════════════════════")
	fmt.Println("✅ CNN DEMO COMPLETE!")
	fmt.Println("═════════════════════════════════════════════════════════════════")
}

func createPatternImage(height, width int, pattern string) *tensor.Tensor3D {
	img := tensor.NewTensor3D(height, width, 1)

	switch pattern {
	case "vertical":
		for h := 0; h < height; h++ {
			for w := 0; w < width; w++ {
				if w%4 < 2 {
					img.Data[0][h][w] = 1.0
				} else {
					img.Data[0][h][w] = 0.0
				}
			}
		}
	case "horizontal":
		for h := 0; h < height; h++ {
			for w := 0; w < width; w++ {
				if h%4 < 2 {
					img.Data[0][h][w] = 1.0
				} else {
					img.Data[0][h][w] = 0.0
				}
			}
		}
	}

	return img
}
