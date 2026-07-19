package main

import (
	"fmt"
	"math/rand"
	"neugo/Network"
	"neugo/tensor"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          🎯  NEUGO CNN COMPLETE SHOWCASE  🎯                  ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	fmt.Println("\n📊 Creating Image Classification Dataset...")
	fmt.Println("   Task: Classify shapes (Circle vs Square)")

	trainImages := make([]*tensor.Tensor3D, 200)
	trainLabels := make([][]float32, 200)

	for i := 0; i < 100; i++ {
		img := createCircleImage(28, 28)
		trainImages[i] = img
		trainLabels[i] = []float32{0}
	}

	for i := 100; i < 200; i++ {
		img := createSquareImage(28, 28)
		trainImages[i] = img
		trainLabels[i] = []float32{1}
	}

	rand.Shuffle(len(trainImages), func(i, j int) {
		trainImages[i], trainImages[j] = trainImages[j], trainImages[i]
		trainLabels[i], trainLabels[j] = trainLabels[j], trainLabels[i]
	})

	testImages := make([]*tensor.Tensor3D, 40)
	testLabels := make([][]float32, 40)

	for i := 0; i < 20; i++ {
		img := createCircleImage(28, 28)
		testImages[i] = img
		testLabels[i] = []float32{0}
	}

	for i := 20; i < 40; i++ {
		img := createSquareImage(28, 28)
		testImages[i] = img
		testLabels[i] = []float32{1}
	}

	fmt.Printf("   Train: %d images (100 circles, 100 squares)\n", len(trainImages))
	fmt.Printf("   Test: %d images (20 circles, 20 squares)\n", len(testImages))

	fmt.Println("\n🏗️  CNN Architecture:")
	cnn := Network.NewCNN(28, 28, 1, Network.BinaryCrossEntropy)

	cnn.AddConv2D(16, 3, 1, 1, Network.ReLU)
	fmt.Println("   Layer 1: Conv2D(1→16, 3x3, stride=1, padding=1) + ReLU")
	fmt.Println("            Output: 28×28×16")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   Layer 2: MaxPool2D(2×2, stride=2)")
	fmt.Println("            Output: 14×14×16")

	cnn.AddConv2D(32, 3, 1, 1, Network.ReLU)
	fmt.Println("   Layer 3: Conv2D(16→32, 3x3, stride=1, padding=1) + ReLU")
	fmt.Println("            Output: 14×14×32")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   Layer 4: MaxPool2D(2×2, stride=2)")
	fmt.Println("            Output: 7×7×32")

	cnn.AddConv2D(64, 3, 1, 1, Network.ReLU)
	fmt.Println("   Layer 5: Conv2D(32→64, 3x3, stride=1, padding=1) + ReLU")
	fmt.Println("            Output: 7×7×64")

	cnn.AddFlatten()
	fmt.Println("   Layer 6: Flatten")
	fmt.Println("            Output: 3136")

	flattenedSize := 7 * 7 * 64
	denseLayers := []Network.Layer{
		Network.NewLayerWithActivation(flattenedSize, Network.Linear),
		Network.NewLayerWithActivation(128, Network.ReLU),
		Network.NewLayerWithActivation(64, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	cnn.SetDenseNetwork(denseLayers)
	fmt.Println("   Layer 7: Dense(3136→128) + ReLU")
	fmt.Println("   Layer 8: Dense(128→64) + ReLU")
	fmt.Println("   Layer 9: Dense(64→1) + Sigmoid")

	fmt.Printf("\n   Total Trainable Parameters: ~%d\n", estimateParameters(cnn))

	fmt.Println("\n🏋️  Training CNN (100 epochs)...")
	fmt.Println("Epoch |   Loss   | LR")
	fmt.Println("------|----------|--------")

	epochs := 100
	initialLR := float32(0.01)

	for epoch := 0; epoch < epochs; epoch++ {
		lr := initialLR * float32(1.0/(1.0+0.01*float64(epoch)))

		epochLoss := float32(0.0)
		for i := 0; i < len(trainImages); i++ {
			cnn.ForwardPass(trainImages[i])
			output := []float32{cnn.DenseNetwork.GetOutput()[0].Activation()}
			loss := cnn.Loss.Calculate(output, trainLabels[i])
			epochLoss += loss
			cnn.BackPropagation(trainImages[i], trainLabels[i], lr)
		}

		avgLoss := epochLoss / float32(len(trainImages))

		if epoch%20 == 0 || epoch == epochs-1 {
			fmt.Printf("%5d | %.6f | %.5f\n", epoch+1, avgLoss, lr)
		}
	}

	fmt.Println("\n📊 Final Evaluation...")
	trainMetrics := cnn.Evaluate(trainImages, trainLabels, 0.5)
	testMetrics := cnn.Evaluate(testImages, testLabels, 0.5)

	fmt.Println("\n┌─────────────────────────────────────────────────────────┐")
	fmt.Println("│                   TRAINING RESULTS                      │")
	fmt.Println("├─────────────────────────────────────────────────────────┤")
	fmt.Printf("│ Accuracy:       %6.2f%%                               │\n", trainMetrics.Accuracy)
	fmt.Printf("│ Precision:      %6.4f                                 │\n", trainMetrics.Precision)
	fmt.Printf("│ Recall:         %6.4f                                 │\n", trainMetrics.Recall)
	fmt.Printf("│ F1 Score:       %6.4f                                 │\n", trainMetrics.F1Score)
	fmt.Printf("│ Loss:           %6.4f                                 │\n", trainMetrics.Loss)
	fmt.Println("└─────────────────────────────────────────────────────────┘")

	fmt.Println("\n┌─────────────────────────────────────────────────────────┐")
	fmt.Println("│                    TEST RESULTS                         │")
	fmt.Println("├─────────────────────────────────────────────────────────┤")
	fmt.Printf("│ Accuracy:       %6.2f%%                               │\n", testMetrics.Accuracy)
	fmt.Printf("│ Precision:      %6.4f                                 │\n", testMetrics.Precision)
	fmt.Printf("│ Recall:         %6.4f                                 │\n", testMetrics.Recall)
	fmt.Printf("│ F1 Score:       %6.4f                                 │\n", testMetrics.F1Score)
	fmt.Printf("│ Loss:           %6.4f                                 │\n", testMetrics.Loss)
	fmt.Println("└─────────────────────────────────────────────────────────┘")

	fmt.Println("\n🔮 Sample Predictions (Test Set):")
	for i := 0; i < 10 && i < len(testImages); i++ {
		pred := cnn.Predict(testImages[i])
		actual := testLabels[i][0]

		predClass := "Circle"
		actualClass := "Circle"
		if pred > 0.5 {
			predClass = "Square"
		}
		if actual > 0.5 {
			actualClass = "Square"
		}

		correct := "✓"
		if (pred > 0.5) != (actual > 0.5) {
			correct = "✗"
		}

		fmt.Printf("   %s Sample %2d: %.1f%% confidence → %s (Actual: %s)\n",
			correct, i+1, pred*100, predClass, actualClass)
	}

	fmt.Println("\n✅ CNN Capabilities Demonstrated:")
	fmt.Println("   • Multi-layer convolutional architecture")
	fmt.Println("   • Spatial feature extraction with Conv2D")
	fmt.Println("   • Dimensionality reduction with MaxPooling")
	fmt.Println("   • Progressive feature learning (16→32→64 filters)")
	fmt.Println("   • ReLU activation for non-linearity")
	fmt.Println("   • Flatten layer connecting CNN to dense network")
	fmt.Println("   • Deep dense network for classification")
	fmt.Println("   • End-to-end gradient backpropagation")
	fmt.Println("   • Learning rate decay for better convergence")
	fmt.Println("   • Binary cross-entropy loss optimization")

	fmt.Println("\n═════════════════════════════════════════════════════════════════")
	fmt.Println("✅ CNN SHOWCASE COMPLETE!")
	fmt.Println("═════════════════════════════════════════════════════════════════")
}

func createCircleImage(height, width int) *tensor.Tensor3D {
	img := tensor.NewTensor3D(height, width, 1)
	centerH := float32(height) / 2
	centerW := float32(width) / 2
	radius := float32(height) / 3

	for h := 0; h < height; h++ {
		for w := 0; w < width; w++ {
			dist := float32((float32(h)-centerH)*(float32(h)-centerH) +
				(float32(w)-centerW)*(float32(w)-centerW))

			if dist <= radius*radius {
				img.Data[0][h][w] = 1.0
			} else {
				img.Data[0][h][w] = 0.0
			}

			if rand.Float32() < 0.05 {
				img.Data[0][h][w] += (rand.Float32() - 0.5) * 0.3
				if img.Data[0][h][w] < 0 {
					img.Data[0][h][w] = 0
				}
				if img.Data[0][h][w] > 1 {
					img.Data[0][h][w] = 1
				}
			}
		}
	}
	return img
}

func createSquareImage(height, width int) *tensor.Tensor3D {
	img := tensor.NewTensor3D(height, width, 1)
	margin := height / 4

	for h := 0; h < height; h++ {
		for w := 0; w < width; w++ {
			if h >= margin && h < height-margin && w >= margin && w < width-margin {
				img.Data[0][h][w] = 1.0
			} else {
				img.Data[0][h][w] = 0.0
			}

			if rand.Float32() < 0.05 {
				img.Data[0][h][w] += (rand.Float32() - 0.5) * 0.3
				if img.Data[0][h][w] < 0 {
					img.Data[0][h][w] = 0
				}
				if img.Data[0][h][w] > 1 {
					img.Data[0][h][w] = 1
				}
			}
		}
	}
	return img
}

func estimateParameters(cnn *Network.CNN) int {
	params := 0

	params += 16 * 1 * 3 * 3
	params += 32 * 16 * 3 * 3
	params += 64 * 32 * 3 * 3

	params += 3136 * 128
	params += 128 * 64
	params += 64 * 1

	params += 16 + 32 + 64
	params += 128 + 64 + 1

	return params
}
