package main

import (
	"fmt"
	"math"
	"math/rand"
	"neugo/Network"
	"neugo/tensor"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       🎯  CATS vs DOGS - Synthetic CNN Demo  🎯               ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	fmt.Println("\n📊 Generating Synthetic Image Dataset...")
	fmt.Println("   Task: Classify Cats (triangular features) vs Dogs (rectangular features)")
	fmt.Println("   Image size: 32×32 pixels, Grayscale")

	trainSize := 300
	testSize := 60

	trainImages := make([]*tensor.Tensor3D, trainSize)
	trainLabels := make([][]float32, trainSize)

	for i := 0; i < trainSize/2; i++ {
		trainImages[i] = createCatImage()
		trainLabels[i] = []float32{0}
	}
	for i := trainSize / 2; i < trainSize; i++ {
		trainImages[i] = createDogImage()
		trainLabels[i] = []float32{1}
	}

	rand.Shuffle(trainSize, func(i, j int) {
		trainImages[i], trainImages[j] = trainImages[j], trainImages[i]
		trainLabels[i], trainLabels[j] = trainLabels[j], trainLabels[i]
	})

	testImages := make([]*tensor.Tensor3D, testSize)
	testLabels := make([][]float32, testSize)

	for i := 0; i < testSize/2; i++ {
		testImages[i] = createCatImage()
		testLabels[i] = []float32{0}
	}
	for i := testSize / 2; i < testSize; i++ {
		testImages[i] = createDogImage()
		testLabels[i] = []float32{1}
	}

	fmt.Printf("   ✓ Generated %d training images (150 cats, 150 dogs)\n", trainSize)
	fmt.Printf("   ✓ Generated %d test images (30 cats, 30 dogs)\n", testSize)

	fmt.Println("\n🏗️  Building CNN for Image Classification...")
	cnn := Network.NewCNN(32, 32, 1, Network.BinaryCrossEntropy)

	cnn.AddConv2D(16, 3, 1, 1, Network.ReLU)
	fmt.Println("   Layer 1: Conv2D (1→16, 3×3) + ReLU → 32×32×16")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   Layer 2: MaxPool2D (2×2) → 16×16×16")

	cnn.AddConv2D(32, 3, 1, 1, Network.ReLU)
	fmt.Println("   Layer 3: Conv2D (16→32, 3×3) + ReLU → 16×16×32")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   Layer 4: MaxPool2D (2×2) → 8×8×32")

	cnn.AddConv2D(64, 3, 1, 1, Network.ReLU)
	fmt.Println("   Layer 5: Conv2D (32→64, 3×3) + ReLU → 8×8×64")

	cnn.AddMaxPool2D(2, 2)
	fmt.Println("   Layer 6: MaxPool2D (2×2) → 4×4×64")

	cnn.AddFlatten()
	fmt.Println("   Layer 7: Flatten → 1024")

	flattenedSize := 4 * 4 * 64
	denseLayers := []Network.Layer{
		Network.NewLayerWithActivation(flattenedSize, Network.Linear),
		Network.NewLayerWithActivation(128, Network.ReLU),
		Network.NewLayerWithActivation(64, Network.ReLU),
		Network.NewLayerWithActivation(1, Network.Sigmoid),
	}
	cnn.SetDenseNetwork(denseLayers)
	fmt.Println("   Layer 8-10: Dense (1024→128→64→1) + Sigmoid")

	fmt.Println("\n🏋️  Training CNN (50 epochs)...")
	fmt.Println("Epoch |   Loss   |  LR     | Val Acc")
	fmt.Println("------|----------|---------|--------")

	epochs := 50
	initialLR := float32(0.01)

	for epoch := 0; epoch < epochs; epoch++ {
		lr := initialLR * float32(math.Exp(-0.05*float64(epoch)))

		epochLoss := float32(0.0)

		for i := 0; i < len(trainImages); i++ {
			cnn.ForwardPass(trainImages[i])
			output := []float32{cnn.DenseNetwork.GetOutput()[0].Activation()}
			loss := cnn.Loss.Calculate(output, trainLabels[i])
			epochLoss += loss
			cnn.BackPropagation(trainImages[i], trainLabels[i], lr)
		}

		avgLoss := epochLoss / float32(len(trainImages))

		if epoch%10 == 0 || epoch == epochs-1 {
			valMetrics := cnn.Evaluate(testImages, testLabels, 0.5)
			fmt.Printf("%5d | %.6f | %.5f | %.2f%%\n",
				epoch+1, avgLoss, lr, valMetrics.Accuracy)
		}
	}

	fmt.Println("\n📊 Final Test Evaluation...")
	testMetrics := cnn.Evaluate(testImages, testLabels, 0.5)

	fmt.Println("\n┌─────────────────────────────────────────────────────────┐")
	fmt.Println("│              CATS vs DOGS TEST RESULTS                  │")
	fmt.Println("├─────────────────────────────────────────────────────────┤")
	fmt.Printf("│ Accuracy:       %6.2f%%                               │\n", testMetrics.Accuracy)
	fmt.Printf("│ Precision:      %6.4f                                 │\n", testMetrics.Precision)
	fmt.Printf("│ Recall:         %6.4f                                 │\n", testMetrics.Recall)
	fmt.Printf("│ F1 Score:       %6.4f                                 │\n", testMetrics.F1Score)
	fmt.Printf("│ Loss:           %6.4f                                 │\n", testMetrics.Loss)
	fmt.Println("└─────────────────────────────────────────────────────────┘")

	fmt.Println("\n🔮 Sample Predictions:")
	for i := 0; i < 10 && i < len(testImages); i++ {
		pred := cnn.Predict(testImages[i])
		actual := testLabels[i][0]

		predClass := "Cat 🐱"
		actualClass := "Cat 🐱"
		if pred > 0.5 {
			predClass = "Dog 🐶"
		}
		if actual > 0.5 {
			actualClass = "Dog 🐶"
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

	fmt.Println("\n✅ Demo Features:")
	fmt.Println("   • Synthetic image generation (no dataset download needed)")
	fmt.Println("   • Feature-based classification (triangles vs rectangles)")
	fmt.Println("   • 3-layer CNN architecture (16→32→64 filters)")
	fmt.Println("   • Progressive spatial downsampling (32→16→8→4)")
	fmt.Println("   • Deep dense network for final classification")
	fmt.Println("   • Exponential learning rate decay")
	fmt.Println("   • Real-time validation accuracy tracking")

	fmt.Println("\n═════════════════════════════════════════════════════════════════")
	fmt.Println("✅ CATS vs DOGS DEMO COMPLETE!")
	fmt.Println("   No dataset download required - fully synthetic!")
	fmt.Println("═════════════════════════════════════════════════════════════════")
}

func createCatImage() *tensor.Tensor3D {
	img := tensor.NewTensor3D(32, 32, 1)

	for h := 0; h < 32; h++ {
		for w := 0; w < 32; w++ {
			img.Data[0][h][w] = 0.0
		}
	}

	drawTriangle(img, 16, 8, 10)
	drawTriangle(img, 8, 20, 6)
	drawTriangle(img, 24, 20, 6)
	drawCircle(img, 10, 22, 2)
	drawCircle(img, 22, 22, 2)

	addNoise(img, 0.08)

	return img
}

func createDogImage() *tensor.Tensor3D {
	img := tensor.NewTensor3D(32, 32, 1)

	for h := 0; h < 32; h++ {
		for w := 0; w < 32; w++ {
			img.Data[0][h][w] = 0.0
		}
	}

	drawRectangle(img, 16, 16, 14, 14)
	drawRectangle(img, 8, 8, 4, 6)
	drawRectangle(img, 24, 8, 4, 6)
	drawCircle(img, 12, 18, 2)
	drawCircle(img, 20, 18, 2)

	addNoise(img, 0.08)

	return img
}

func drawTriangle(img *tensor.Tensor3D, centerH, centerW, size int) {
	for h := 0; h < size; h++ {
		for w := -h; w <= h; w++ {
			ph := centerH + h
			pw := centerW + w
			if ph >= 0 && ph < 32 && pw >= 0 && pw < 32 {
				img.Data[0][ph][pw] = 1.0
			}
		}
	}
}

func drawRectangle(img *tensor.Tensor3D, centerH, centerW, height, width int) {
	startH := centerH - height/2
	startW := centerW - width/2

	for h := startH; h < startH+height; h++ {
		for w := startW; w < startW+width; w++ {
			if h >= 0 && h < 32 && w >= 0 && w < 32 {
				img.Data[0][h][w] = 1.0
			}
		}
	}
}

func drawCircle(img *tensor.Tensor3D, centerH, centerW, radius int) {
	for h := centerH - radius; h <= centerH+radius; h++ {
		for w := centerW - radius; w <= centerW+radius; w++ {
			dist := math.Sqrt(float64((h-centerH)*(h-centerH) + (w-centerW)*(w-centerW)))
			if dist <= float64(radius) && h >= 0 && h < 32 && w >= 0 && w < 32 {
				img.Data[0][h][w] = 1.0
			}
		}
	}
}

func addNoise(img *tensor.Tensor3D, amount float32) {
	for h := 0; h < 32; h++ {
		for w := 0; w < 32; w++ {
			if rand.Float32() < amount {
				img.Data[0][h][w] += (rand.Float32() - 0.5) * 0.5
				if img.Data[0][h][w] < 0 {
					img.Data[0][h][w] = 0
				}
				if img.Data[0][h][w] > 1 {
					img.Data[0][h][w] = 1
				}
			}
		}
	}
}
