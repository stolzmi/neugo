package train

import (
	"math/rand"
	"neugo/nn"
	"testing"
)

// syntheticImages builds n 8x8 single-channel images alternating between
// two easily-separable classes (bright left half vs. bright right half)
// with small per-pixel jitter, deterministic given rng.
func syntheticImages(rng *rand.Rand, n int) (*nn.Tensor, *nn.Tensor) {
	x := nn.NewTensor([]int{n, 8, 8, 1})
	y := nn.NewTensor([]int{n, 1})
	for i := 0; i < n; i++ {
		class := i % 2
		for h := 0; h < 8; h++ {
			for w := 0; w < 8; w++ {
				base := float32(0.1)
				if (w < 4) == (class == 0) {
					base = 0.9
				}
				jitter := (rng.Float32() - 0.5) * 0.05
				x.Data[(i*8+h)*8+w] = base + jitter
			}
		}
		y.Data[i] = float32(class)
	}
	return x, y
}

func TestSyntheticCNNConverges(t *testing.T) {
	dataRNG := nn.NewRNG(5)
	x, y := syntheticImages(dataRNG, 40)

	modelRNG := nn.NewRNG(6)
	model, err := nn.Sequential([]int{40, 8, 8, 1},
		nn.Conv2D(modelRNG, 1, 4, 3, nn.HeInit()),
		nn.ReLU(),
		nn.MaxPool2D(2, 2),
		nn.Flatten(),
		nn.Linear(modelRNG, 0, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}

	trainer := New(model, Adam(0.01, 0.9, 0.999, 1e-8), BCELoss())
	if _, err := trainer.Fit(x, y, Epochs(300), BatchSize(8), Shuffle(true), Seed(7)); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if metrics.Accuracy < 90 {
		t.Fatalf("Accuracy = %v after training on separable synthetic images, want >= 90", metrics.Accuracy)
	}
}
