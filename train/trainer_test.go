package train

import (
	"github.com/stolzmi/neugo/nn"
	"testing"
)

func TestFitRequiresEpochs(t *testing.T) {
	rng := nn.NewRNG(1)
	model, _ := nn.Sequential([]int{1, 2}, nn.Linear(rng, 2, 1, nn.XavierInit()))
	trainer := New(model, SGD(0.1), MSELoss())
	x := nn.NewTensor([]int{1, 2})
	y := nn.NewTensor([]int{1, 1})
	_, err := trainer.Fit(x, y) // no Epochs(...) option
	if err == nil {
		t.Fatal("expected error when Epochs is not set, got nil")
	}
}

func TestXORConvergesWithAdam(t *testing.T) {
	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{4, 2},
		nn.Linear(rng, 2, 8, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 8, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	hist, err := trainer.Fit(x, y, Epochs(2000), BatchSize(4), Seed(1))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	finalLoss := hist.TrainLoss[len(hist.TrainLoss)-1]
	if finalLoss >= 0.05 {
		t.Fatalf("final train loss = %v after 2000 Adam epochs, want < 0.05", finalLoss)
	}
}

func TestXORConvergesWithSGD(t *testing.T) {
	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{4, 2},
		nn.Linear(rng, 2, 8, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 8, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := New(model, SGD(0.5), BCELoss())
	hist, err := trainer.Fit(x, y, Epochs(5000), BatchSize(4), Seed(1))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	finalLoss := hist.TrainLoss[len(hist.TrainLoss)-1]
	if finalLoss >= 0.05 {
		t.Fatalf("final train loss = %v after 5000 SGD epochs, want < 0.05", finalLoss)
	}
}

func TestMulticlassConvergesWithFusedSoftmax(t *testing.T) {
	rng := nn.NewRNG(2)
	model, err := nn.Sequential([]int{6, 2},
		nn.Linear(rng, 2, 8, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 8, 3, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	// Three well-separated 2D clusters, two samples each.
	x, _ := nn.NewTensorFromData([]float32{
		-2, -2, -1.8, -2.1,
		0, 2, 0.1, 1.9,
		2, -2, 1.9, -2.1,
	}, []int{6, 2})
	y, _ := nn.NewTensorFromData([]float32{
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
	}, []int{6, 3})

	ce := CrossEntropy()
	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), ce)
	if !ce.Fused() {
		t.Fatal("New did not detect trailing Softmax and enable fused CrossEntropy")
	}
	if _, err := trainer.Fit(x, y, Epochs(500), BatchSize(6), Seed(2)); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if metrics.Accuracy < 100 {
		t.Fatalf("Accuracy = %v after training on separable clusters, want 100", metrics.Accuracy)
	}
}

func TestPredictMatchesForwardInInferenceMode(t *testing.T) {
	rng := nn.NewRNG(3)
	model, err := nn.Sequential([]int{2, 3},
		nn.Linear(rng, 3, 4, nn.XavierInit()),
		nn.ReLU(),
		nn.Dropout(0.5),
		nn.Linear(rng, 4, 1, nn.XavierInit()),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{2, 3})
	trainer := New(model, SGD(0.1), MSELoss())
	p1, err := trainer.Predict(x)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	p2, _ := trainer.Predict(x)
	for i := range p1.Data {
		if p1.Data[i] != p2.Data[i] {
			t.Fatalf("Predict is non-deterministic in inference mode at index %d: %v vs %v", i, p1.Data[i], p2.Data[i])
		}
	}
}

func TestNewResetsFusedFlagWhenReusingCrossEntropyLoss(t *testing.T) {
	rng := nn.NewRNG(5)
	softmaxModel, err := nn.Sequential([]int{2, 2},
		nn.Linear(rng, 2, 3, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential (softmax model): %v", err)
	}
	nonSoftmaxModel, err := nn.Sequential([]int{2, 2},
		nn.Linear(rng, 2, 3, nn.XavierInit()),
	)
	if err != nil {
		t.Fatalf("Sequential (non-softmax model): %v", err)
	}

	ce := CrossEntropy()
	New(softmaxModel, SGD(0.1), ce)
	if !ce.Fused() {
		t.Fatal("New did not detect trailing Softmax and enable fused CrossEntropy")
	}

	// Reusing the same *CrossEntropyLoss instance with a model that does not
	// end in Softmax must reset fused back to false, not leave it stuck at
	// true from the previous New call.
	New(nonSoftmaxModel, SGD(0.1), ce)
	if ce.Fused() {
		t.Fatal("New left ce.Fused() == true after building a Trainer for a non-Softmax-ending model; fused flag must be reset to false")
	}
}

func TestEvaluateReturnsPopulatedMetrics(t *testing.T) {
	rng := nn.NewRNG(4)
	model, _ := nn.Sequential([]int{4, 2}, nn.Linear(rng, 2, 1, nn.XavierInit()), nn.Sigmoid())
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})
	trainer := New(model, SGD(0.1), BCELoss())
	m, err := trainer.Evaluate(x, y)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if m.ConfusionMatrix == nil {
		t.Fatal("Evaluate returned nil ConfusionMatrix")
	}
}
