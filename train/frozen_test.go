package train

import (
	"neugo/nn"
	"testing"
)

// TestFrozenLayerWeightsDoNotMoveDuringFit is the real end-to-end promise
// of nn.Frozen: wrap an early layer, train the model, and confirm its
// weights are byte-identical to their initial values afterward while the
// unfrozen layer's weights change and the loss still goes down — proving
// gradients kept flowing through the frozen layer to reach it, even
// though the frozen layer's own weights never got a Step applied.
func TestFrozenLayerWeightsDoNotMoveDuringFit(t *testing.T) {
	rng := nn.NewRNG(1)
	frozenLinear := nn.Linear(rng, 2, 8, nn.HeInit())
	trainableLinear := nn.Linear(rng, 8, 1, nn.XavierInit())
	model, err := nn.Sequential([]int{4, 2},
		nn.Frozen(frozenLinear),
		nn.ReLU(),
		trainableLinear,
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}

	frozenWBefore := append([]float32(nil), frozenLinear.W.Value.Data...)
	frozenBBefore := append([]float32(nil), frozenLinear.B.Value.Data...)
	trainableWBefore := append([]float32(nil), trainableLinear.W.Value.Data...)

	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	hist, err := trainer.Fit(x, y, Epochs(300), BatchSize(4), Seed(1))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	if len(model.Params()) != len(trainableLinear.Params()) {
		t.Fatalf("model.Params() has %d entries, want %d (frozen layer's params must be excluded)",
			len(model.Params()), len(trainableLinear.Params()))
	}

	for i, v := range frozenLinear.W.Value.Data {
		if v != frozenWBefore[i] {
			t.Fatalf("frozen layer's W.Value[%d] changed from %v to %v during Fit", i, frozenWBefore[i], v)
		}
	}
	for i, v := range frozenLinear.B.Value.Data {
		if v != frozenBBefore[i] {
			t.Fatalf("frozen layer's B.Value[%d] changed from %v to %v during Fit", i, frozenBBefore[i], v)
		}
	}

	changed := false
	for i, v := range trainableLinear.W.Value.Data {
		if v != trainableWBefore[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("trainable layer's weights did not change at all during Fit")
	}

	firstLoss, lastLoss := hist.TrainLoss[0], hist.TrainLoss[len(hist.TrainLoss)-1]
	if lastLoss >= firstLoss {
		t.Fatalf("loss did not decrease (first=%v last=%v) — gradient may not be flowing through the frozen layer", firstLoss, lastLoss)
	}
}
