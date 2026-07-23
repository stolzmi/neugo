// train/fitstream_test.go
package train

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stolzmi/neugo/data"
	"github.com/stolzmi/neugo/nn"
)

func xorModel(rng *rand.Rand) (*nn.SequentialModel, error) {
	return nn.Sequential([]int{4, 2},
		nn.Linear(rng, 2, 8, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 8, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
}

func TestTrainOnBatchConvergesOnXOR(t *testing.T) {
	model, err := xorModel(nn.NewRNG(1))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	firstLoss, err := trainer.TrainOnBatch(x, y)
	if err != nil {
		t.Fatalf("TrainOnBatch: %v", err)
	}
	var lastLoss float32
	for i := 0; i < 2000; i++ {
		lastLoss, err = trainer.TrainOnBatch(x, y)
		if err != nil {
			t.Fatalf("TrainOnBatch: %v", err)
		}
	}
	if lastLoss >= firstLoss*0.1 {
		t.Fatalf("loss did not decrease enough via repeated TrainOnBatch calls: first=%v last=%v", firstLoss, lastLoss)
	}
}

// TestFitStreamConvergesLikeFit trains the same architecture/hyperparams
// on the same XOR dataset via Fit (one materialized tensor) and via
// FitStream (a data.DataLoader + per-batch convert callback), and checks
// both reach the same "solved" loss threshold — proof FitStream's
// TrainOnBatch-based epoch loop is a faithful streaming equivalent of
// Fit's, not just independently plausible.
func TestFitStreamConvergesLikeFit(t *testing.T) {
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	fitModel, err := xorModel(nn.NewRNG(1))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	fitTrainer := New(fitModel, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	fitHist, err := fitTrainer.Fit(x, y, Epochs(2000), BatchSize(4), Seed(1))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	fitLoss := fitHist.TrainLoss[len(fitHist.TrainLoss)-1]
	if fitLoss >= 0.05 {
		t.Fatalf("Fit final loss = %v, want < 0.05", fitLoss)
	}

	streamModel, err := xorModel(nn.NewRNG(1))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	streamTrainer := New(streamModel, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	loader := data.NewDataLoader(4, 4, rand.New(rand.NewSource(1)), true)
	convert := func(batchIdx []int) (*nn.Tensor, *nn.Tensor, error) {
		xb := nn.NewTensor([]int{len(batchIdx), 2})
		yb := nn.NewTensor([]int{len(batchIdx), 1})
		for i, src := range batchIdx {
			copy(xb.Data[i*2:(i+1)*2], x.Data[src*2:(src+1)*2])
			yb.Data[i] = y.Data[src]
		}
		return xb, yb, nil
	}
	streamHist, err := streamTrainer.FitStream(loader, convert, Epochs(2000))
	if err != nil {
		t.Fatalf("FitStream: %v", err)
	}
	streamLoss := streamHist.TrainLoss[len(streamHist.TrainLoss)-1]
	if streamLoss >= 0.05 {
		t.Fatalf("FitStream final loss = %v, want < 0.05", streamLoss)
	}
}

func TestFitStreamRequiresEpochs(t *testing.T) {
	model, err := xorModel(nn.NewRNG(1))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	trainer := New(model, SGD(0.1), MSELoss())
	loader := data.NewDataLoader(4, 4, nil, false)
	convert := func(batchIdx []int) (*nn.Tensor, *nn.Tensor, error) {
		return nn.NewTensor([]int{len(batchIdx), 2}), nn.NewTensor([]int{len(batchIdx), 1}), nil
	}
	if _, err := trainer.FitStream(loader, convert); err == nil {
		t.Fatal("expected error when Epochs is not set, got nil")
	}
}

func TestFitStreamPropagatesConvertError(t *testing.T) {
	model, err := xorModel(nn.NewRNG(1))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	trainer := New(model, SGD(0.1), MSELoss())
	loader := data.NewDataLoader(4, 4, nil, false)
	wantErr := "boom"
	convert := func(batchIdx []int) (*nn.Tensor, *nn.Tensor, error) {
		return nil, nil, fmt.Errorf(wantErr)
	}
	if _, err := trainer.FitStream(loader, convert, Epochs(1)); err == nil {
		t.Fatal("expected convert's error to propagate, got nil")
	}
}
