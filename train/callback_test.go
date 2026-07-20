package train

import (
	"errors"
	"github.com/stolzmi/neugo/nn"
	"testing"
)

func TestEarlyStoppingTriggersAfterPatience(t *testing.T) {
	es := EarlyStopping(2)
	p := newTestParam([]float32{1}, []float32{0})
	params := []*nn.Param{p}

	es.OnEpochEnd(0, 1.0, nil, params) // improves (inf -> 1.0)
	if es.ShouldStop {
		t.Fatal("ShouldStop after epoch 0, want false")
	}
	es.OnEpochEnd(1, 1.0, nil, params) // no improvement, counter=1
	if es.ShouldStop {
		t.Fatal("ShouldStop after epoch 1, want false")
	}
	es.OnEpochEnd(2, 1.0, nil, params) // no improvement, counter=2 >= patience
	if !es.ShouldStop {
		t.Fatal("ShouldStop after epoch 2, want true")
	}
}

func TestEarlyStoppingRestoresBestWeights(t *testing.T) {
	es := EarlyStopping(1)
	p := newTestParam([]float32{1}, []float32{0})
	params := []*nn.Param{p}

	es.OnEpochEnd(0, 1.0, nil, params) // best = 1.0, snapshot value=[1]
	p.Value.Data[0] = 999              // simulate further training moving weights
	es.OnEpochEnd(1, 2.0, nil, params) // worse, no new snapshot

	es.RestoreBestWeights(params)
	if p.Value.Data[0] != 1 {
		t.Fatalf("Value[0] after RestoreBestWeights = %v, want 1", p.Value.Data[0])
	}
}

func TestModelCheckpointSavesOnlyOnImprovement(t *testing.T) {
	mc := ModelCheckpoint("model.json", "loss", "min", true)
	saves := 0
	mc.Save = func(path string) error { saves++; return nil }

	mc.OnEpochEnd(0, 0, &Metrics{Loss: 1.0}, nil) // improves -> save
	mc.OnEpochEnd(1, 0, &Metrics{Loss: 1.5}, nil) // worse -> no save
	mc.OnEpochEnd(2, 0, &Metrics{Loss: 0.5}, nil) // improves -> save

	if saves != 2 {
		t.Fatalf("saves = %d, want 2", saves)
	}
}

func TestModelCheckpointRecordsSaveError(t *testing.T) {
	mc := ModelCheckpoint("model.json", "loss", "min", true)
	wantErr := errors.New("disk full")
	mc.Save = func(path string) error { return wantErr }
	mc.OnEpochEnd(0, 0, &Metrics{Loss: 1.0}, nil)
	if mc.LastError != wantErr {
		t.Fatalf("LastError = %v, want %v", mc.LastError, wantErr)
	}
}

func TestHistoryRecordsLossesInOrder(t *testing.T) {
	h := NewHistory()
	h.OnEpochEnd(0, 0.9, &Metrics{Loss: 0.8, Accuracy: 50}, nil)
	h.OnEpochEnd(1, 0.5, &Metrics{Loss: 0.4, Accuracy: 70}, nil)
	wantTrain := []float32{0.9, 0.5}
	wantVal := []float32{0.8, 0.4}
	for i := range wantTrain {
		if h.TrainLoss[i] != wantTrain[i] {
			t.Errorf("TrainLoss[%d] = %v, want %v", i, h.TrainLoss[i], wantTrain[i])
		}
		if h.ValLoss[i] != wantVal[i] {
			t.Errorf("ValLoss[%d] = %v, want %v", i, h.ValLoss[i], wantVal[i])
		}
	}
}

func TestCallbackListFanOut(t *testing.T) {
	h1, h2 := NewHistory(), NewHistory()
	cl := NewCallbackList(h1, h2)
	cl.OnEpochEnd(0, 1.0, nil, nil)
	if len(h1.TrainLoss) != 1 || len(h2.TrainLoss) != 1 {
		t.Fatalf("expected both callbacks to observe the epoch end, got h1=%v h2=%v", h1.TrainLoss, h2.TrainLoss)
	}
}
