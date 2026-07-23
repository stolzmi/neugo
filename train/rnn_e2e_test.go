// train/rnn_e2e_test.go
package train

import (
	"github.com/stolzmi/neugo/nn"
	"testing"
)

// TestLSTMModelConvergesOnToySequenceTask trains
// Embedding -> LSTM -> LastTimestep -> Linear -> Sigmoid through the real
// Fit loop on the same "does the first token equal the last token" toy
// task used by TestTransformerModelConvergesOnToySequenceTask — proof the
// new recurrent layers' BPTT actually trains end to end via the shared
// Trainer.Fit loop, not just that each layer passes its own gradcheck in
// isolation.
func TestLSTMModelConvergesOnToySequenceTask(t *testing.T) {
	const vocabSize, seqLen, dModel, hidden = 4, 3, 6, 8

	var xData []float32
	var yData []float32
	for a := 0; a < vocabSize; a++ {
		for b := 0; b < vocabSize; b++ {
			for c := 0; c < vocabSize; c++ {
				xData = append(xData, float32(a), float32(b), float32(c))
				if a == c {
					yData = append(yData, 1)
				} else {
					yData = append(yData, 0)
				}
			}
		}
	}
	n := len(yData)
	x, err := nn.NewTensorFromData(xData, []int{n, seqLen})
	if err != nil {
		t.Fatalf("NewTensorFromData x: %v", err)
	}
	y, err := nn.NewTensorFromData(yData, []int{n, 1})
	if err != nil {
		t.Fatalf("NewTensorFromData y: %v", err)
	}

	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{n, seqLen},
		nn.Embedding(rng, vocabSize, dModel, nn.NormalInit(0, 0.5)),
		nn.LSTM(rng, dModel, hidden, nn.XavierInit()),
		nn.LastTimestep(),
		nn.Linear(rng, hidden, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}

	trainer := New(model, Adam(0.02, 0.9, 0.999, 1e-8), BCELoss())
	hist, err := trainer.Fit(x, y, Epochs(300), BatchSize(16), Seed(2))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	firstLoss, lastLoss := hist.TrainLoss[0], hist.TrainLoss[len(hist.TrainLoss)-1]
	if lastLoss >= firstLoss*0.5 {
		t.Fatalf("loss did not decrease meaningfully (first=%v last=%v) — LSTM BPTT may not be training correctly", firstLoss, lastLoss)
	}
}

// TestGRUModelConvergesOnToySequenceTask is the GRU analogue of the LSTM
// test above, on the same toy task, guarding against a BPTT bug specific
// to GRU's reset-gate-gated candidate path.
func TestGRUModelConvergesOnToySequenceTask(t *testing.T) {
	const vocabSize, seqLen, dModel, hidden = 4, 3, 6, 8

	var xData []float32
	var yData []float32
	for a := 0; a < vocabSize; a++ {
		for b := 0; b < vocabSize; b++ {
			for c := 0; c < vocabSize; c++ {
				xData = append(xData, float32(a), float32(b), float32(c))
				if a == c {
					yData = append(yData, 1)
				} else {
					yData = append(yData, 0)
				}
			}
		}
	}
	n := len(yData)
	x, err := nn.NewTensorFromData(xData, []int{n, seqLen})
	if err != nil {
		t.Fatalf("NewTensorFromData x: %v", err)
	}
	y, err := nn.NewTensorFromData(yData, []int{n, 1})
	if err != nil {
		t.Fatalf("NewTensorFromData y: %v", err)
	}

	rng := nn.NewRNG(3)
	model, err := nn.Sequential([]int{n, seqLen},
		nn.Embedding(rng, vocabSize, dModel, nn.NormalInit(0, 0.5)),
		nn.GRU(rng, dModel, hidden, nn.XavierInit()),
		nn.LastTimestep(),
		nn.Linear(rng, hidden, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}

	trainer := New(model, Adam(0.02, 0.9, 0.999, 1e-8), BCELoss())
	hist, err := trainer.Fit(x, y, Epochs(300), BatchSize(16), Seed(4))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	firstLoss, lastLoss := hist.TrainLoss[0], hist.TrainLoss[len(hist.TrainLoss)-1]
	if lastLoss >= firstLoss*0.5 {
		t.Fatalf("loss did not decrease meaningfully (first=%v last=%v) — GRU BPTT may not be training correctly", firstLoss, lastLoss)
	}
}
