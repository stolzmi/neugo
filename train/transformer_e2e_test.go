// train/transformer_e2e_test.go
package train

import (
	"github.com/stolzmi/neugo/nn"
	"testing"
)

// TestTransformerModelConvergesOnToySequenceTask trains
// Embedding -> PositionalEmbedding -> TransformerBlock -> Flatten ->
// Linear -> Sigmoid, through the real Fit loop, on a tiny synthetic
// task: does the first token of a length-3 sequence equal the last
// token? This is the attention-stack equivalent of
// TestFrozenLayerWeightsDoNotMoveDuringFit — proof the whole composition
// actually trains, not just that each piece passes its own gradcheck in
// isolation.
func TestTransformerModelConvergesOnToySequenceTask(t *testing.T) {
	const vocabSize, seqLen, dModel, numHeads, ffHidden = 4, 3, 8, 2, 16

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
	block, err := nn.TransformerBlock(rng, dModel, numHeads, ffHidden, false, nn.XavierInit())
	if err != nil {
		t.Fatalf("TransformerBlock: %v", err)
	}
	model, err := nn.Sequential([]int{n, seqLen},
		nn.Embedding(rng, vocabSize, dModel, nn.NormalInit(0, 0.5)),
		nn.PositionalEmbedding(rng, seqLen, dModel, nn.NormalInit(0, 0.1)),
		block,
		nn.Flatten(),
		nn.Linear(rng, seqLen*dModel, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}

	trainer := New(model, Adam(0.01, 0.9, 0.999, 1e-8), BCELoss())
	hist, err := trainer.Fit(x, y, Epochs(300), BatchSize(16), Seed(2))
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	firstLoss, lastLoss := hist.TrainLoss[0], hist.TrainLoss[len(hist.TrainLoss)-1]
	if lastLoss >= firstLoss*0.5 {
		t.Fatalf("loss did not decrease meaningfully (first=%v last=%v) — attention stack may not be training correctly", firstLoss, lastLoss)
	}
}
