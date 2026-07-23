// Command tokenizer_stream demonstrates the text package's byte-level BPE
// tokenizer paired with the streaming training path (data.DataLoader +
// Trainer.FitStream/TrainOnBatch) instead of Fit's single-materialized-
// tensor path: a tiny sentiment-style toy dataset ("does this sentence
// contain a positive word?") gets tokenized, embedded, and classified by
// an Embedding -> GRU -> LastTimestep -> Linear -> Sigmoid model, trained
// batch by batch through a data.DataLoader the same way a real out-of-core
// dataset would be — this example just keeps its (already-tokenized)
// samples in memory for simplicity, but the training loop itself never
// assumes that.
package main

import (
	"fmt"
	"math/rand"

	"github.com/stolzmi/neugo/data"
	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/text"
	"github.com/stolzmi/neugo/train"
)

const seqLen = 14 // fixed-length padding/truncation for every tokenized sentence — the longest sentence below encodes to 14 tokens, so nothing gets truncated

func main() {
	sentences := []string{
		"this movie was great",
		"a wonderful and great film",
		"truly great acting",
		"what a great story",
		"the movie was great fun",
		"this movie was terrible",
		"a boring and bad film",
		"truly bad acting",
		"what a bad story",
		"the movie was bad and boring",
	}
	labels := []float32{1, 1, 1, 1, 1, 0, 0, 0, 0, 0} // 1 = positive ("great"), 0 = negative ("bad")

	tok := text.TrainBPE(sentences, 300)
	fmt.Printf("trained tokenizer: vocab size %d\n\n", tok.VocabSize())

	// Encode every sentence once upfront, padded/truncated to seqLen — the
	// in-memory "dataset" that convert() below reads from per batch.
	tokenIDs := make([][]float32, len(sentences))
	for i, s := range sentences {
		ids := tok.Encode(s)
		row := make([]float32, seqLen)
		for j := 0; j < seqLen && j < len(ids); j++ {
			row[j] = float32(ids[j])
		}
		tokenIDs[i] = row
	}

	rng := nn.NewRNG(1)
	const dModel, hidden = 8, 8
	model, err := nn.Sequential([]int{len(sentences), seqLen},
		nn.Embedding(rng, tok.VocabSize(), dModel, nn.NormalInit(0, 0.3)),
		nn.GRU(rng, dModel, hidden, nn.XavierInit()),
		nn.LastTimestep(),
		nn.Linear(rng, hidden, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}

	trainer := train.New(model, train.Adam(0.03, 0.9, 0.999, 1e-8), train.BCELoss())
	loader := data.NewDataLoader(len(sentences), 4, rand.New(rand.NewSource(1)), true)
	convert := func(batchIdx []int) (*nn.Tensor, *nn.Tensor, error) {
		xb := nn.NewTensor([]int{len(batchIdx), seqLen})
		yb := nn.NewTensor([]int{len(batchIdx), 1})
		for i, idx := range batchIdx {
			copy(xb.Data[i*seqLen:(i+1)*seqLen], tokenIDs[idx])
			yb.Data[i] = labels[idx]
		}
		return xb, yb, nil
	}

	hist, err := trainer.FitStream(loader, convert, train.Epochs(400))
	if err != nil {
		fmt.Println("fit stream:", err)
		return
	}
	fmt.Printf("first train loss: %.4f\n", hist.TrainLoss[0])
	fmt.Printf("final train loss: %.4f\n\n", hist.TrainLoss[len(hist.TrainLoss)-1])

	fmt.Println("predictions:")
	for i, s := range sentences {
		x := nn.NewTensor([]int{1, seqLen})
		copy(x.Data, tokenIDs[i])
		pred, err := trainer.Predict(x)
		if err != nil {
			fmt.Println("predict:", err)
			return
		}
		fmt.Printf("  %-35q -> %.3f (actual %.0f)\n", s, pred.Data[0], labels[i])
	}
}
