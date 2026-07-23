// Command sequence_rnn demonstrates the recurrent layers (LSTM, GRU) added
// alongside the existing feed-forward/attention layers: it trains a small
// Embedding -> LSTM -> LastTimestep -> Linear -> Sigmoid model to answer a
// toy sequence question — "does this length-3 sequence's first token equal
// its last?" — exercising full backpropagation-through-time via the same
// Trainer.Fit loop every other model in this library uses.
package main

import (
	"fmt"

	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

func main() {
	const vocabSize, seqLen, dModel, hidden = 4, 3, 6, 8

	var xData, yData []float32
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
	x, _ := nn.NewTensorFromData(xData, []int{n, seqLen})
	y, _ := nn.NewTensorFromData(yData, []int{n, 1})

	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{n, seqLen},
		nn.Embedding(rng, vocabSize, dModel, nn.NormalInit(0, 0.5)),
		nn.LSTM(rng, dModel, hidden, nn.XavierInit()),
		nn.LastTimestep(),
		nn.Linear(rng, hidden, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}
	summary, _ := nn.Summary(model, []int{n, seqLen})
	fmt.Print(summary)

	trainer := train.New(model, train.Adam(0.02, 0.9, 0.999, 1e-8), train.BCELoss())
	hist, err := trainer.Fit(x, y, train.Epochs(300), train.BatchSize(16), train.Seed(2))
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("first train loss: %.4f\n", hist.TrainLoss[0])
	fmt.Printf("final train loss: %.4f\n", hist.TrainLoss[len(hist.TrainLoss)-1])

	preds, err := trainer.Predict(x)
	if err != nil {
		fmt.Println("predict:", err)
		return
	}
	fmt.Println("\nsample predictions (sequence -> P(first==last), actual):")
	for i := 0; i < 6; i++ {
		fmt.Printf("  [%.0f %.0f %.0f] -> %.3f (actual %.0f)\n",
			x.Data[i*seqLen], x.Data[i*seqLen+1], x.Data[i*seqLen+2], preds.Data[i], y.Data[i])
	}
}
