package main

import (
	"fmt"
	"neugo/nn"
	"neugo/train"
)

func main() {
	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{4, 2},
		nn.Linear(rng, 2, 8, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 8, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}
	summary, _ := nn.Summary(model, []int{4, 2})
	fmt.Print(summary)

	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := train.New(model, train.Adam(0.05, 0.9, 0.999, 1e-8), train.BCELoss())
	hist, err := trainer.Fit(x, y, train.Epochs(2000), train.BatchSize(4), train.Seed(1))
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("final train loss: %.4f\n", hist.TrainLoss[len(hist.TrainLoss)-1])

	preds, err := trainer.Predict(x)
	if err != nil {
		fmt.Println("predict:", err)
		return
	}
	inputs := [][2]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	for i, in := range inputs {
		fmt.Printf("XOR(%v, %v) = %.4f\n", in[0], in[1], preds.Data[i])
	}
}
