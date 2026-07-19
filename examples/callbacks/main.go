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
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	opt := train.Adam(0.05, 0.9, 0.999, 1e-8)
	scheduler := train.StepDecay(opt, 0.5, 200)
	earlyStop := train.EarlyStopping(50)
	checkpoint := train.ModelCheckpoint("callbacks_best_model.json", "loss", "min", true)
	progress := train.ProgressBar(1000, 100)

	trainer := train.New(model, opt, train.BCELoss())
	hist, err := trainer.Fit(x, y,
		train.Epochs(1000), train.BatchSize(4), train.Seed(1),
		train.Validation(x, y), // XOR is small enough to validate on itself for this demo
		train.Callbacks(scheduler, earlyStop, checkpoint, progress),
		train.WithSaveFunc(nn.Save),
	)
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("stopped after %d epochs (early stopping patience 50)\n", len(hist.TrainLoss))
	if checkpoint.LastError != nil {
		fmt.Println("checkpoint save error:", checkpoint.LastError)
	}
}
