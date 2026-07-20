package main

import (
	"fmt"
	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

func toTensor(rows [][]float32) *nn.Tensor {
	cols := len(rows[0])
	flat := make([]float32, len(rows)*cols)
	for i, row := range rows {
		copy(flat[i*cols:(i+1)*cols], row)
	}
	t, _ := nn.NewTensorFromData(flat, []int{len(rows), cols})
	return t
}

func main() {
	// A small linearly-separable-ish binary classification set, built
	// inline so cross-validation has something nontrivial to fold.
	dataRNG := nn.NewRNG(1)
	n := 60
	x := make([][]float32, n)
	y := make([][]float32, n)
	for i := 0; i < n; i++ {
		class := i % 2
		v := float32(class)*2 - 1 // -1 or 1
		x[i] = []float32{v + (dataRNG.Float32()-0.5)*0.3, v + (dataRNG.Float32()-0.5)*0.3}
		y[i] = []float32{float32(class)}
	}

	folds := train.KFoldSplits(dataRNG, x, y, 5, true)
	result, err := train.CrossValidate(folds, func(fold train.Fold) (train.Metrics, error) {
		modelRNG := nn.NewRNG(2)
		model, err := nn.Sequential([]int{1, 2},
			nn.Linear(modelRNG, 2, 8, nn.HeInit()),
			nn.ReLU(),
			nn.Linear(modelRNG, 8, 1, nn.XavierInit()),
			nn.Sigmoid(),
		)
		if err != nil {
			return train.Metrics{}, err
		}
		trainer := train.New(model, train.Adam(0.05, 0.9, 0.999, 1e-8), train.BCELoss())
		if _, err := trainer.Fit(toTensor(fold.TrainX), toTensor(fold.TrainY), train.Epochs(200), train.Seed(3)); err != nil {
			return train.Metrics{}, err
		}
		return trainer.Evaluate(toTensor(fold.TestX), toTensor(fold.TestY))
	})
	if err != nil {
		fmt.Println("cross-validate:", err)
		return
	}
	fmt.Printf("mean accuracy: %.2f%% (± %.2f)  mean F1: %.4f\n", result.MeanAccuracy, result.StdAccuracy, result.MeanF1)
	fmt.Printf("best fold: %d  worst fold: %d\n", result.BestFold, result.WorstFold)
}
