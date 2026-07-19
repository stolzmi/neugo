package main

import (
	"fmt"
	"neugo/data"
	"neugo/nn"
	"neugo/train"
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
	dataset, err := data.LoadCSV("dataset/wine_quality/winequality-red.csv", data.CSVConfig{
		Delimiter:       ';',
		HasHeader:       true,
		LabelColumn:     -1,
		LabelType:       "binary",
		BinaryThreshold: 6.0,
	})
	if err != nil {
		fmt.Println("load csv:", err)
		return
	}

	stats := data.CalculateStats(dataset.Features)
	normalized := data.NormalizeZScore(dataset.Features, stats)

	rng := nn.NewRNG(1)
	dataRNG := nn.NewRNG(2)
	split := data.SplitData(dataRNG, normalized, dataset.Labels, data.SplitConfig{
		TrainRatio: 0.7, ValRatio: 0.15, TestRatio: 0.15, Shuffle: true,
	})

	model, err := nn.Sequential([]int{1, dataset.NumFeatures},
		nn.Linear(rng, dataset.NumFeatures, 16, nn.HeInit()),
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(rng, 16, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}

	trainer := train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.BCELoss())
	hist, err := trainer.Fit(
		toTensor(split.TrainX), toTensor(split.TrainY),
		train.Epochs(100), train.BatchSize(32), train.Shuffle(true), train.Seed(3),
		train.Validation(toTensor(split.ValX), toTensor(split.ValY)),
		train.Callbacks(train.EarlyStopping(10)),
	)
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("trained %d epochs, final train loss %.4f\n", len(hist.TrainLoss), hist.TrainLoss[len(hist.TrainLoss)-1])

	metrics, err := trainer.Evaluate(toTensor(split.TestX), toTensor(split.TestY))
	if err != nil {
		fmt.Println("evaluate:", err)
		return
	}
	fmt.Printf("test accuracy: %.2f%%  f1: %.4f\n", metrics.Accuracy, metrics.F1Score)

	if err := nn.Save(model, "wine_quality_model.json"); err != nil {
		fmt.Println("save:", err)
	}
}
