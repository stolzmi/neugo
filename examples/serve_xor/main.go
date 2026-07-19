package main

import (
	"context"
	"fmt"
	"neugo/nn"
	"neugo/serve"
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

	// Create server with holdout set for online learning
	holdoutSamples := []serve.Sample{
		{X: []float32{0, 0}, Y: []float32{0}},
		{X: []float32{0, 1}, Y: []float32{1}},
		{X: []float32{1, 0}, Y: []float32{1}},
		{X: []float32{1, 1}, Y: []float32{0}},
	}

	cfg := serve.Config{
		InputDim:     2,
		Loss:         train.BCELoss(),
		Holdout:      holdoutSamples,
		BufferSize:   256,
		RetrainEvery: 16,
		Epochs:       5,
		LearningRate: 0.05,
	}

	server, err := serve.New(model, cfg)
	if err != nil {
		fmt.Println("create server:", err)
		return
	}

	// Start online learning
	err = server.StartOnline(context.Background())
	if err != nil {
		fmt.Println("start online:", err)
		return
	}

	fmt.Println("\n=== Server started on :8080 ===")
	fmt.Println("\nCurl walkthrough:")
	fmt.Print(`

# 1. Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [0, 0]}'

# 2. Send feedback to train online
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{"x": [0, 1], "y": [1]}'

# 3. Check model generation (bumps after retraining)
curl http://localhost:8080/healthz

# 4. View metrics (includes model_generation and predict_total)
curl http://localhost:8080/metrics

# 5. Rollback to previous model version
curl -X POST http://localhost:8080/admin/rollback
`)

	err = server.ListenAndServe(":8080")
	if err != nil {
		fmt.Println("serve:", err)
	}
}
