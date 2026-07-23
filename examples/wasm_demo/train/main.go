// Command train (examples/wasm_demo/train) trains the tiny XOR model used
// by the WASM browser demo, saves it, and exports it to the dependency-
// free Go source in ../model/model_gen.go — regenerate that file by
// running `go run ./examples/wasm_demo/train` from the repo root whenever
// this training code changes. See ../README.md for the full pipeline
// (this step, then compiling ../wasm for GOOS=js GOARCH=wasm, then
// opening ../index.html in a browser).
package main

import (
	"fmt"
	"os"

	"github.com/stolzmi/neugo/export"
	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
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
		os.Exit(1)
	}

	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := train.New(model, train.Adam(0.05, 0.9, 0.999, 1e-8), train.BCELoss())
	hist, err := trainer.Fit(x, y, train.Epochs(2000), train.BatchSize(4), train.Seed(1))
	if err != nil {
		fmt.Println("fit:", err)
		os.Exit(1)
	}
	fmt.Printf("final train loss: %.4f\n", hist.TrainLoss[len(hist.TrainLoss)-1])

	modelPath := "examples/wasm_demo/train/xor_model.json"
	if err := nn.Save(model, modelPath); err != nil {
		fmt.Println("save:", err)
		os.Exit(1)
	}

	modelJSON, err := os.ReadFile(modelPath)
	if err != nil {
		fmt.Println("read saved model:", err)
		os.Exit(1)
	}
	genCode, err := export.GenerateGo(modelJSON, export.Options{Package: "model", FuncName: "Predict"})
	if err != nil {
		fmt.Println("export:", err)
		os.Exit(1)
	}
	outPath := "examples/wasm_demo/model/model_gen.go"
	if err := os.WriteFile(outPath, genCode, 0644); err != nil {
		fmt.Println("write generated model:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s\n", outPath)
}
