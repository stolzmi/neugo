# Export Guide

The neugo export system converts trained models into single-file Go source code that runs inference with zero external dependencies (only Go's standard library `math` package, when needed by activations).

## Quick Start

### 1. Train and Save Your Model

```go
package main

import (
	"neugo/nn"
)

func main() {
	rng := nn.NewRNG(42)
	m, _ := nn.Sequential([]int{1, 784},
		nn.Linear(rng, 784, 128, nil), nn.ReLU(),
		nn.Linear(rng, 128, 10, nil), nn.Softmax(),
	)
	// ... training loop ...
	nn.Save(m, "my_model.json")
}
```

### 2. Export to Go

```bash
go run ./cmd/neugo export -model my_model.json -out model_gen.go -pkg model
```

Flags:
- `-model`: Path to the saved model JSON (required)
- `-out`: Path to output Go file (required)
- `-pkg`: Package name for generated code (default: "model")
- `-fn`: Function name for the inference function (default: "Predict")

### 3. Use the Generated Code

Drop `model_gen.go` into any Go project and call the generated function:

```go
package main

import (
	"fmt"
	"your_project/model"
)

func main() {
	input := []float32{0.1, 0.2, 0.3, ...} // length must match model input
	output := model.Predict(input)
	fmt.Printf("Predictions: %v\n", output)
}
```

The generated code:
- Contains no external imports (only `math` when needed)
- Is production-ready and safe for distribution
- Has zero compilation overhead
- Works on any platform Go supports

## Cross-Compilation

Generated code compiles to any target platform Go supports.

### Native Linux ARM64

```bash
GOOS=linux GOARCH=arm64 go build ./...
```

### Browser WASM

```bash
GOOS=js GOARCH=wasm go build ./...
```

For browser use, wrap the generated package in a JavaScript bridge using `syscall/js`:

```go
// wasm_wrapper.go
package main

import (
	"syscall/js"
	"your_project/model"
)

func predict(this js.Value, args []js.Value) interface{} {
	// Convert args[0] (JS array) to []float32
	input := jsArrayToFloat32Slice(args[0])
	output := model.Predict(input)
	// Convert output []float32 back to JS array
	return float32SliceToJSArray(output)
}

func main() {
	js.Global().Set("predict", js.FuncOf(predict))
	select {} // keep running
}

func jsArrayToFloat32Slice(jsArr js.Value) []float32 {
	length := jsArr.Length()
	result := make([]float32, length)
	for i := 0; i < length; i++ {
		result[i] = float32(jsArr.Index(i).Float())
	}
	return result
}

func float32SliceToJSArray(arr []float32) js.Value {
	jsArr := js.Global().Get("Float32Array").New(len(arr))
	for i, v := range arr {
		jsArr.SetIndex(i, v)
	}
	return jsArr
}
```

Compile to WASM:

```bash
GOOS=js GOARCH=wasm go build -o model.wasm
```

### TinyGo (Embedded/Edge)

```bash
tinygo build -target=<board> -o firmware
```

**Note:** TinyGo compilation is not covered by CI and may require platform-specific adjustments. Test on your target device.

## Supported Module Types

The export system currently supports:

- **Linear layers** — fully connected layers with weights and biases
- **ReLU activation** — rectified linear units
- **Sigmoid activation** — logistic sigmoid function
- **Tanh activation** — hyperbolic tangent
- **LeakyReLU activation** — leaky rectified linear units with alpha parameter
- **GELU activation** — Gaussian Error Linear Unit
- **Softmax activation** — softmax normalization across output
- **Dropout layers** — silently ignored at inference time (no-op)

## Unsupported Module Types (v1)

The following module types are not yet supported and will cause export to fail:

- **Conv2D** — 2D convolutional layers
- **MaxPool2D** — max pooling layers
- **AvgPool2D** — average pooling layers
- **Flatten** — reshape layers
- **BatchNorm** — batch normalization layers

Nested Sequential modules are also not supported.

## Exactness Guarantee

Generated Go code produces **bitwise identical results** to the neugo engine:

1. **Hex-float literals** — All weights and biases are stored as IEEE 754 hexadecimal float literals (e.g., `0x1.4p-2`), preserving exact bit patterns across any platform.

2. **Arithmetic parity** — Inference uses the same floating-point operations in the same order as the engine's forward pass, ensuring no numerical divergence.

3. **Verification** — Parity is verified by:
   - Running the same inputs through the neugo engine
   - Running them through compiled generated code
   - Comparing results with exact float32 equality (no epsilon tolerance)
   - Test: `go test ./export -run TestGeneratedCodeMatchesEngineExactly -v`

This guarantee means you can safely replace your neugo engine with the compiled binary and get identical predictions, bit-for-bit.

## Tips and Troubleshooting

### Size and Performance

- Generated code is typically 10–100 KB depending on model size
- Inference is CPU-only (no CUDA/TensorRT)
- Speed depends on layer count and activation functions; linear-only models are fastest

### Input/Output Shapes

- Input is always a flat `[]float32` slice of length equal to the model's input feature count
- Output is a flat `[]float32` slice of length equal to the model's output feature count
- Use helper functions in your code to reshape for your application (e.g., images as `[height][width][channels]`)

### Deployment

For production:
1. Test generated code on your target platform before deployment
2. Version the generated code along with your model version
3. Consider embedding both model and generated code in release artifacts
4. Re-export after training if the model architecture changes

