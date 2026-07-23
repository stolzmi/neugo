//go:build js

// Command wasm (examples/wasm_demo/wasm) is the syscall/js bridge
// exposing the exported XOR model to JavaScript — the exact pattern
// documented in docs/EXPORT_GUIDE.md's "Browser WASM" section, made into
// a complete runnable example. Build for the browser with:
//
//	GOOS=js GOARCH=wasm go build -o predict.wasm ./examples/wasm_demo/wasm
//
// then open ../index.html (see ../README.md for the full pipeline,
// including where to find wasm_exec.js).
package main

import (
	"syscall/js"

	"github.com/stolzmi/neugo/examples/wasm_demo/model"
)

func predict(this js.Value, args []js.Value) any {
	input := jsArrayToFloat32Slice(args[0])
	output := model.Predict(input)
	return float32SliceToJSArray(output)
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

func main() {
	js.Global().Set("predict", js.FuncOf(predict))
	select {} // keep the WASM instance alive to answer callbacks from JS
}
