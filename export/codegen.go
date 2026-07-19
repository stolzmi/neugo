package export

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/format"
	"strconv"
)

type Options struct {
	Package  string
	FuncName string
}

// paramDoc matches the structure from nn/serialize.go
type paramDoc struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

// moduleDoc matches the structure from nn/serialize.go
type moduleDoc struct {
	Type    string              `json:"type"`
	Config  json.RawMessage     `json:"config,omitempty"`
	Params  map[string]paramDoc `json:"params,omitempty"`
	Modules []moduleDoc         `json:"modules,omitempty"`
}

// Linear config
type linearConfig struct {
	InFeatures  int `json:"in_features"`
	OutFeatures int `json:"out_features"`
}

// LeakyReLU config
type leakyReLUConfig struct {
	Alpha float32 `json:"alpha"`
}

func GenerateGo(modelJSON []byte, opts Options) ([]byte, error) {
	if opts.Package == "" {
		opts.Package = "model"
	}
	if opts.FuncName == "" {
		opts.FuncName = "Predict"
	}

	var root moduleDoc
	if err := json.Unmarshal(modelJSON, &root); err != nil {
		return nil, fmt.Errorf("export: failed to unmarshal model JSON: %w", err)
	}

	if root.Type != "sequential" {
		return nil, fmt.Errorf("export: root module must be sequential, got %s", root.Type)
	}

	// Track used activations and build code
	usedActivations := make(map[string]bool)
	var declarations bytes.Buffer
	var body bytes.Buffer

	varIndex := 0
	inputDim := -1     // Captured from first linear layer's in_features
	outputDim := -1    // Captured from last linear layer's out_features
	prevLinearOut := -1 // For chain-width validation

	for _, mod := range root.Modules {
		switch mod.Type {
		case "linear":
			var cfg linearConfig
			if err := json.Unmarshal(mod.Config, &cfg); err != nil {
				return nil, fmt.Errorf("export: failed to parse linear config: %w", err)
			}

			// Validate chain-width compatibility
			if prevLinearOut != -1 && cfg.InFeatures != prevLinearOut {
				return nil, fmt.Errorf("export: linear chain width mismatch: previous layer output %d does not match current layer input %d", prevLinearOut, cfg.InFeatures)
			}

			// Validate params
			if w, ok := mod.Params["W"]; ok {
				if len(w.Data) != cfg.InFeatures*cfg.OutFeatures {
					return nil, fmt.Errorf("export: linear W size mismatch: got %d, want %d", len(w.Data), cfg.InFeatures*cfg.OutFeatures)
				}
				if len(w.Shape) != 2 || w.Shape[0] != cfg.InFeatures || w.Shape[1] != cfg.OutFeatures {
					return nil, fmt.Errorf("export: linear W shape mismatch")
				}
			} else {
				return nil, fmt.Errorf("export: linear layer missing W param")
			}

			if b, ok := mod.Params["B"]; ok {
				if len(b.Data) != cfg.OutFeatures {
					return nil, fmt.Errorf("export: linear B size mismatch: got %d, want %d", len(b.Data), cfg.OutFeatures)
				}
				if len(b.Shape) != 1 || b.Shape[0] != cfg.OutFeatures {
					return nil, fmt.Errorf("export: linear B shape mismatch")
				}
			} else {
				return nil, fmt.Errorf("export: linear layer missing B param")
			}

			// Emit variable declarations
			wVar := fmt.Sprintf("w%d", varIndex)
			bVar := fmt.Sprintf("b%d", varIndex)
			emitWeightVar(&declarations, wVar, mod.Params["W"].Data)
			emitWeightVar(&declarations, bVar, mod.Params["B"].Data)

			// Capture inputDim from first linear layer
			if inputDim == -1 {
				inputDim = cfg.InFeatures
			}

			// Update output dimension (will hold the last linear's output at the end)
			outputDim = cfg.OutFeatures

			// Emit linear call
			fmt.Fprintf(&body, "\tcur = nnLinear(cur, %s, %s, %d, %d)\n", wVar, bVar, cfg.InFeatures, cfg.OutFeatures)
			prevLinearOut = cfg.OutFeatures
			varIndex++

		case "relu":
			usedActivations["relu"] = true
			body.WriteString("\tnnReLU(cur)\n")

		case "sigmoid":
			usedActivations["sigmoid"] = true
			body.WriteString("\tnnSigmoid(cur)\n")

		case "tanh":
			usedActivations["tanh"] = true
			body.WriteString("\tnnTanh(cur)\n")

		case "leaky_relu":
			usedActivations["leaky_relu"] = true
			var cfg leakyReLUConfig
			if err := json.Unmarshal(mod.Config, &cfg); err != nil {
				return nil, fmt.Errorf("export: failed to parse leaky_relu config: %w", err)
			}
			alphaHex := strconv.FormatFloat(float64(cfg.Alpha), 'x', -1, 32)
			fmt.Fprintf(&body, "\tnnLeakyReLU(cur, %s)\n", alphaHex)

		case "gelu":
			usedActivations["gelu"] = true
			body.WriteString("\tnnGELU(cur)\n")

		case "softmax":
			usedActivations["softmax"] = true
			body.WriteString("\tnnSoftmax(cur)\n")

		case "dropout":
			// Emit nothing at inference time
			continue

		case "conv2d", "maxpool2d", "avgpool2d", "flatten", "batchnorm":
			return nil, fmt.Errorf("export: unsupported module type: %s", mod.Type)

		case "sequential":
			return nil, fmt.Errorf("export: unsupported module type: nested sequential")

		default:
			return nil, fmt.Errorf("export: unknown module type: %s", mod.Type)
		}
	}

	// Ensure at least one linear layer was found
	if inputDim == -1 || outputDim == -1 {
		return nil, fmt.Errorf("export: model contains no linear layer; cannot determine input size")
	}

	// Build the output
	var output bytes.Buffer
	output.WriteString("// Code generated by neugo export. DO NOT EDIT.\n")
	fmt.Fprintf(&output, "package %s\n\n", opts.Package)

	// Emit import if needed
	if needsMathImport(usedActivations) {
		output.WriteString("import \"math\"\n\n")
	}

	// Emit variable declarations
	output.Write(declarations.Bytes())

	// Emit inference function
	fmt.Fprintf(&output, "// %s runs inference. len(input) must be %d;\n", opts.FuncName, inputDim)
	fmt.Fprintf(&output, "// returns %d values.\n", outputDim)
	fmt.Fprintf(&output, "func %s(input []float32) []float32 {\n", opts.FuncName)
	output.WriteString("\tcur := input\n")
	output.Write(body.Bytes())
	output.WriteString("\treturn cur\n")
	output.WriteString("}\n\n")

	// Emit helper functions
	emitLinearHelper(&output)
	if usedActivations["relu"] {
		emitReLUHelper(&output)
	}
	if usedActivations["sigmoid"] {
		emitSigmoidHelper(&output)
	}
	if usedActivations["tanh"] {
		emitTanhHelper(&output)
	}
	if usedActivations["leaky_relu"] {
		emitLeakyReLUHelper(&output)
	}
	if usedActivations["gelu"] {
		emitGELUHelper(&output)
	}
	if usedActivations["softmax"] {
		emitSoftmaxHelper(&output)
	}

	// Format the output
	formatted, err := format.Source(output.Bytes())
	if err != nil {
		return output.Bytes(), fmt.Errorf("export: failed to format generated code: %w", err)
	}

	return formatted, nil
}

func emitWeightVar(buf *bytes.Buffer, varName string, data []float32) {
	fmt.Fprintf(buf, "var %s = []float32{", varName)
	for i, v := range data {
		if i > 0 {
			buf.WriteString(", ")
		}
		hex := strconv.FormatFloat(float64(v), 'x', -1, 32)
		buf.WriteString(hex)
	}
	buf.WriteString("}\n")
}

func needsMathImport(usedActivations map[string]bool) bool {
	return usedActivations["sigmoid"] || usedActivations["tanh"] ||
		usedActivations["gelu"] || usedActivations["softmax"]
}

func emitLinearHelper(buf *bytes.Buffer) {
	buf.WriteString(`func nnLinear(x, w, b []float32, in, out int) []float32 {
	result := make([]float32, out)
	for o := 0; o < out; o++ {
		sum := b[o]
		for i := 0; i < in; i++ {
			sum += x[i] * w[i*out+o]
		}
		result[o] = sum
	}
	return result
}

`)
}

func emitReLUHelper(buf *bytes.Buffer) {
	buf.WriteString(`func nnReLU(x []float32) {
	for i := range x {
		if x[i] > 0 {
			// positive: keep value
		} else {
			x[i] = 0
		}
	}
}

`)
}

func emitSigmoidHelper(buf *bytes.Buffer) {
	buf.WriteString(`func nnSigmoid(x []float32) {
	for i := range x {
		x[i] = float32(1 / (1 + math.Exp(float64(-x[i]))))
	}
}

`)
}

func emitTanhHelper(buf *bytes.Buffer) {
	buf.WriteString(`func nnTanh(x []float32) {
	for i := range x {
		x[i] = float32(math.Tanh(float64(x[i])))
	}
}

`)
}

func emitLeakyReLUHelper(buf *bytes.Buffer) {
	buf.WriteString(`func nnLeakyReLU(x []float32, alpha float32) {
	for i := range x {
		if x[i] > 0 {
			// positive: keep value
		} else {
			x[i] = alpha * x[i]
		}
	}
}

`)
}

func emitGELUHelper(buf *bytes.Buffer) {
	buf.WriteString(`func nnGELU(x []float32) {
	for i := range x {
		x[i] = 0.5 * x[i] * (1 + float32(math.Erf(float64(x[i])/math.Sqrt2)))
	}
}

`)
}

func emitSoftmaxHelper(buf *bytes.Buffer) {
	buf.WriteString(`func nnSoftmax(x []float32) {
	classes := len(x)
	maxV := x[0]
	for c := 1; c < classes; c++ {
		if v := x[c]; v > maxV {
			maxV = v
		}
	}
	var sum float32
	for c := 0; c < classes; c++ {
		e := float32(math.Exp(float64(x[c] - maxV)))
		x[c] = e
		sum += e
	}
	for c := 0; c < classes; c++ {
		x[c] /= sum
	}
}

`)
}
