package nn

import (
	"fmt"
	"strings"
)

func moduleTypeName(m Module) string {
	switch m.(type) {
	case *LinearLayer:
		return "Linear"
	case *Conv2DLayer:
		return "Conv2D"
	case *MaxPool2DLayer:
		return "MaxPool2D"
	case *AvgPool2DLayer:
		return "AvgPool2D"
	case *FlattenLayer:
		return "Flatten"
	case *DropoutLayer:
		return "Dropout"
	case *BatchNormLayer:
		return "BatchNorm"
	case *SoftmaxModule:
		return "Softmax"
	case *SequentialModel:
		return "Sequential"
	case *ResidualBlock:
		return "Residual"
	case *GroupNormLayer:
		return "GroupNorm"
	case *EmbeddingLayer:
		return "Embedding"
	case *Conv1DLayer:
		return "Conv1D"
	case *ConvTranspose2DLayer:
		return "ConvTranspose2D"
	case *FrozenModule:
		return "Frozen"
	case *ActivationModule:
		name := m.(*ActivationModule).Name()
		// Capitalize activation names for display
		switch name {
		case "relu":
			return "ReLU"
		case "sigmoid":
			return "Sigmoid"
		case "tanh":
			return "Tanh"
		case "leaky_relu":
			return "LeakyReLU"
		case "gelu":
			return "GELU"
		default:
			return name
		}
	default:
		return fmt.Sprintf("%T", m)
	}
}

func paramCountOf(m Module) int {
	count := 0
	for _, p := range m.Params() {
		count += p.Value.Size()
	}
	return count
}

// ParamCount returns the total number of trainable scalars across every
// module in model.
func ParamCount(model *SequentialModel) int {
	return paramCountOf(model)
}

// Summary re-runs the same OutputShape chain Sequential used at
// construction (safe — OutputShape is idempotent for already-built
// modules) to print each layer's output shape and parameter count plus a
// grand total. This and ProgressBar are the only stdout writers permitted
// by the Global Constraints.
func Summary(model *SequentialModel, inputShape []int) (string, error) {
	var b strings.Builder
	fmt.Fprintf(&b, "%-4s %-12s %-24s %10s\n", "#", "Layer", "Output Shape", "Params")
	fmt.Fprintln(&b, strings.Repeat("-", 54))

	shape := inputShape
	total := 0
	for i, m := range model.Modules() {
		out, err := m.OutputShape(shape)
		if err != nil {
			return "", fmt.Errorf("nn: Summary module %d: %w", i, err)
		}
		shape = out
		n := paramCountOf(m)
		total += n
		fmt.Fprintf(&b, "%-4d %-12s %-24v %10d\n", i, moduleTypeName(m), shape, n)
	}
	fmt.Fprintln(&b, strings.Repeat("-", 54))
	fmt.Fprintf(&b, "Total params: %d\n", total)
	return b.String(), nil
}
