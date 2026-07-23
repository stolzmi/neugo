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
	case *InstanceNormLayer:
		return "InstanceNorm"
	case *GroupNormLayer:
		return "GroupNorm"
	case *LayerNormLayer:
		return "LayerNorm"
	case *RMSNormLayer:
		return "RMSNorm"
	case *AdaptiveAvgPool2DLayer:
		return "AdaptiveAvgPool2D"
	case *GlobalMaxPool2DLayer:
		return "GlobalMaxPool2D"
	case *EmbeddingLayer:
		return "Embedding"
	case *MultiHeadAttentionLayer:
		return "MultiHeadAttention"
	case *RotaryMultiHeadAttentionLayer:
		return "RotaryMultiHeadAttention"
	case *PositionalEmbeddingLayer:
		return "PositionalEmbedding"
	case *Conv1DLayer:
		return "Conv1D"
	case *ConvTranspose2DLayer:
		return "ConvTranspose2D"
	case *FrozenModule:
		return "Frozen"
	case *PReLULayer:
		return "PReLU"
	case *RNNLayer:
		return "RNN"
	case *LSTMLayer:
		return "LSTM"
	case *GRULayer:
		return "GRU"
	case *LastTimestepLayer:
		return "LastTimestep"
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
		case "elu":
			return "ELU"
		case "selu":
			return "SELU"
		case "silu":
			return "SiLU"
		case "mish":
			return "Mish"
		case "softplus":
			return "Softplus"
		case "hardswish":
			return "Hardswish"
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

// ShapeTrace re-runs the same OutputShape chain Sequential used at
// construction (safe — OutputShape is idempotent for already-built
// modules) and returns each module's output shape as data, in order —
// the programmatic counterpart to Summary's formatted string, for
// tests/tooling that want to assert on or otherwise consume the shapes
// directly rather than parse a printed table.
func ShapeTrace(model *SequentialModel, inputShape []int) ([][]int, error) {
	shape := inputShape
	modules := model.Modules()
	shapes := make([][]int, len(modules))
	for i, m := range modules {
		out, err := m.OutputShape(shape)
		if err != nil {
			return nil, fmt.Errorf("nn: ShapeTrace module %d: %w", i, err)
		}
		shape = out
		shapes[i] = out
	}
	return shapes, nil
}

// Summary re-runs the same OutputShape chain Sequential used at
// construction (safe — OutputShape is idempotent for already-built
// modules) to print each layer's output shape and parameter count plus a
// grand total. This is the one stdout writer in nn; train has several
// opt-in ones (ProgressBar, TUI, GradientHistogram).
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
