package Network

import "math"

// LossType represents the type of loss function
type LossType int

const (
	MSE LossType = iota
	BinaryCrossEntropy
	CategoricalCrossEntropy
	MAE
)

// LossFunction calculates loss and its gradient
type LossFunction struct {
	Type          LossType
	Calculate     func(predictions, labels []float32) float32
	Gradient      func(prediction, label float32) float32
	GradientBatch func(predictions, labels []float32) []float32
}

// Mean Squared Error (MSE)
func mseCalculate(predictions, labels []float32) float32 {
	sum := float32(0)
	for i := range predictions {
		diff := predictions[i] - labels[i]
		sum += diff * diff
	}
	return sum / float32(len(predictions))
}

func mseGradient(prediction, label float32) float32 {
	// Derivative of MSE: 2(y_pred - y_true) / n
	// We simplify to (y_pred - y_true) as the constant gets absorbed into learning rate
	return prediction - label
}

func mseGradientBatch(predictions, labels []float32) []float32 {
	gradients := make([]float32, len(predictions))
	for i := range predictions {
		gradients[i] = mseGradient(predictions[i], labels[i])
	}
	return gradients
}

// Binary Cross-Entropy (for binary classification)
func bceLoss(predictions, labels []float32) float32 {
	epsilon := float32(1e-7) // Small value to prevent log(0)
	sum := float32(0)
	for i := range predictions {
		p := predictions[i]
		// Clip predictions to prevent log(0)
		p = float32(math.Max(float64(epsilon), math.Min(float64(1-epsilon), float64(p))))
		y := labels[i]
		sum += -(y*float32(math.Log(float64(p))) + (1-y)*float32(math.Log(float64(1-p))))
	}
	return sum / float32(len(predictions))
}

// Weighted Binary Cross-Entropy (for imbalanced binary classification)
func bceWeightedLoss(predictions, labels []float32, posWeight float32) float32 {
	epsilon := float32(1e-7) // Small value to prevent log(0)
	sum := float32(0)
	for i := range predictions {
		p := predictions[i]
		// Clip predictions to prevent log(0)
		p = float32(math.Max(float64(epsilon), math.Min(float64(1-epsilon), float64(p))))
		y := labels[i]
		// Apply weight to positive class
		sum += -(y*posWeight*float32(math.Log(float64(p))) + (1-y)*float32(math.Log(float64(1-p))))
	}
	return sum / float32(len(predictions))
}

func bceWeightedGradient(prediction, label, posWeight float32) float32 {
	epsilon := float32(1e-7)
	p := float32(math.Max(float64(epsilon), math.Min(float64(1-epsilon), float64(prediction))))
	// Weighted gradient for imbalanced data
	if label > 0.5 {
		return posWeight * (p - label)
	}
	return p - label
}

func bceGradient(prediction, label float32) float32 {
	epsilon := float32(1e-7)
	p := float32(math.Max(float64(epsilon), math.Min(float64(1-epsilon), float64(prediction))))
	// Gradient: (p - y) / (p * (1 - p))
	// Simplified for sigmoid output: p - y
	return p - label
}

func bceGradientBatch(predictions, labels []float32) []float32 {
	gradients := make([]float32, len(predictions))
	for i := range predictions {
		gradients[i] = bceGradient(predictions[i], labels[i])
	}
	return gradients
}

// Categorical Cross-Entropy (for multi-class classification)
func cceLoss(predictions, labels []float32) float32 {
	epsilon := float32(1e-7)
	sum := float32(0)
	for i := range predictions {
		p := predictions[i]
		p = float32(math.Max(float64(epsilon), math.Min(float64(1-epsilon), float64(p))))
		if labels[i] > 0 {
			sum += -labels[i] * float32(math.Log(float64(p)))
		}
	}
	return sum
}

func cceGradient(prediction, label float32) float32 {
	epsilon := float32(1e-7)
	p := float32(math.Max(float64(epsilon), math.Min(float64(1-epsilon), float64(prediction))))
	// Gradient for softmax + CCE: p - y
	return p - label
}

func cceGradientBatch(predictions, labels []float32) []float32 {
	gradients := make([]float32, len(predictions))
	for i := range predictions {
		gradients[i] = cceGradient(predictions[i], labels[i])
	}
	return gradients
}

// Mean Absolute Error (MAE)
func maeCalculate(predictions, labels []float32) float32 {
	sum := float32(0)
	for i := range predictions {
		sum += float32(math.Abs(float64(predictions[i] - labels[i])))
	}
	return sum / float32(len(predictions))
}

func maeGradient(prediction, label float32) float32 {
	// Derivative: sign(y_pred - y_true)
	diff := prediction - label
	if diff > 0 {
		return 1
	} else if diff < 0 {
		return -1
	}
	return 0
}

func maeGradientBatch(predictions, labels []float32) []float32 {
	gradients := make([]float32, len(predictions))
	for i := range predictions {
		gradients[i] = maeGradient(predictions[i], labels[i])
	}
	return gradients
}

// GetLossFunction returns the loss function for a given type
func GetLossFunction(lossType LossType) LossFunction {
	switch lossType {
	case MSE:
		return LossFunction{
			Type:          MSE,
			Calculate:     mseCalculate,
			Gradient:      mseGradient,
			GradientBatch: mseGradientBatch,
		}
	case BinaryCrossEntropy:
		return LossFunction{
			Type:          BinaryCrossEntropy,
			Calculate:     bceLoss,
			Gradient:      bceGradient,
			GradientBatch: bceGradientBatch,
		}
	case CategoricalCrossEntropy:
		return LossFunction{
			Type:          CategoricalCrossEntropy,
			Calculate:     cceLoss,
			Gradient:      cceGradient,
			GradientBatch: cceGradientBatch,
		}
	case MAE:
		return LossFunction{
			Type:          MAE,
			Calculate:     maeCalculate,
			Gradient:      maeGradient,
			GradientBatch: maeGradientBatch,
		}
	default:
		// Default to MSE
		return LossFunction{
			Type:          MSE,
			Calculate:     mseCalculate,
			Gradient:      mseGradient,
			GradientBatch: mseGradientBatch,
		}
	}
}
