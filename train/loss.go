package train

import (
	"fmt"
	"math"
	"github.com/stolzmi/neugo/nn"
)

// Loss returns scalar loss (batch-mean) and dLoss/dPred.
type Loss interface {
	Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error)
}

const lossEpsilon = 1e-7

func clip32(p float32) float32 {
	if p < lossEpsilon {
		return lossEpsilon
	}
	if p > 1-lossEpsilon {
		return 1 - lossEpsilon
	}
	return p
}

func sameShape(a, b *nn.Tensor) error {
	if len(a.Shape) != len(b.Shape) {
		return fmt.Errorf("train: shape mismatch %v vs %v", a.Shape, b.Shape)
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return fmt.Errorf("train: shape mismatch %v vs %v", a.Shape, b.Shape)
		}
	}
	return nil
}

type MeanSquaredError struct{}

func MSELoss() *MeanSquaredError { return &MeanSquaredError{} }

func (l *MeanSquaredError) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(len(pred.Data))
	var sum float32
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		diff := pred.Data[i] - target.Data[i]
		sum += diff * diff
		grad.Data[i] = 2 * diff / n
	}
	return sum / n, grad, nil
}

type BinaryCrossEntropy struct{}

func BCELoss() *BinaryCrossEntropy { return &BinaryCrossEntropy{} }

func (l *BinaryCrossEntropy) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(pred.Shape[0])
	var sum float64
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		p := clip32(pred.Data[i])
		y := target.Data[i]
		sum += -(float64(y)*math.Log(float64(p)) + float64(1-y)*math.Log(float64(1-p)))
		grad.Data[i] = (-(y/p) + (1-y)/(1-p)) / n
	}
	return float32(sum) / n, grad, nil
}

// CrossEntropyLoss computes categorical cross-entropy. When fused is true,
// pred is assumed to already be softmax probabilities (the model's last
// module is a *nn.SoftmaxModule, detected and set by train.New) and the
// gradient (probs-target)/batch is meant to bypass that module's own
// Backward. When fused is false, pred is raw logits and CrossEntropyLoss
// applies softmax internally before computing the identical gradient
// formula — see design decision #3 in the plan header.
type CrossEntropyLoss struct {
	fused bool
}

func CrossEntropy() *CrossEntropyLoss { return &CrossEntropyLoss{} }

func (l *CrossEntropyLoss) SetFused(fused bool) { l.fused = fused }
func (l *CrossEntropyLoss) Fused() bool         { return l.fused }

func (l *CrossEntropyLoss) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	if len(pred.Shape) != 2 {
		return 0, nil, fmt.Errorf("train: CrossEntropy expects [batch, classes], got %v", pred.Shape)
	}
	batch, classes := pred.Shape[0], pred.Shape[1]

	probs := pred.Data
	if !l.fused {
		sm := nn.Softmax()
		out, err := sm.Forward(&nn.Context{}, pred)
		if err != nil {
			return 0, nil, err
		}
		probs = out.Data
	}

	var sum float64
	for i, p := range probs {
		if target.Data[i] > 0 {
			sum += float64(target.Data[i]) * math.Log(float64(clip32(p)))
		}
	}
	loss := float32(-sum / float64(batch))

	grad := nn.NewTensor(pred.Shape)
	invBatch := 1.0 / float32(batch)
	for i := range probs {
		grad.Data[i] = (probs[i] - target.Data[i]) * invBatch
	}
	_ = classes
	return loss, grad, nil
}

type MeanAbsoluteError struct{}

func MAELoss() *MeanAbsoluteError { return &MeanAbsoluteError{} }

func (l *MeanAbsoluteError) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(len(pred.Data))
	var sum float32
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		diff := pred.Data[i] - target.Data[i]
		sum += float32(math.Abs(float64(diff)))
		switch {
		case diff > 0:
			grad.Data[i] = 1 / n
		case diff < 0:
			grad.Data[i] = -1 / n
		default:
			grad.Data[i] = 0
		}
	}
	return sum / n, grad, nil
}
