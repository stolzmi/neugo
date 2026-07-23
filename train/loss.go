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

// Huber is quadratic for small errors (|diff| <= Delta) and linear beyond
// that, making it less sensitive to outliers than MSE while staying
// differentiable everywhere (unlike MAE at diff=0). Normalized like
// MSE/MAE — by total element count, not batch size.
type Huber struct {
	Delta float32
}

func HuberLoss(delta float32) *Huber { return &Huber{Delta: delta} }

// SmoothL1Loss is Huber with Delta=1, the special case PyTorch names
// SmoothL1Loss.
func SmoothL1Loss() *Huber { return HuberLoss(1.0) }

func (h *Huber) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(len(pred.Data))
	delta := h.Delta
	var sum float32
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		diff := pred.Data[i] - target.Data[i]
		absDiff := float32(math.Abs(float64(diff)))
		if absDiff <= delta {
			sum += 0.5 * diff * diff
			grad.Data[i] = diff / n
		} else {
			sum += delta * (absDiff - 0.5*delta)
			if diff > 0 {
				grad.Data[i] = delta / n
			} else {
				grad.Data[i] = -delta / n
			}
		}
	}
	return sum / n, grad, nil
}

// KLDivergence computes the batch-mean KL divergence KL(target || pred),
// treating both pred and target as probability distributions over the
// last dimension (same [batch, classes] convention as CrossEntropyLoss,
// and likewise expecting pred already normalized — apply nn.Softmax
// first if pred is raw logits).
type KLDivergence struct{}

func KLDivergenceLoss() *KLDivergence { return &KLDivergence{} }

func (l *KLDivergence) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	if len(pred.Shape) != 2 {
		return 0, nil, fmt.Errorf("train: KLDivergence expects [batch, classes], got %v", pred.Shape)
	}
	batch := pred.Shape[0]
	invBatch := 1.0 / float32(batch)
	var sum float64
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		p := clip32(pred.Data[i])
		y := target.Data[i]
		if y > 0 {
			sum += float64(y) * (math.Log(float64(y)) - math.Log(float64(p)))
		}
		grad.Data[i] = -y / p * invBatch
	}
	return float32(sum) * invBatch, grad, nil
}

// Hinge is the standard binary/margin classification loss max(0, 1 -
// target*pred). It expects target values in {-1, +1} (not BCE's {0, 1}),
// following the classical hinge-loss convention. Normalized like
// MSE/MAE/Huber — by total element count.
type Hinge struct{}

func HingeLoss() *Hinge { return &Hinge{} }

func (h *Hinge) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(len(pred.Data))
	var sum float32
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		margin := 1 - target.Data[i]*pred.Data[i]
		if margin > 0 {
			sum += margin
			grad.Data[i] = -target.Data[i] / n
		}
	}
	return sum / n, grad, nil
}

// Focal (Lin et al., 2017, "Focal Loss for Dense Object Detection") is a
// BCE variant that down-weights easy (already well-classified) examples
// via (1-p_t)^Gamma, focusing training on hard examples; Alpha is the
// usual class-balancing weight (weight for the positive class, 1-Alpha
// for the negative). Gamma=0, Alpha=0.5 recovers (unweighted) BCE.
// Expects pred as post-sigmoid probabilities and target in {0, 1}, same
// convention and batch-size normalization as BinaryCrossEntropy.
type Focal struct {
	Gamma, Alpha float32
}

func FocalLoss(gamma, alpha float32) *Focal { return &Focal{Gamma: gamma, Alpha: alpha} }

func (f *Focal) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	n := float32(pred.Shape[0])
	var sum float64
	grad := nn.NewTensor(pred.Shape)
	for i := range pred.Data {
		p := clip32(pred.Data[i])
		y := target.Data[i]
		// pt/alphat collapse the y=1/y=0 cases into one continuous
		// formula: pt = p and alphat = Alpha when y=1, pt = 1-p and
		// alphat = 1-Alpha when y=0.
		pt := clip32(y*p + (1-y)*(1-p))
		alphat := y*f.Alpha + (1-y)*(1-f.Alpha)
		logPt := math.Log(float64(pt))
		sum += float64(alphat) * math.Pow(float64(pt), float64(f.Gamma)) * -logPt

		// d/dp of -alphat*pt^Gamma*ln(pt), with d(pt)/dp = +1 (y=1) or -1
		// (y=0), collapses to sign=(1-2y) times a shared magnitude term.
		sign := 1 - 2*y
		ptPowGm1 := float32(math.Pow(float64(pt), float64(f.Gamma-1)))
		grad.Data[i] = sign * alphat * ptPowGm1 * (f.Gamma*float32(logPt) + 1) / n
	}
	return float32(sum) / n, grad, nil
}

// CosineSimilarity computes the batch-mean of 1-cos_sim(pred_row,
// target_row) over [batch, features] rows — a common loss for embedding/
// metric-learning tasks where absolute magnitude shouldn't matter.
type CosineSimilarity struct{}

func CosineSimilarityLoss() *CosineSimilarity { return &CosineSimilarity{} }

func (c *CosineSimilarity) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	if err := sameShape(pred, target); err != nil {
		return 0, nil, err
	}
	if len(pred.Shape) != 2 {
		return 0, nil, fmt.Errorf("train: CosineSimilarity expects [batch, features], got %v", pred.Shape)
	}
	batch, features := pred.Shape[0], pred.Shape[1]
	grad := nn.NewTensor(pred.Shape)
	var sum float32
	for b := 0; b < batch; b++ {
		base := b * features
		var dot, predNormSq, targetNormSq float32
		for c := 0; c < features; c++ {
			pv, tv := pred.Data[base+c], target.Data[base+c]
			dot += pv * tv
			predNormSq += pv * pv
			targetNormSq += tv * tv
		}
		predNorm := float32(math.Sqrt(float64(predNormSq))) + lossEpsilon
		targetNorm := float32(math.Sqrt(float64(targetNormSq))) + lossEpsilon
		cosSim := dot / (predNorm * targetNorm)
		sum += 1 - cosSim
		for c := 0; c < features; c++ {
			pv, tv := pred.Data[base+c], target.Data[base+c]
			grad.Data[base+c] = (-tv/(predNorm*targetNorm) + dot*pv/(predNorm*predNorm*predNorm*targetNorm)) / float32(batch)
		}
	}
	return sum / float32(batch), grad, nil
}

// LabelSmoothing wraps another Loss, replacing hard targets with a
// smoothed distribution (Szegedy et al., 2016) before delegating: each
// class's target probability becomes target*(1-Epsilon) + Epsilon/K,
// which for a one-hot target puts (1-Epsilon)+Epsilon/K on the true class
// and Epsilon/K everywhere else, discouraging overconfident predictions.
// Works with any inner Loss (CrossEntropyLoss, BCELoss, ...); when
// wrapping a *CrossEntropyLoss that's meant to use the fused softmax
// shortcut, set that up (ce.SetFused(true)) before wrapping — Trainer.New
// only auto-detects fusion on a *CrossEntropyLoss passed to it directly.
type LabelSmoothing struct {
	inner      Loss
	Epsilon    float32
	NumClasses int
}

func LabelSmoothingLoss(inner Loss, epsilon float32, numClasses int) *LabelSmoothing {
	return &LabelSmoothing{inner: inner, Epsilon: epsilon, NumClasses: numClasses}
}

func (l *LabelSmoothing) Loss(pred, target *nn.Tensor) (float32, *nn.Tensor, error) {
	uniform := l.Epsilon / float32(l.NumClasses)
	scale := 1 - l.Epsilon
	smoothed := nn.NewTensor(target.Shape)
	for i, y := range target.Data {
		smoothed.Data[i] = y*scale + uniform
	}
	return l.inner.Loss(pred, smoothed)
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
